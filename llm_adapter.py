import torch
import sys
from prompt_template import (
    get_inference_prompt,
    get_instruct_inference_prompt,
    get_reflexion_critique_prompt,
    get_reflexion_critique_context,
    get_reflexion_refine_prompt
)


class BaseVQAAdapter:
    """Abstract Base Class for VQA Model Adapters."""

    def load(self): raise NotImplementedError
    def generate(self, image, question, context=""): raise NotImplementedError

    def reflexion_generate(self, image, question):
        """
        Implements the 'Draft -> Critique -> Refine' loop.
        """
        # 1. Draft
        draft_dict = self.generate(image, question)
        draft_answer = draft_dict["prediction"]

        # 2. Critique
        critique_prompt = get_reflexion_critique_prompt(draft_answer)
        critique_context = get_reflexion_critique_context(question)
        critique_dict = self.generate(image, critique_prompt, context=critique_context)
        critique_response = critique_dict["prediction"]

        # 3. Refine
        refine_prompt = get_reflexion_refine_prompt(question, draft_answer, critique_response)
        final_dict = self.generate(image, refine_prompt)

        return {
            "prediction": final_dict["prediction"],
            "reflexion_draft": draft_answer,
            "reflexion_critique": critique_response
        }


class LLaVAAdapter(BaseVQAAdapter):
    """Adapter for LLaVA-Med models."""

    def __init__(self, repo_path, model_path, prompt, model_base=None):
        self.repo_path = repo_path
        self.model_path = model_path
        self.model_base = model_base
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.idx = None
        self.tok_img = None
        self.prompt = prompt

    def load(self):
        print(f"🚀 Loading LLaVA-Med from {self.model_path}...")

        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=self.model_path,
                model_base=self.model_base,
                model_name="llava_mistral",
                load_4bit=True,
                load_8bit=False,
                device="cuda",
                device_map={"": "cuda"}
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token_id = 0
                self.tokenizer.pad_token = "<unk>"

            print("🔧 Casting Vision Tower to Float16...")
            vision_tower = self.model.get_vision_tower()
            if hasattr(vision_tower, 'to'):
                vision_tower.to(dtype=torch.float16, device="cuda")

            print("🔧 Casting Adapter Weights to Float16...")
            for name, param in self.model.named_parameters():
                if "lora" in name or "dora" in name or "magnitude" in name:
                    param.data = param.data.to(dtype=torch.float16, device="cuda")

            self.tokenizer_image_token = tokenizer_image_token
            self.idx = IMAGE_TOKEN_INDEX
            self.tok_img = DEFAULT_IMAGE_TOKEN
            print("✅ LLaVA-Med loaded via Source!")

        except ImportError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    def _run_inference(self, image, prompt, max_tokens=128):

        # print(f"   [DEBUG] Start Inference...", end=" ", flush=True)
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.idx,
                                               return_tensors='pt').unsqueeze(0).cuda()
        # attention_mask = torch.ones_like(input_ids, device="cuda")

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # print("Generating...", end=" ", flush=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                max_new_tokens=max_tokens,
                use_cache=True
            )
        # print("Decoding...", end=" ", flush=True)
        result = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # print("Done!", flush=True)
        return result

    def generate(self, image, question, context=""):
        if self.prompt == "Basic":
            text = get_inference_prompt(question, context)
        elif self.prompt == "Instruct":
            text = get_instruct_inference_prompt(question, context)
        else:
            text = question
        qs = self.tok_img + "\n" + text
        prompt = f"USER: {qs}\nASSISTANT:"
        prediction = self._run_inference(image, prompt, max_tokens=128)
        return {"prediction": prediction}


class QwenAdapter(BaseVQAAdapter):
    """Adapter for Qwen-VL models."""

    def __init__(self, model_id, prompt, adapter_path=None):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None
        self.prompt = prompt

    def load(self):
        print(f"🚀 Loading Qwen from {self.model_id}...")
        from transformers import AutoProcessor, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        if "Qwen3" in self.model_id:
            from transformers import Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id, quantization_config=bnb_config, device_map="auto"
            )
        elif "Qwen2.5" in self.model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id, quantization_config=bnb_config, device_map="auto"
            )
        else:
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id, quantization_config=bnb_config, device_map="auto"
            )

        if self.adapter_path:
            print(f"🔗 Attaching LoRA Adapter from {self.adapter_path}...")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.processor = AutoProcessor.from_pretrained(self.model_id, min_pixels=256*256, max_pixels=1280*1280)
        print("✅ Qwen loaded!")

    def _run_inference(self, image, text_prompt, max_tokens=128):
        from qwen_vl_utils import process_vision_info
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text_prompt}]}]
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text_input], images=image_inputs, videos=video_inputs, padding=True,
                                return_tensors="pt").to("cuda")

        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

        ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, ids)]
        return self.processor.batch_decode(ids_trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0].strip()

    def generate(self, image, question, context=""):
        if self.prompt == "Basic":
            text = get_inference_prompt(question, context)
        elif self.prompt == "Instruct":
            text = get_instruct_inference_prompt(question, context)
        else:
            text = question
        prediction = self._run_inference(image, text, max_tokens=128)
        return {"prediction": prediction}


def get_llm_adapter(model_type, **kwargs):
    """Factory function to instantiate the correct model adapter."""
    model_name = model_type.lower()
    adapter_path = kwargs.get('adapter_path')
    if "llava" in model_name:
        if adapter_path:
            final_model_path = kwargs.get('adapter_path', adapter_path)
            final_model_base = kwargs.get('model_path', model_type)
        else:
            final_model_path = kwargs.get('model_path', model_type)
            final_model_base = None
        return LLaVAAdapter(
            repo_path=kwargs.get('repo_path'),
            model_path=final_model_path,
            prompt=kwargs.get('prompt', "Basic"),
            model_base=final_model_base
        )
    elif "qwen" in model_name:
        return QwenAdapter(
            model_id=kwargs.get('model_id', model_type),
            prompt=kwargs.get('prompt', "Basic"),
            adapter_path=kwargs.get('adapter_path', adapter_path)
        )
    else:
        raise ValueError(f"Unknown Model Family for input: {model_type}")
