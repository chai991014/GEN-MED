import torch
import sys


def get_inference_text(question):
    """Unified logic for inference input."""
    return question


def get_judge_text(question, raw_answer):
    """Unified logic for judge input."""
    return (
        f"Question: {question}\n"
        f"Answer: {raw_answer}\n"
        "Does this answer mean Yes or No? Answer with one word."
    )


class VQAModel:
    def load(self): raise NotImplementedError
    def generate(self, image, question): raise NotImplementedError
    def judge_answer(self, image, question, raw_answer): raise NotImplementedError


class LLavaHandler(VQAModel):
    def __init__(self, repo_path, model_path):
        self.repo_path = repo_path
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.idx = None
        self.tok_img = None

    def load(self):
        print(f"🚀 Loading LLaVA-Med from {self.model_path}...")

        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=self.model_path,
                model_base=None,
                model_name="llava_mistral",
                load_4bit=True,
                device="cuda"
            )
            self.tokenizer_image_token = tokenizer_image_token
            self.idx = IMAGE_TOKEN_INDEX
            self.tok_img = DEFAULT_IMAGE_TOKEN
            print("✅ LLaVA-Med loaded via Source!")

        except ImportError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    def _run_inference(self, image, prompt, max_tokens=128):
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.idx,
                                               return_tensors='pt').unsqueeze(0).cuda()
        attention_mask = torch.ones_like(input_ids, device="cuda")

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                max_new_tokens=max_tokens
            )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def generate(self, image, question):
        text = get_inference_text(question)
        qs = self.tok_img + "\n" + text
        prompt = f"USER: {qs}\nASSISTANT:"
        return self._run_inference(image, prompt, max_tokens=128)

    def judge_answer(self, image, question, raw_answer):
        judge_q = get_judge_text(question, raw_answer)
        qs = self.tok_img + "\n" + judge_q
        prompt = f"USER: {qs}\nASSISTANT:"
        return self._run_inference(image, prompt, max_tokens=5)


class QwenHandler(VQAModel):
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load(self):
        print(f"🚀 Loading Qwen from {self.model_id}...")
        from transformers import AutoProcessor, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
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

    def generate(self, image, question):
        text = get_inference_text(question)
        return self._run_inference(image, text, max_tokens=128)

    def judge_answer(self, image, question, raw_answer):
        judge_q = get_judge_text(question, raw_answer)
        return self._run_inference(image, judge_q, max_tokens=5)


def get_llm_handler(model_choice, **kwargs):
    model_name = model_choice.lower()
    if "llava" in model_name:
        return LLavaHandler(
            repo_path=kwargs.get('repo_path'),
            model_path=kwargs.get('model_path', model_choice)
        )
    elif "qwen" in model_name:
        return QwenHandler(
            model_id=kwargs.get('model_id', model_choice)
        )
    else:
        raise ValueError(f"Unknown Model Family for input: {model_choice}")
