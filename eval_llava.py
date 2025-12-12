import torch
import pandas as pd
import evaluate
import sys
import os
import string
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
# CHANGE THIS TO "LLAVA" OR "QWEN"
MODEL_CHOICE = "QWEN"

# Paths
DATASET_ID = "flaviagiammarino/vqa-rad"
OUTPUT_FILE = f"results_{MODEL_CHOICE.lower()}.csv"

# Path to the cloned folder (Relative to this script)
LLAVA_REPO_PATH = os.path.abspath("./LLaVA-Med")
# Path to the model weights (Hugging Face ID or local path)
LLAVA_MODEL_PATH = "microsoft/llava-med-v1.5-mistral-7b"
QWEN_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"


# ==========================================
# 2. MODEL HANDLERS
# ==========================================

class VQAModel:
    def load(self): raise NotImplementedError

    def generate(self, image, question): raise NotImplementedError

    def judge_answer(self, image, question, raw_answer): raise NotImplementedError


class LLavaHandler(VQAModel):
    def load(self):
        print(f"🚀 Loading LLaVA-Med from {LLAVA_MODEL_PATH}...")

        # --- CRITICAL: Add local repo to Python path ---
        if LLAVA_REPO_PATH not in sys.path:
            sys.path.append(LLAVA_REPO_PATH)

        try:
            # Now we can import from the folder we just cloned!
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

            # Load with 4-bit quantization for RTX 3060
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=LLAVA_MODEL_PATH,
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
            print(f"Could not find 'llava' module. Ensure '{LLAVA_REPO_PATH}' exists.")
            sys.exit(1)

    def _run_inference(self, image, prompt, max_tokens=128):
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.idx,
                                               return_tensors='pt').unsqueeze(0).cuda()

        # Manual attention mask to fix warnings
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
        qs = self.tok_img + "\n" + question
        prompt = f"USER: {qs}\nASSISTANT:"
        return self._run_inference(image, prompt, max_tokens=128)

    def judge_answer(self, image, question, raw_answer):
        judge_q = (
            f"The original question was: '{question}'\n"
            f"The answer given was: '{raw_answer}'\n"
            "Does this answer imply Yes or No? Answer with a single word."
        )
        qs = self.tok_img + "\n" + judge_q
        prompt = f"USER: {qs}\nASSISTANT:"
        return self._run_inference(image, prompt, max_tokens=5)


class QwenHandler(VQAModel):
    def load(self):
        print(f"🚀 Loading Qwen2-VL from {QWEN_MODEL_ID}...")
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID, quantization_config=bnb_config, device_map="auto"
        )
        # Higher resolution settings
        self.processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, min_pixels=256 * 256, max_pixels=1280 * 1280)
        print("✅ Qwen2-VL loaded!")

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
        return self._run_inference(image, question, max_tokens=128)

    def judge_answer(self, image, question, raw_answer):
        judge_q = (
            f"Question: {question}\n"
            f"Answer: {raw_answer}\n"
            "Does this answer mean Yes or No? Answer with one word."
        )
        return self._run_inference(image, judge_q, max_tokens=5)


# ==========================================
# 3. EXECUTION LOGIC
# ==========================================
if MODEL_CHOICE == "LLAVA":
    handler = LLavaHandler()
elif MODEL_CHOICE == "QWEN":
    handler = QwenHandler()
else:
    raise ValueError("Set MODEL_CHOICE to 'LLAVA' or 'QWEN'")

handler.load()

# ==========================================
# 4. EVALUATION LOOP
# ==========================================
print("📂 Loading Dataset...")
dataset = load_dataset(DATASET_ID, split="test")

# dataset = dataset.select(range(20))
# print("⚠️ WARNING: Running in Test Mode (20 samples only)")

bert_metric = evaluate.load("bertscore")
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

results = []
print(f"▶️  Starting inference on {len(dataset)} samples...")


def normalize(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()


for i, item in tqdm(enumerate(dataset), total=len(dataset)):
    try:
        image = item['image'].convert("RGB")
        question = item['question']
        gt = str(item['answer'])

        # 1. Generate Raw Answer
        raw_pred = handler.generate(image, question)

        # 2. Determine Question Type
        clean_gt = normalize(gt)
        is_closed = clean_gt in ['yes', 'no']

        final_pred = raw_pred

        if is_closed:
            # If the answer isn't a simple "yes" or "no", ask the judge!
            normalized_raw = normalize(raw_pred)
            if normalized_raw not in ['yes', 'no']:
                # The model is rambling. Use Judge to fix it.
                judge_output = handler.judge_answer(image, question, raw_pred)
                final_pred = judge_output

                # 3. Scoring
        clean_final = normalize(final_pred)

        if is_closed:
            # Strict match on the JUDGED answer
            score = 1 if clean_final == clean_gt else 0
        else:
            # Open ended exact match (BERTScore will cover semantics later)
            score = 1 if clean_final == clean_gt else 0

        results.append({
            "id": i,
            "question": question,
            "type": 'CLOSED' if is_closed else 'OPEN',
            "ground_truth": gt,
            "raw_prediction": raw_pred,
            "final_prediction": final_pred,  # Save the fixed answer
            "exact_match": score
        })
    except Exception as e:
        print(f"⚠️ Error on sample {i}: {e}")
        continue

# ==========================================
# 5. METRICS & SAVING
# ==========================================
df = pd.DataFrame(results)

# 1. Calculate Closed Accuracy
if not df[df['type'] == 'CLOSED'].empty:
    closed_acc = df[df['type'] == 'CLOSED']['exact_match'].mean() * 100
else:
    closed_acc = 0.0

# 2. Calculate Open Metrics
open_df = df[df['type'] == 'OPEN']
if not open_df.empty:
    preds = list(open_df['final_prediction'])  # Use FINAL prediction
    refs = list(open_df['ground_truth'])
    print("Computing NLP Metrics...")

    # BERTScore
    bert_f1 = bert_metric.compute(predictions=preds, references=refs, lang="en")['f1']
    bert_score = sum(bert_f1) / len(bert_f1) * 100

    # ROUGE
    rouge_score = rouge_metric.compute(predictions=preds, references=refs)['rougeL'] * 100

    # BLEU (Requires references to be list of lists)
    bleu_output = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])
    bleu_score = bleu_output['precisions'][0] * 100
else:
    bert_score = rouge_score = bleu_score = 0.0

# 3. Print to Terminal
print("\n" + "=" * 40)
print(f"✅ FINAL RESULTS: {MODEL_CHOICE} + JUDGE")
print("-" * 40)
print(f"Closed Accuracy: {closed_acc:.2f}%")
print(f"Open BERTScore:  {bert_score:.2f}")
print(f"Open ROUGE-L:    {rouge_score:.2f}")
print(f"Open BLEU-1:     {bleu_score:.2f}")
print("=" * 40)

# 4. Save Detailed CSV
df.to_csv(OUTPUT_FILE, index=False)
print(f"📄 Detailed predictions saved to {OUTPUT_FILE}")

# 5. Save Summary Text File (NEW)
summary_file = OUTPUT_FILE.replace(".csv", "_summary.txt")
with open(summary_file, "w") as f:
    f.write(f"Model: {MODEL_CHOICE} + LLM-Judge\n")
    f.write("=" * 30 + "\n")
    f.write(f"Closed Accuracy: {closed_acc:.2f}%\n")
    f.write(f"Open BERTScore:  {bert_score:.2f}\n")
    f.write(f"Open ROUGE-L:    {rouge_score:.2f}\n")
    f.write(f"Open BLEU-1:     {bleu_score:.2f}\n")
print(f"📊 Metrics summary saved to {summary_file}")
