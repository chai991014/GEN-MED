import pandas as pd
import evaluate
import os
import sys
import time
import string
from tqdm import tqdm
from datasets import load_dataset
from llm_handler import get_llm_handler

# ==========================================
# 0. CONFIGURATION SETTING
# ==========================================
TEST_MODE = False

# Choose: "LLAVA", "QWEN"
MODEL_CHOICE = "LLAVA"
DATASET_ID = "flaviagiammarino/vqa-rad"

LLAVA_REPO_PATH = os.path.abspath("./LLaVA-Med")
LLAVA_MODEL_PATH = "microsoft/llava-med-v1.5-mistral-7b"
# QWEN_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
# QWEN_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct" [OOM]
# QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct" [OOM]
QWEN_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
# QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct" [OOM]


# ==========================================
# 1. LOAD CONFIGURATION
# ==========================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


llm_params = {}

if MODEL_CHOICE == "LLAVA":
    active_id = LLAVA_MODEL_PATH
    llm_params = {"repo_path": LLAVA_REPO_PATH, "model_path": LLAVA_MODEL_PATH}

elif MODEL_CHOICE == "QWEN":
    active_id = QWEN_MODEL_ID
    llm_params = {"model_id": QWEN_MODEL_ID}

else:
    raise ValueError("Set MODEL_CHOICE to 'LLAVA' or 'QWEN'")


safe_name = active_id.replace("/", "_")
os.makedirs("./result", exist_ok=True)
if TEST_MODE:
    OUTPUT_FILE = f"./result/Test5_results_{safe_name}.csv"
    LOG_FILE = f"./result/Test5_log_{safe_name}.txt"
else:
    OUTPUT_FILE = f"./result/results_{safe_name}.csv"
    LOG_FILE = f"./result/log_{safe_name}.txt"

sys.stdout = Logger(LOG_FILE)

print(f"Selected Model: {MODEL_CHOICE}")
print(f"Output File: {OUTPUT_FILE}")
print(f"Log File:    {LOG_FILE}")


# ==========================================
# 2. LOAD MODEL
# ==========================================
llm = get_llm_handler(MODEL_CHOICE, **llm_params)
llm.load()


# ==========================================
# 3. EVALUATION LOOP
# ==========================================
print("📂 Loading Dataset...")
dataset = load_dataset(DATASET_ID, split="test")

if TEST_MODE:
    dataset = dataset.select(range(5))
    print("⚠️ WARNING: Running in Test Mode (5 samples only)")

bert_metric = evaluate.load("bertscore")
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

results = []
print(f"▶️  Starting inference on {len(dataset)} samples...")


def normalize(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()


start_time = time.time()
for i, item in tqdm(enumerate(dataset), total=len(dataset)):
    try:
        image = item['image'].convert("RGB")
        question = item['question']
        gt = str(item['answer'])

        # 1. Generate Raw Answer
        raw_pred = llm.generate(image, question)

        # 2. Determine Question Type
        clean_gt = normalize(gt)
        is_closed = clean_gt in ['yes', 'no']
        final_pred = raw_pred

        # 3. Apply Judge if needed (for Closed questions)
        if is_closed:
            normalized_raw = normalize(raw_pred)
            if normalized_raw not in ['yes', 'no']:
                # Model rambled; ask Judge to extract Yes/No
                final_pred = llm.judge_answer(image, question, raw_pred)

        # 4. Scoring
        clean_final = normalize(final_pred)
        if is_closed:
            score = 1 if clean_final == clean_gt else 0
        else:
            # Exact match for open ended (semantics covered by BERTScore)
            score = 1 if clean_final == clean_gt else 0

        results.append({
            "id": i,
            "question": question,
            "type": 'CLOSED' if is_closed else 'OPEN',
            "ground_truth": gt,
            "raw_prediction": raw_pred,
            "final_prediction": final_pred,
            "exact_match": score
        })

    except Exception as e:
        print(f"⚠️ Error on sample {i}: {e}")
        continue
end_time = time.time()
total_time = end_time - start_time
avg_time = total_time / len(dataset) if len(dataset) > 0 else 0


# ==========================================
# 4. METRICS & SAVING
# ==========================================
df = pd.DataFrame(results)

# Accuracy
closed_acc = df[df['type'] == 'CLOSED']['exact_match'].mean() * 100 if not df[df['type'] == 'CLOSED'].empty else 0.0

# NLP Metrics (Open Ended)
open_df = df[df['type'] == 'OPEN']
if not open_df.empty:
    preds = list(open_df['final_prediction'])
    refs = list(open_df['ground_truth'])
    print("Computing NLP Metrics...")

    bert_f1 = bert_metric.compute(predictions=preds, references=refs, lang="en")['f1']
    bert_score = sum(bert_f1) / len(bert_f1) * 100

    rouge_score = rouge_metric.compute(predictions=preds, references=refs)['rougeL'] * 100

    # BLEU-1 (Unigram Precision)
    bleu_output = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])
    bleu_score = bleu_output['precisions'][0] * 100
else:
    bert_score = rouge_score = bleu_score = 0.0

# Console Output
print("\n" + "=" * 40)
print(f"✅ FINAL RESULTS: {MODEL_CHOICE}")
print("-" * 40)
print(f"Closed Accuracy: {closed_acc:.2f}%")
print(f"Open BERTScore:  {bert_score:.2f}")
print(f"Open ROUGE-L:    {rouge_score:.2f}")
print(f"Open BLEU-1:     {bleu_score:.2f}")
print("-" * 40)
print(f"Total Time:      {total_time:.2f} sec")
print(f"Avg Time/Sample: {avg_time:.2f} sec")
print("=" * 40)

# Save CSV
df.to_csv(OUTPUT_FILE, index=False)
print(f"📄 Detailed predictions saved to {OUTPUT_FILE}")

# Save Summary
summary_file = OUTPUT_FILE.replace(".csv", "_summary.txt")
with open(summary_file, "w") as f:
    f.write(f"Model: {MODEL_CHOICE} ({active_id})\n")
    f.write("=" * 30 + "\n")
    f.write(f"Closed Accuracy: {closed_acc:.2f}%\n")
    f.write(f"Open BERTScore:  {bert_score:.2f}\n")
    f.write(f"Open ROUGE-L:    {rouge_score:.2f}\n")
    f.write(f"Open BLEU-1:     {bleu_score:.2f}\n")
    f.write("-" * 30 + "\n")
    f.write(f"Total Time:      {total_time:.2f} sec\n")
    f.write(f"Avg Time/Sample: {avg_time:.2f} sec\n")
print(f"📊 Metrics summary saved to {summary_file}")
