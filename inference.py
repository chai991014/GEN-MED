import pandas as pd
import evaluate
import os
import sys
import time
import string
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from llm_adapter import get_llm_adapter
from rag_pipeline import RAGPipeline
from retriever import MultimodalRetriever

# ==========================================
# 0. CONFIGURATION
# ==========================================
CONFIG = {
    "TEST_MODE": True,  # True = Run 5 samples only
    "USE_RAG": True,  # Toggle RAG on/off
    "USE_RERANKER": True,  # Toggle Rerank on/off

    # "MODEL_CHOICE": "microsoft/llava-med-v1.5-mistral-7b",
    # "MODEL_CHOICE": "Qwen/Qwen2-VL-2B-Instruct",
    # "MODEL_CHOICE": "Qwen/Qwen2-VL-7B-Instruct",  # [OOM]
    # "MODEL_CHOICE": "Qwen/Qwen2.5-VL-3B-Instruct",
    # "MODEL_CHOICE": "Qwen/Qwen2.5-VL-7B-Instruct",  # [OOM]
    "MODEL_CHOICE": "Qwen/Qwen3-VL-4B-Instruct",
    # "MODEL_CHOICE": "Qwen/Qwen3-VL-8B-Instruct",  # [OOM]

    "DATASET_ID": "flaviagiammarino/vqa-rad",
    "LLAVA_REPO_PATH": os.path.abspath("./LLaVA-Med"),

    # RAG Settings
    "RAG_K": 2,  # Number of exemplars to retrieve
    # "RERANKER_MODEL": "BAAI/bge-reranker-base",
    "RERANKER_MODEL": "ncbi/MedCPT-Cross-Encoder",
    "RERANK_K": 20,  # Number of exemplars to retrieve for rerank
    "RAG_ALPHA": 0.5,  # Alpha weight Text similarity and 1-Alpha weight Image similarity
}


# ==========================================
# 1. LOGGING & UTILS
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


def normalize_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()


# Setup paths
if CONFIG["USE_RERANKER"]:
    tech_tag = "Rerank_RAG"
elif CONFIG["USE_RAG"]:
    tech_tag = "RAG"
else:
    tech_tag = "ZeroShot"

model_map = {
    "microsoft/llava-med-v1.5-mistral-7b": "LLaVA-Med",
    "Qwen/Qwen2-VL-2B-Instruct": "Qwen2-2B",
    "Qwen/Qwen2-VL-7B-Instruct": "Qwen2-7B",
    "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen2.5-3B",
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-7B",
    "Qwen/Qwen3-VL-4B-Instruct": "Qwen3-4B",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-8B",
}
model_short = model_map.get(CONFIG["MODEL_CHOICE"], CONFIG["MODEL_CHOICE"].replace("/", "_"))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "./result"
os.makedirs(output_dir, exist_ok=True)

if CONFIG["TEST_MODE"]:
    base_name = f"Test5_{tech_tag}_{model_short}_{timestamp}"
else:
    base_name = f"{tech_tag}_{model_short}_{timestamp}"

OUTPUT_FILE = f"{output_dir}/results_{base_name}.csv"
LOG_FILE = f"{output_dir}/log_{base_name}.txt"

sys.stdout = Logger(LOG_FILE)
print("\n" + "="*40)
print("⚙️ SYSTEM CONFIGURATION")
print("="*40)
print(f"   [General]")
print(f"   • Test Mode:      {CONFIG['TEST_MODE']}")
print(f"   • Dataset:        {CONFIG['DATASET_ID']}")
print(f"   • Model:          {CONFIG['MODEL_CHOICE']}")
print(f"\n   [RAG Pipeline]")
print(f"   • Enabled:        {CONFIG['USE_RAG']}")
print(f"   • Retrieval K:    {CONFIG['RAG_K']}")
print(f"   • Alpha:          {CONFIG['RAG_ALPHA']}")
print(f"   • Reranker:       {CONFIG['USE_RERANKER']}")
print(f"   • Reranker Model: {CONFIG['RERANKER_MODEL'] if CONFIG['USE_RERANKER'] else 'N/A'}")
print(f"   • Reranker K:     {CONFIG['RERANK_K'] if CONFIG['USE_RERANKER'] else 'N/A'}")
print("="*40 + "\n")


# ==========================================
# 2. PIPELINE INITIALIZATION
# ==========================================
# A. Prepare LLM Params
llm_params = {}
if "llava" in CONFIG["MODEL_CHOICE"].lower():
    llm_params = {"repo_path": CONFIG["LLAVA_REPO_PATH"], "model_path": CONFIG["MODEL_CHOICE"]}
elif "qwen" in CONFIG["MODEL_CHOICE"].lower():
    llm_params = {"model_id": CONFIG["MODEL_CHOICE"]}

# B. Load LLM Adapter
llm = get_llm_adapter(CONFIG["MODEL_CHOICE"], **llm_params)
llm.load()

# C. Configure Execution Pipeline
if CONFIG["USE_RAG"]:
    print("\n🔍 Initializing RAG Pipeline...")
    retriever_engine = MultimodalRetriever(device="cpu")

    print("📂 Loading Knowledge Base (Train Split)...")
    train_dataset = load_dataset(CONFIG["DATASET_ID"], split="train")

    # Wrap LLM with RAG Pipeline
    if CONFIG["USE_RERANKER"]:
        inference_engine = RAGPipeline(
            llm,
            retriever_engine,
            k=CONFIG["RAG_K"],
            alpha=CONFIG["RAG_ALPHA"],
            use_reranker=CONFIG["USE_RERANKER"],
            reranker_model=CONFIG["RERANKER_MODEL"],
            rerank_k=CONFIG["RERANK_K"],
            device="cpu"
        )
    else:
        inference_engine = RAGPipeline(
            llm,
            retriever_engine,
            k=CONFIG["RAG_K"],
            alpha=CONFIG["RAG_ALPHA"]
        )
    inference_engine.build_index(train_dataset)
else:
    print("\n🛡️ Using Standard Pipeline (No RAG)...")
    inference_engine = llm


# ==========================================
# 3. INFERENCE LOOP
# ==========================================
print("\n📂 Loading Test Dataset...")
dataset = load_dataset(CONFIG["DATASET_ID"], split="test")

if CONFIG["TEST_MODE"]:
    dataset = dataset.select(range(5))
    print("⚠️ WARNING: Running in Test Mode (5 samples only)")

# Load Metrics
bert_metric = evaluate.load("bertscore")
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

results = []
print(f"▶️ Starting inference on {len(dataset)} samples...")

start_time = time.time()

for i, item in tqdm(enumerate(dataset), total=len(dataset)):
    try:
        image = item['image'].convert("RGB")
        question = item['question']
        gt = str(item['answer'])

        # 1. Generate
        raw_pred = inference_engine.generate(image, question)

        # 2. Normalize & Categorize
        clean_gt = normalize_text(gt)
        is_closed = clean_gt in ['yes', 'no']
        final_pred = raw_pred

        # 3. Apply Judge if needed (Closed-Ended Only)
        if is_closed:
            normalized_raw = normalize_text(raw_pred)
            if normalized_raw not in ['yes', 'no']:
                final_pred = inference_engine.judge_answer(image, question, raw_pred)

        # 4. Score
        clean_final = normalize_text(final_pred)
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
# 4. METRICS & REPORTS
# ==========================================
df = pd.DataFrame(results)

# Accuracy
closed_acc = df[df['type'] == 'CLOSED']['exact_match'].mean() * 100 if not df[df['type'] == 'CLOSED'].empty else 0.0

# NLP Metrics
open_df = df[df['type'] == 'OPEN']
if not open_df.empty:
    print("Computing NLP Metrics...")
    preds = list(open_df['final_prediction'])
    refs = list(open_df['ground_truth'])

    bert_f1 = bert_metric.compute(predictions=preds, references=refs, lang="en")['f1']
    bert_score = sum(bert_f1) / len(bert_f1) * 100

    rouge_score = rouge_metric.compute(predictions=preds, references=refs)['rougeL'] * 100

    bleu_output = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])
    bleu_score = bleu_output['precisions'][0] * 100
else:
    bert_score = rouge_score = bleu_score = 0.0

# Final Console Report
print("\n" + "=" * 40)
print(f"✅ FINAL RESULTS: {tech_tag} - {CONFIG['MODEL_CHOICE']}")
print("-" * 40)
print(f"Closed Accuracy: {closed_acc:.2f}%")
print(f"Open BERTScore:  {bert_score:.2f}")
print(f"Open ROUGE-L:    {rouge_score:.2f}")
print(f"Open BLEU-1:     {bleu_score:.2f}")
print("-" * 40)
print(f"Total Time:      {total_time:.2f} sec")
print(f"Avg Time/Sample: {avg_time:.2f} sec")
print("=" * 40)

# Save Files
df.to_csv(OUTPUT_FILE, index=False)
print(f"📄 Predictions saved: {OUTPUT_FILE}")
