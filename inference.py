import pandas as pd
import evaluate
import os
import time
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from llm_adapter import get_llm_adapter
from rag_pipeline import RAGPipeline
from retriever import MultimodalRetriever
from reranker import Reranker
from utils import normalize_text, setup_logger, print_system_config, print_final_report


# ==========================================
# 0. CONFIGURATION
# ==========================================
CONFIG = {
    "TEST_MODE": True,      # Run 5 samples only
    "USE_RAG": False,        # Toggle RAG
    "USE_RERANKER": False,   # Toggle Rerank
    "USE_REFLEXION": False,  # Toggle Reflexion Thinking
    # "USE_AMANDA": False,     # Toggle AMANDA Multi-Agent RAG Framework

    "MODEL_CHOICE": "microsoft/llava-med-v1.5-mistral-7b",
    # "MODEL_CHOICE": "Qwen/Qwen2-VL-2B-Instruct",
    # "MODEL_CHOICE": "Qwen/Qwen2-VL-7B-Instruct",  # [OOM]
    # "MODEL_CHOICE": "Qwen/Qwen2.5-VL-3B-Instruct",
    # "MODEL_CHOICE": "Qwen/Qwen2.5-VL-7B-Instruct",  # [OOM]
    # "MODEL_CHOICE": "Qwen/Qwen3-VL-2B-Instruct",
    # "MODEL_CHOICE": "Qwen/Qwen3-VL-4B-Instruct",
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
# Setup paths
tags = []
if CONFIG["USE_RAG"]:
    tags.append("RAG")
if CONFIG["USE_RERANKER"]:
    tags.append("Rerank")
if CONFIG["USE_REFLEXION"]:
    tags.append("Reflexion")
# if CONFIG["USE_AMANDA"]:
#     tags.append("AMANDA")

if not tags:
    tech_tag = "ZeroShot"
else:
    tech_tag = "+".join(tags)

# if CONFIG["USE_AMANDA"]:
#     CONFIG["USE_RAG"] = True
#     CONFIG["USE_RERANKER"] = True
#     CONFIG["USE_REFLEXION"] = False

model_map = {
    "microsoft/llava-med-v1.5-mistral-7b": "LLaVA-Med",
    "Qwen/Qwen2-VL-2B-Instruct": "Qwen2-2B",
    "Qwen/Qwen2-VL-7B-Instruct": "Qwen2-7B",
    "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen2.5-3B",
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-7B",
    "Qwen/Qwen3-VL-2B-Instruct": "Qwen3-2B",
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

# Setup Logging via Utils
setup_logger(output_dir, base_name)
print_system_config(CONFIG, tech_tag)


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
    print(f"\n🔍 Initializing RAG Pipeline ({tech_tag})...")
    retriever_engine = MultimodalRetriever(device="cpu")

    print("📂 Loading Knowledge Base (Train Split)...")
    train_dataset = load_dataset(CONFIG["DATASET_ID"], split="train")

    if CONFIG["USE_RERANKER"]:
        reranker_engine = Reranker(
            model_id=CONFIG["RERANKER_MODEL"],
            device="cpu"
        )
    else:
        reranker_engine = None

    # if CONFIG["USE_AMANDA"]:
    #     inference_engine = AMANDAPipeline(
    #         llm,
    #         retriever_engine,
    #         reranker_engine=reranker_engine,
    #         k=CONFIG["RAG_K"],
    #         alpha=CONFIG["RAG_ALPHA"],
    #         rerank_k=CONFIG["RERANK_K"],
    #         device="cpu"
    #     )
    # else:
    # Wrap LLM with RAG Pipeline
    inference_engine = RAGPipeline(
        llm,
        retriever_engine,
        reranker_engine=reranker_engine,
        k=CONFIG["RAG_K"],
        alpha=CONFIG["RAG_ALPHA"],
        rerank_k=CONFIG["RERANK_K"],
    )

    inference_engine.build_index(train_dataset)
else:
    print(f"\n🛡️ Using Standard Pipeline ({tech_tag})...")
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
        if CONFIG["USE_REFLEXION"]:
            raw_pred = inference_engine.reflexion_generate(image, question)
        else:
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
    print("📊 Computing NLP Metrics...")
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
print_final_report(
    tech_tag=tech_tag,
    model_choice=CONFIG['MODEL_CHOICE'],
    closed_acc=closed_acc,
    bert_score=bert_score,
    rouge_score=rouge_score,
    bleu_score=bleu_score,
    total_time=total_time,
    avg_time=avg_time
)

# Save Files
df.to_csv(OUTPUT_FILE, index=False)
print(f"📄 Predictions saved: {OUTPUT_FILE}")
