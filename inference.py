import os
import time
from tqdm import tqdm
import pandas as pd
import evaluate
from datasets import load_dataset
from llm_adapter import get_llm_adapter
from rag_pipeline import RAGPipeline
from retriever import MultimodalRetriever
from mevf.adapter import MEVFAdapter
from utils import get_config, normalize_text

# ==========================================
# 1. GET CONFIG
# ==========================================
CONFIG = get_config()
CONFIG["LLAVA_REPO_PATH"] = os.path.abspath("./LLaVA-Med")

# ==========================================
# 2. PIPELINE INITIALIZATION
# ==========================================
inference_engine = None

if CONFIG["MODEL_TYPE"] == "MEVF":
    print(f"""🧠 Initializing Custom {CONFIG["TECH_TAG"]} Adapter...""")
    inference_engine = MEVFAdapter(
        model_path=CONFIG["MEVF_WEIGHTS"],
        maml_path=CONFIG["MAML_WEIGHTS"],
        ae_path=CONFIG["AE_WEIGHTS"],
        reasoning_model=CONFIG["REASONING_MODEL"]
    )
else:
    # A. Prepare LLM Params
    llm_params = {}
    if "llava" in CONFIG["MODEL_TYPE"].lower():
        llm_params = {
            "repo_path": CONFIG["LLAVA_REPO_PATH"],
            "model_path": CONFIG["MODEL_TYPE"],
            "prompt": CONFIG["PROMPT"]
        }
    elif "qwen" in CONFIG["MODEL_TYPE"].lower():
        llm_params = {
            "model_id": CONFIG["MODEL_TYPE"],
            "prompt": CONFIG["PROMPT"]
        }

    # B. Load LLM Adapter
    llm = get_llm_adapter(CONFIG["MODEL_TYPE"], **llm_params)
    llm.load()

    # C. Configure Execution Pipeline
    if CONFIG["USE_RAG"]:
        print(f"""\n🔍 Initializing RAG Pipeline ({CONFIG["TECH_TAG"]})...""")
        retriever_engine = MultimodalRetriever(device="cpu")

        print("📂 Loading Knowledge Base (Train Split)...")
        train_dataset = load_dataset(CONFIG["DATASET_ID"], split="train")

        # Wrap LLM with RAG Pipeline
        inference_engine = RAGPipeline(
            llm,
            retriever_engine,
            k=CONFIG["RAG_K"],
            alpha=CONFIG["RAG_ALPHA"],
        )

        inference_engine.build_index(train_dataset)
    else:
        print(f"""\n🛡️ Using Standard Pipeline ({CONFIG["TECH_TAG"]})...""")
        inference_engine = llm


# ==========================================
# 3. INFERENCE LOOP
# ==========================================
print("\n📂 Loading Test Dataset...")
dataset = load_dataset(CONFIG["DATASET_ID"], split=CONFIG["DATASET"])

if CONFIG["TEST_MODE"]:
    dataset = dataset.select(range(20))
    print("⚠️ WARNING: Running in Test Mode (20 samples only)")

# Load Metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
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

        results.append({
            "id": i,
            "question": question,
            "type": 'CLOSED' if is_closed else 'OPEN',
            "ground_truth": gt,
            "raw_prediction": raw_pred
        })

    except Exception as e:
        print(f"⚠️ Error on sample {i}: {e}")
        continue

end_time = time.time()
total_time = end_time - start_time
avg_time = total_time / len(dataset) if len(dataset) > 0 else 0


# ==========================================
# 4. SUMMARY
# ==========================================
df = pd.DataFrame(results)

print("\n" + "=" * 60)
print(f"   [Performance]")
print(f"   • Total Time:         {total_time:.2f} sec")
print(f"   • Avg Time/Inference: {avg_time:.2f} sec")
print("=" * 60)

# Save Files
df.to_csv(CONFIG["OUTPUT_FILE"], index=False)
print(f"""📄 Predictions saved: {CONFIG["OUTPUT_FILE"]}""")
