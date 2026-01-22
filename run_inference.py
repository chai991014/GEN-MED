import os
import sys
import time
from tqdm import tqdm
import torch
import gc
import pandas as pd
from datasets import load_dataset, load_from_disk
from inference.llm_adapter import get_llm_adapter
from inference.rag_pipeline import RAGPipeline
from inference.retriever import MultimodalRetriever
from inference.mevf.adapter import MEVFAdapter
from inference.utils import get_config, normalize_text


def load_data_source(path, split):
    """Helper to load from disk if local path exists, else from HF Hub"""
    if os.path.exists(path) and os.path.isdir(path):
        print(f"   (Loading from local disk: {path})")
        # load_from_disk returns a DatasetDict (containing all splits)
        full_ds = load_from_disk(path)
        return full_ds[split]
    else:
        # Standard Hugging Face loading
        return load_dataset(path, split=split)


def run_inference(config_dict):
    # ==========================================
    # 1. GET CONFIG
    # ==========================================
    config = get_config(config_dict)
    config["LLAVA_REPO_PATH"] = os.path.abspath("LLaVA-Med")

    # ==========================================
    # 2. PIPELINE INITIALIZATION
    # ==========================================
    if config["MODEL_TYPE"] == "MEVF":
        print(f"""🧠 Initializing Custom {config["TECH_TAG"]} Adapter...""")
        inference_engine = MEVFAdapter(
            model_path=config["MEVF_WEIGHTS"],
            maml_path=config["MAML_WEIGHTS"],
            ae_path=config["AE_WEIGHTS"],
            reasoning_model=config["REASONING_MODEL"]
        )
    else:
        # A. Prepare LLM Params
        llm_params = {}
        if "llava" in config["MODEL_TYPE"].lower():
            llm_params = {
                "repo_path": config["LLAVA_REPO_PATH"],
                "model_path": config["MODEL_TYPE"],
                "prompt": config["PROMPT"],
                "adapter_path": config["ADAPTER_PATH"]
            }
        elif "qwen" in config["MODEL_TYPE"].lower():
            llm_params = {
                "model_id": config["MODEL_TYPE"],
                "prompt": config["PROMPT"],
                "adapter_path": config["ADAPTER_PATH"]
            }

        # B. Load LLM Adapter
        llm = get_llm_adapter(config["MODEL_TYPE"], **llm_params)
        llm.load()

        # C. Configure Execution Pipeline
        if config["USE_RAG"]:
            print(f"""\n🔍 Initializing RAG Pipeline ({config["TECH_TAG"]})...""")
            retriever_engine = MultimodalRetriever(device="cpu")

            print("📂 Loading Knowledge Base (Train Split)...")
            train_dataset = load_dataset("flaviagiammarino/vqa-rad", split="train")
            train_dataset = train_dataset.add_column("idx", range(len(train_dataset)))

            # Wrap LLM with RAG Pipeline
            inference_engine = RAGPipeline(
                llm,
                retriever_engine,
                k=config["RAG_K"],
                alpha=config["RAG_ALPHA"],
            )

            inference_engine.build_index(train_dataset)
        else:
            print(f"""\n🛡️ Using Standard Pipeline ({config["TECH_TAG"]})...""")
            inference_engine = llm

    # ==========================================
    # 3. INFERENCE LOOP
    # ==========================================
    print("\n📂 Loading Test Dataset...")
    dataset = load_data_source(config["DATASET_ID"], split=config["DATASET"])

    if config["TEST_MODE"]:
        dataset = dataset.select(range(3))
        print("⚠️ WARNING: Running in Test Mode (20 samples only)")

    results = []
    print(f"▶️ Starting inference on {len(dataset)} samples...")

    start_time = time.time()

    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        try:
            image = item['image'].convert("RGB")
            question = item['question']
            gt = str(item['answer'])

            # 1. Generate
            if config["USE_REFLEXION"]:
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
                "raw_prediction": raw_pred.get("prediction"),
                "reflexion_draft": raw_pred.get("reflexion_draft", ""),
                "reflexion_critique": raw_pred.get("reflexion_critique", ""),
                "retrieved_ids": raw_pred.get("retrieved_ids", []),
                "retrieved_scores": raw_pred.get("retrieved_scores", [])
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
    df.to_csv(config["OUTPUT_FILE"], index=False)
    print(f"""📄 Predictions saved: {config["OUTPUT_FILE"]}""")

    if hasattr(sys.stdout, 'terminal'):
        sys.stdout = sys.stdout.terminal

    del inference_engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    from config import CONFIG
    run_inference(CONFIG)
    # if len(sys.argv) < 2:
    #     sys.exit(1)
    #
    # try:
    #     config_idx = int(sys.argv[1])
    #
    #     # Dynamically retrieve the variable CONFIG_i from the config module
    #     cfg = getattr(config, f"CONFIG_{config_idx}")
    #     run_inference(cfg)
    #
    # except (ValueError, AttributeError) as e:
    #     print(f"❌ Error: Could not find CONFIG_{sys.argv[1]} in config.py")
    #     sys.exit(1)
