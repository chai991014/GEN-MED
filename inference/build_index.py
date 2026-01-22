import os
from datasets import load_dataset
from inference.retriever import MultimodalRetriever

# Configuration
DATASET_ID = "flaviagiammarino/vqa-rad"
CACHE_DIR = "./rag_cache"


def main():
    print(f"🚀 Starting Offline Index Builder...")

    # 1. Initialize Retriever (loads BioMedCLIP)
    # Note: This will use GPU if available. If OOM occurs HERE,
    # set device="cpu" in the constructor below.
    retriever = MultimodalRetriever(
        device="cuda",
        cache_dir=CACHE_DIR
    )

    # 2. Load Dataset
    print(f"📂 Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train")

    # 3. Build & Save Index
    # This function automatically checks for cache, builds if missing, and saves to disk.
    print("🏗️ Building Index (this may take a while)...")
    dataset = dataset.add_column("idx", range(len(dataset)))
    retriever.build_index(dataset)

    print("✅ Index built and saved successfully!")
    print(f"Location: {os.path.abspath(CACHE_DIR)}")


if __name__ == "__main__":
    main()
