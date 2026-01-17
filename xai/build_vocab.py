import os
import re
import csv
from collections import Counter
from datasets import load_dataset, concatenate_datasets, load_from_disk

# --- CONFIG ---
OUTPUT_CSV = "medical_concepts_stats.csv"
SLAKE_PATH = "../slake_vqa_rad_format"


def load_data():
    print("⏳ Loading VQA-RAD...")
    ds_rad = load_dataset("flaviagiammarino/vqa-rad", split="train")

    # Try loading SLAKE
    try:
        if os.path.exists(SLAKE_PATH):
            print("⏳ Loading SLAKE...")
            ds_slake = load_from_disk(SLAKE_PATH)["train"]

            # Align columns
            keep_cols = ["image", "question", "answer"]
            ds_rad = ds_rad.select_columns(keep_cols)
            ds_slake = ds_slake.select_columns(keep_cols)

            # Merge
            dataset = concatenate_datasets([ds_rad, ds_slake])
            print(f"✅ Combined Dataset: {len(dataset)} items")
        else:
            print("⚠️ SLAKE not found. Using VQA-RAD only.")
            dataset = ds_rad
    except Exception as e:
        print(f"❌ Error merging: {e}")
        dataset = ds_rad

    return dataset


def analyze_and_save(dataset):
    print("📊 Analyzing Vocabulary Statistics...")

    # 1. Define Standard Stop Words (Noise)
    stop_words = {
        'the', 'is', 'are', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
        'this', 'that', 'it', 'there', 'what', 'where', 'how', 'does', 'show', 'demonstrate',
        'image', 'seen', 'located', 'finding', 'abnormality', 'view', 'projection', 'film',
        'patient', 'chest', 'xray', 'x-ray', 'radiograph', 'scan', 'feature', 'sign',
        'side', 'left', 'right', 'upper', 'lower', 'middle', 'zone', 'lobe', 'field',
        'yes', 'no', 'normal', 'abnormal', 'and', 'or', 'but'
    }

    doc_freq = Counter()
    total_docs = len(dataset)

    # 2. Count Frequencies
    for item in dataset:
        text = (str(item['question']) + " " + str(item['answer'])).lower()
        words = set(re.findall(r'\b[a-z]{3,}\b', text))
        for w in words:
            if w not in stop_words:
                doc_freq[w] += 1

    # 3. Prepare Data for CSV
    stats_data = []
    for word, freq in doc_freq.items():
        if freq < 5: continue  # Ignore rare noise

        prevalence = (freq / total_docs) * 100
        stats_data.append({
            "Rank": 0,  # Placeholder
            "Concept": word,
            "Count": freq,
            "Prevalence_Percent": round(prevalence, 2),
            "Status": "Candidate"  # You can manually mark 'Selected' later
        })

    # Sort by Count (High to Low)
    stats_data.sort(key=lambda x: x["Count"], reverse=True)

    # Add Rank
    for i, row in enumerate(stats_data):
        row["Rank"] = i + 1

    # 4. Save to CSV
    keys = ["Rank", "Concept", "Count", "Prevalence_Percent", "Status"]
    try:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(stats_data)  # Save top 100
        print(f"✅ Proof saved to: {os.path.abspath(OUTPUT_CSV)}")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")


if __name__ == "__main__":
    ds = load_data()
    analyze_and_save(ds)
