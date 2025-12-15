import sys
import os
import string
from datetime import datetime
from config import CONFIG


def normalize_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()


class DualLogger(object):
    """
    Redirects stdout to both the terminal and a log file.
    """

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


def setup_logger(output_dir, base_name):
    """
    Sets up the logger to write to a file in output_dir.
    Returns the full path to the log file.
    """
    log_file_path = os.path.join(output_dir, f"log_{base_name}.txt")
    sys.stdout = DualLogger(log_file_path)
    return log_file_path


def print_system_config(config, tech_tag):
    """
    Prints the formatted system configuration to the log.
    """
    print("\n" + "=" * 60)
    print("⚙️ SYSTEM CONFIGURATION")
    print("=" * 60)
    print(f"   [General]")
    print(f"   • Test Mode       : {config.get('TEST_MODE')}")
    print(f"   • Dataset         : {config.get('DATASET_ID')}")
    print(f"   • Model           : {config.get('MODEL_CHOICE')}")

    if tech_tag == "AMANDA":
        print(f"\n   [ AMANDA - MultiAgent RAG System ]")
    elif tech_tag != "ZeroShot":
        print(f"\n   [Technique]")

    if config.get("USE_RAG"):
        print(f"   • RAG             : Enabled")
        print(f"     - Retrieval K   : {config.get('RAG_K')}")
        print(f"     - Alpha         : {config.get('RAG_ALPHA')}")

    if config.get("RERANKER_MODEL") is not None:
        print(f"   • Reranker        : Enabled")
        print(f"     - Reranker Model: {config.get('RERANKER_MODEL')}")
        print(f"     - Rerank K      : {config.get('RERANK_K')}")

    if config.get("USE_REFLEXION"):
        print(f"   • Reflexion       : {config.get('USE_REFLEXION')}")

    print("=" * 60 + "\n")


def print_final_report(tech_tag, model_choice, closed_acc, bert_score, rouge_score, bleu_score, total_time, avg_time):
    """
    Prints the final performance metrics table.
    """
    print("\n" + "=" * 60)
    print(f"✅ FINAL RESULTS: {tech_tag} - {model_choice}")
    print("-" * 60)
    print(f"   [Closed-Ended]")
    print(f"   • Accuracy:           {closed_acc:.2f}%")
    print(f"\n   [Open-Ended]")
    print(f"   • BERTScore:          {bert_score:.2f}")
    print(f"   • ROUGE-L:            {rouge_score:.2f}")
    print(f"   • BLEU-1:             {bleu_score:.2f}")
    print("-" * 60)
    print(f"   [Performance]")
    print(f"   • Total Time:         {total_time:.2f} sec")
    print(f"   • Avg Time/Inference: {avg_time:.2f} sec")
    print("=" * 60)


def get_config():
    # Setup paths
    tags = []
    if CONFIG["USE_RAG"]:
        tags.append("RAG")
    if CONFIG["RERANKER_MODEL"] is not None:
        tags.append("Rerank")
    if CONFIG["USE_REFLEXION"]:
        tags.append("Reflexion")

    if not tags:
        tech_tag = "ZeroShot"
    else:
        tech_tag = "+".join(tags)

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

    if CONFIG["MODEL_CHOICE"] == "MEVF":
        CONFIG["TEST_MODE"] = False
        CONFIG["USE_RAG"] = False
        CONFIG["USE_REFLEXION"] = False
        CONFIG["USE_AMANDA"] = False
        tech_tag = f"""MEVF+{CONFIG["REASONING_MODEL"]}"""
        base_name = f"{tech_tag}_{timestamp}"

    if CONFIG["USE_AMANDA"]:
        CONFIG["USE_RAG"] = True
        CONFIG["USE_REFLEXION"] = False
        tech_tag = "AMANDA"

    if CONFIG["TEST_MODE"]:
        base_name = f"Test5_{tech_tag}_{model_short}_{timestamp}"
    else:
        base_name = f"{tech_tag}_{model_short}_{timestamp}"

    OUTPUT_FILE = f"{output_dir}/results_{base_name}.csv"

    CONFIG["OUTPUT_FILE"] = OUTPUT_FILE
    CONFIG["TECH_TAG"] = tech_tag

    # Setup Logging via Utils
    setup_logger(output_dir, base_name)
    print_system_config(CONFIG, tech_tag)
    return CONFIG
