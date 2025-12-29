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
    print(f"   • Model           : {config.get('MODEL_TYPE')}")

    if config.get('MODEL_TYPE') == "MEVF":
        print(f"   • Reasoning Model : {config.get('REASONING_MODEL')}")
    else:
        print(f"\n   [Technique]")
        print(f"   • Prompt Style    : {config.get('PROMPT')}")

        if config.get("USE_REFLEXION"):
            print(f"   • Reflexion       : Enabled")

        if config.get("USE_RAG"):
            print(f"   • RAG             : Enabled")
            print(f"     - Retrieval K   : {config.get('RAG_K')}")
            print(f"     - Alpha         : {config.get('RAG_ALPHA')}")

    print("=" * 60 + "\n")


def get_config():
    # Setup paths
    tags = []
    if CONFIG["USE_RAG"]:
        tags.append("RAG")
    if CONFIG["USE_REFLEXION"]:
        tags.append("Reflexion")
    if not tags:
        tags.append("ZeroShot")
    if CONFIG["PROMPT"] is not None:
        tags.append(CONFIG["PROMPT"])

    tech_tag = "+".join(tags)

    model_map = {
        "microsoft/llava-med-v1.5-mistral-7b": "LLaVA-Med",
        "Qwen/Qwen3-VL-2B-Instruct": "Qwen3-2B",
        "Qwen/Qwen3-VL-4B-Instruct": "Qwen3-4B",
    }
    model_short = model_map.get(CONFIG["MODEL_TYPE"], CONFIG["MODEL_TYPE"].replace("/", "_"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "./result"
    os.makedirs(output_dir, exist_ok=True)

    if CONFIG["TEST_MODE"]:
        base_name = f"Test5_{tech_tag}_{model_short}_{timestamp}"
    else:
        base_name = f"{tech_tag}_{model_short}_{timestamp}"

    if CONFIG["MODEL_TYPE"] == "MEVF":
        CONFIG["TEST_MODE"] = False
        CONFIG["USE_RAG"] = False
        CONFIG["USE_REFLEXION"] = False
        tech_tag = f"""MEVF+{CONFIG["REASONING_MODEL"]}"""
        base_name = f"{tech_tag}_{timestamp}"

    OUTPUT_FILE = f"{output_dir}/results_{base_name}.csv"

    CONFIG["OUTPUT_FILE"] = OUTPUT_FILE
    CONFIG["TECH_TAG"] = tech_tag

    # Setup Logging via Utils
    setup_logger(output_dir, base_name)
    print_system_config(CONFIG, tech_tag)
    return CONFIG
