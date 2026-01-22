import sys
import os
import string
from datetime import datetime


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


def print_system_config(config):
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
    if config.get('ADAPTER_PATH'):
        print(f"   • Fine-Tuned      : {config.get('ADAPTER_PATH')}")

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


def get_config(config):
    # Setup paths
    tags = []
    if config["USE_RAG"]:
        tags.append("RAG")
    if config["USE_REFLEXION"]:
        tags.append("Reflexion")
    if not tags:
        tags.append("ZeroShot")
    if config["PROMPT"] is not None:
        tags.append(config["PROMPT"])

    tech_tag = "+".join(tags)

    model_map = {
        "microsoft/llava-med-v1.5-mistral-7b": "LLaVA-Med",
        "Qwen/Qwen3-VL-2B-Instruct": "Qwen3-2B",
        "Qwen/Qwen3-VL-4B-Instruct": "Qwen3-4B",
        "./finetune/qlora-llava": "QLoRA-LLaVA-Med",
        "./finetune/dora-llava": "DoRA-LLaVA-Med"
    }

    if config["ADAPTER_PATH"] is not None:
        if "qlora" in config["ADAPTER_PATH"]:
            adapter_short = "QLoRA"
        elif "dora" in config["ADAPTER_PATH"]:
            adapter_short = "DoRA"
        else:
            adapter_short = None
    else:
        adapter_short = None

    model_short = model_map.get(config["MODEL_TYPE"], config["MODEL_TYPE"].replace("/", "_"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "./inference/result"
    os.makedirs(output_dir, exist_ok=True)

    if adapter_short is not None:
        base_name = f"{tech_tag}_{adapter_short}-{model_short}_{timestamp}"
    else:
        base_name = f"{tech_tag}_{model_short}_{timestamp}"

    if config["TEST_MODE"]:
        base_name = f"Test5_{base_name}"

    if config["MODEL_TYPE"] == "MEVF":
        config["TEST_MODE"] = False
        config["USE_RAG"] = False
        config["USE_REFLEXION"] = False
        tech_tag = f"""MEVF+{config["REASONING_MODEL"]}"""
        base_name = f"{tech_tag}_{timestamp}"

    OUTPUT_FILE = f"{output_dir}/results_{base_name}.csv"

    config["OUTPUT_FILE"] = OUTPUT_FILE
    config["TECH_TAG"] = tech_tag

    # Setup Logging via Utils
    setup_logger(output_dir, base_name)
    print_system_config(config)
    return config
