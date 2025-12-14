import sys
import os


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
    print(f"   • Test Mode:        {config.get('TEST_MODE')}")
    print(f"   • Dataset:          {config.get('DATASET_ID')}")
    print(f"   • Model:            {config.get('MODEL_CHOICE')}")

    if tech_tag != "ZeroShot":
        print(f"\n   [Technique]")

    if config.get("USE_RAG"):
        print(f"   • RAG:              {config.get('USE_RAG')}")
        print(f"     - Retrieval K:    {config.get('RAG_K')}")
        print(f"     - Alpha:          {config.get('RAG_ALPHA')}")

    if config.get("USE_RERANKER"):
        print(f"   • Reranker:         {config.get('USE_RERANKER')}")
        print(f"     - Model:          {config.get('RERANKER_MODEL')}")
        print(f"     - Rerank K:       {config.get('RERANK_K')}")

    if config.get("USE_REFLEXION"):
        print(f"   • Reflexion:        {config.get('USE_REFLEXION')}")

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
