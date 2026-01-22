CONFIG = {
    "TECH_TAG": None,
    "OUTPUT_FILE": None,
    "DATASET_ID": "flaviagiammarino/vqa-rad",
    # "DATASET_ID": "./slake_vqa_rad_format",
    "DATASET": "test",

    "TEST_MODE": True,  # Run 20 samples only

    # ==========================================
    # MEVF Settings
    # ==========================================
    # "MODEL_TYPE": "MEVF",
    "REASONING_MODEL": "BAN",
    "MEVF_WEIGHTS": "./inference/mevf/BAN_best_model.pth",
    "MAML_WEIGHTS": "./inference/mevf/pretrained_maml.weights",
    "AE_WEIGHTS": "./inference/mevf/pretrained_ae.pth",

    # ==========================================
    # LLM Model Selection
    # ==========================================
    "LLAVA_REPO_PATH": "./LLaVA-Med",
    # "MODEL_TYPE": "microsoft/llava-med-v1.5-mistral-7b",
    # "MODEL_TYPE": "Qwen/Qwen3-VL-2B-Instruct",
    "MODEL_TYPE": "Qwen/Qwen3-VL-4B-Instruct",

    # "ADAPTER_PATH": None,
    # "MODEL_TYPE": "./finetune/qlora-llava",
    # "MODEL_TYPE": "./finetune/dora-llava",
    # "ADAPTER_PATH": "./finetune/qlora-qwen3-2b/checkpoint-600",
    # "ADAPTER_PATH": "./finetune/dora-qwen3-2b/checkpoint-580",
    "ADAPTER_PATH": "./finetune/qlora-qwen3-4b/checkpoint-390",
    # "ADAPTER_PATH": "./finetune/dora-qwen3-4b/checkpoint-390",

    # ==========================================
    # LLM Technique
    # ==========================================
    # "PROMPT": "Basic",
    "PROMPT": "Instruct",

    "USE_REFLEXION": True,  # Toggle Reflexion Thinking

    # ==========================================
    #  RAG Settings
    # ==========================================
    "USE_RAG": True,  # Toggle RAG
    "RAG_K": 3,  # Number of exemplars to retrieve
    "RAG_ALPHA": 0.5,  # Alpha weight Text similarity and 1-Alpha weight Image similarity
}
