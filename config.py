CONFIG = {
    "TECH_TAG": None,
    "OUTPUT_FILE": None,
    "DATASET_ID": "flaviagiammarino/vqa-rad",
    "DATASET": "test",

    "TEST_MODE": True,  # Run 20 samples only

    # ==========================================
    # MEVF Settings
    # ==========================================
    "MODEL_TYPE": "MEVF",
    "REASONING_MODEL": "BAN",
    "MEVF_WEIGHTS": "./mevf/BAN_best_model.pth",
    "MAML_WEIGHTS": "./mevf/pretrained_maml.weights",
    "AE_WEIGHTS": "./mevf/pretrained_ae.pth",

    # ==========================================
    # LLM Model Selection
    # ==========================================
    "LLAVA_REPO_PATH": "./LLaVA-Med",
    # "MODEL_TYPE": "microsoft/llava-med-v1.5-mistral-7b",
    # "MODEL_TYPE": "Qwen/Qwen3-VL-2B-Instruct",
    # "MODEL_TYPE": "Qwen/Qwen3-VL-4B-Instruct",

    # ==========================================
    # LLM Technique
    # ==========================================
    "PROMPT": "Basic",
    # "PROMPT": "Instruct",

    "USE_REFLEXION": False,  # Toggle Reflexion Thinking

    # ==========================================
    #  RAG Settings
    # ==========================================
    "USE_RAG": False,  # Toggle RAG
    "RAG_K": 3,  # Number of exemplars to retrieve
    "RAG_ALPHA": 0.5,  # Alpha weight Text similarity and 1-Alpha weight Image similarity
}
