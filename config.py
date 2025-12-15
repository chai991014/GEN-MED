CONFIG = {
    "OUTPUT_FILE": None,
    "DATASET_ID": "flaviagiammarino/vqa-rad",

    "TECH_TAG": None,
    "TEST_MODE": True,  # Run 5 samples only
    "USE_RAG": True,  # Toggle RAG
    "USE_REFLEXION": False,  # Toggle Reflexion Thinking
    "USE_AMANDA": False,  # Toggle AMANDA Multi-Agent RAG Framework

    # ==========================================
    # MEVF Settings
    # ==========================================
    "MODEL_CHOICE": "MEVF",
    "REASONING_MODEL": "SAN",
    "MEVF_WEIGHTS": "./mevf/SAN_best_model.pth",
    "MAML_WEIGHTS": "./mevf/pretrained_maml.weights",
    "AE_WEIGHTS": "./mevf/pretrained_ae.pth",

    # ==========================================
    # LLM Model Selection
    # ==========================================
    # "MODEL_CHOICE": "microsoft/llava-med-v1.5-mistral-7b",
    # "MODEL_CHOICE": "Qwen/Qwen2-VL-2B-Instruct",
    # "MODEL_CHOICE": "Qwen/Qwen2-VL-7B-Instruct",  # [OOM]
    # "MODEL_CHOICE": "Qwen/Qwen2.5-VL-3B-Instruct",
    # "MODEL_CHOICE": "Qwen/Qwen2.5-VL-7B-Instruct",  # [OOM]
    # "MODEL_CHOICE": "Qwen/Qwen3-VL-2B-Instruct",
    # "MODEL_CHOICE": "Qwen/Qwen3-VL-4B-Instruct",
    # "MODEL_CHOICE": "Qwen/Qwen3-VL-8B-Instruct",  # [OOM]

    "LLAVA_REPO_PATH": "./LLaVA-Med",

    # ==========================================
    #  RAG Settings
    # ==========================================
    "RAG_K": 2,  # Number of exemplars to retrieve
    "RAG_ALPHA": 0.5,  # Alpha weight Text similarity and 1-Alpha weight Image similarity

    # ==========================================
    #  Reranker Settings
    # ==========================================
    # "RERANKER_MODEL": None,
    # "RERANKER_MODEL": "BAAI/bge-reranker-base",
    "RERANKER_MODEL": "ncbi/MedCPT-Cross-Encoder",
    "RERANK_K": 20,  # Number of exemplars to retrieve for rerank
}
