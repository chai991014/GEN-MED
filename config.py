CONFIG = {
    "TECH_TAG": None,
    "OUTPUT_FILE": None,
    "DATASET_ID": "flaviagiammarino/vqa-rad",
    "DATASET": "test",

    "TEST_MODE": True,  # Run 20 samples only

    # ==========================================
    # MEVF Settings
    # ==========================================
    # "MODEL_CHOICE": "MEVF",
    "REASONING_MODEL": "BAN",
    "MEVF_WEIGHTS": "./mevf/BAN_best_model.pth",
    "MAML_WEIGHTS": "./mevf/pretrained_maml.weights",
    "AE_WEIGHTS": "./mevf/pretrained_ae.pth",

    # ==========================================
    #  Groq Judge Settings
    # ==========================================
    "USE_GROQ_JUDGE": False,
    "GROQ_API_KEY": "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",
    "GROQ_MODEL": "meta-llama/llama-4-scout-17b-16e-instruct",

    # ==========================================
    # LLM Model Selection
    # ==========================================
    # "MODEL_CHOICE": "microsoft/llava-med-v1.5-mistral-7b",
    "LLAVA_REPO_PATH": "./LLaVA-Med",
    # "MODEL_CHOICE": "Qwen/Qwen2-VL-2B-Instruct",
    # "MODEL_CHOICE": "Qwen/Qwen2-VL-7B-Instruct",  # [OOM]
    # "MODEL_CHOICE": "Qwen/Qwen2.5-VL-3B-Instruct",
    # "MODEL_CHOICE": "Qwen/Qwen2.5-VL-7B-Instruct",  # [OOM]
    # "MODEL_CHOICE": "Qwen/Qwen3-VL-2B-Instruct",
    "MODEL_CHOICE": "Qwen/Qwen3-VL-4B-Instruct",
    # "MODEL_CHOICE": "Qwen/Qwen3-VL-8B-Instruct",  # [OOM]

    "USE_REFLEXION": False,  # Toggle Reflexion Thinking

    # ==========================================
    #  RAG Settings
    # ==========================================
    "USE_RAG": False,  # Toggle RAG
    "RAG_K": 2,  # Number of exemplars to retrieve
    "RAG_ALPHA": 0.5,  # Alpha weight Text similarity and 1-Alpha weight Image similarity

    # ==========================================
    #  Reranker Settings
    # ==========================================
    # "RERANKER_MODEL": None,
    # "RERANKER_MODEL": "BAAI/bge-reranker-base",
    "RERANKER_MODEL": "ncbi/MedCPT-Cross-Encoder",
    "RERANK_K": 20,  # Number of exemplars to retrieve for rerank
    "RERANK_VISUAL_WEIGHT": 0.4  # How much to trust the original CLIP/Visual score.
}
