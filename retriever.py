import torch
import os
import open_clip
import faiss
from tqdm import tqdm
from PIL import Image
from prompt_template import get_retrieval_context_string


class MultimodalRetriever:
    """
    Engine for multimodal retrieval (Text + Image) using BioMedCLIP + FAISS.
    """

    def __init__(self,
                 model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 cache_dir="./rag_cache"):

        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        print(f"🔄 Initializing BioMedCLIP on {self.device}...")

        self.model, self.preprocess = open_clip.create_model_from_pretrained(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device)
        self.model.eval()

        # FAISS Indices (Initialized in build_index)
        self.text_index = None
        self.image_index = None

        # Knowledge Base (Metadata)
        self.knowledge_base = []

    def _get_image_features(self, images):
        """Internal helper to encode images."""
        # OpenCLIP expects a stacked tensor of preprocessed images
        processed_imgs = []

        for img in images:
            # Validate and convert to RGB
            if not isinstance(img, Image.Image):
                continue
            img = img.convert("RGB")

            # Apply BioMedCLIP's specific preprocessing
            processed_imgs.append(self.preprocess(img))

        if not processed_imgs:
            return torch.tensor([]).to(self.device)

        # Stack into a single tensor batch [Batch_Size, Channels, Height, Width]
        image_input = torch.stack(processed_imgs).to(self.device)

        # Encode
        with torch.no_grad():
            features = self.model.encode_image(image_input)
            # Normalize features (L2 norm) for Cosine Similarity
            features /= features.norm(dim=-1, keepdim=True)

        return features

    def _get_text_features(self, texts):
        """Internal helper to encode texts."""
        if not isinstance(texts, list):
            texts = [texts]

        # Tokenize (context_length=256 is standard for BioMedCLIP)
        text_input = self.tokenizer(texts, context_length=256).to(self.device)

        # Encode
        with torch.no_grad():
            features = self.model.encode_text(text_input)
            # Normalize features
            features /= features.norm(dim=-1, keepdim=True)

        return features

    def _build_faiss_index(self, embeddings):
        """
        Helper to build a FAISS index from embeddings.
        Uses IndexFlatIP (Inner Product) which acts as Cosine Similarity
        since the vectors are already normalized.
        """
        vectors = embeddings.cpu().numpy().astype('float32')
        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        return index

    def build_index(self, dataset):
        """
        Indexes the provided dataset with caching support.
        """
        self.knowledge_base = dataset
        text_index_path = os.path.join(self.cache_dir, "text.index")
        img_index_path = os.path.join(self.cache_dir, "image.index")

        # Load Existing Indices
        if os.path.exists(text_index_path) and os.path.exists(img_index_path):
            print(f"📂 Loading FAISS indices from disk...")
            self.text_index = faiss.read_index(text_index_path)
            self.image_index = faiss.read_index(img_index_path)
            print("✅ Indices loaded successfully.")
            return

        print(f"🏗️ Index not found. Encoding {len(dataset)} samples...")

        # Encode Texts (Questions)
        questions = [item['question'] for item in dataset]
        text_feats = []
        batch_size = 32

        for i in range(0, len(questions), batch_size):
            batch = questions[i: i + batch_size]
            text_feats.append(self._get_text_features(batch))
        text_embeds = torch.cat(text_feats, dim=0)

        # Encode Images
        img_feats = []
        images = [item['image'] for item in dataset]

        for i in tqdm(range(0, len(images), batch_size), desc="Indexing Images"):
            batch = images[i: i + batch_size]
            img_feats.append(self._get_image_features(batch))
        img_embeds = torch.cat(img_feats, dim=0)

        # Build FAISS Indices
        print("⚡ Building FAISS Indices...")
        self.text_index = self._build_faiss_index(text_embeds)
        self.image_index = self._build_faiss_index(img_embeds)

        # Save FAISS Indices
        print(f"💾 Saving indices to {self.cache_dir}...")
        faiss.write_index(self.text_index, text_index_path)
        faiss.write_index(self.image_index, img_index_path)
        print("✅ Indexing Complete.")

    def retrieve(self, query_text, query_image, k=2, alpha=0.5):
        """
        Retrieves top-k examples using Late Fusion of Text and Image scores.
        Score = (alpha * Text_Similarity) + ((1 - alpha) * Image_Similarity)
        """
        # Encode Query
        q_text_emb = self._get_text_features([query_text]).cpu().numpy().astype('float32')

        if not isinstance(query_image, list):
            query_image = [query_image]
        q_img_emb = self._get_image_features(query_image).cpu().numpy().astype('float32')

        # Search FAISS
        search_k = k * 5

        # search() returns distances (scores) and indices
        text_d, text_i = self.text_index.search(q_text_emb, search_k)
        img_d, img_i = self.image_index.search(q_img_emb, search_k)

        # Manual Late Fusion
        combined_scores = {}

        # Process Text Scores (Weighted by Alpha)
        for score, idx in zip(text_d[0], text_i[0]):
            if idx != -1:
                combined_scores[idx] = combined_scores.get(idx, 0) + (alpha * score)

        # Process Image Scores (Weighted by 1-Alpha)
        for score, idx in zip(img_d[0], img_i[0]):
            if idx != -1:
                combined_scores[idx] = combined_scores.get(idx, 0) + ((1 - alpha) * score)

        # Sort by best combined score
        sorted_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_candidates[:k]

        # Format Results
        results = []
        for idx, score in top_k:
            item = self.knowledge_base[int(idx)]
            results.append({
                "idx": item.get('idx', int(idx)),
                "question": item['question'],
                "answer": str(item['answer']),
                "score": float(score)
            })
        return results

    def format_prompt(self, examples):
        """"Formats retrieved examples into a context string."""
        return get_retrieval_context_string(examples)
