import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker:
    """
    Implements a Cross-Encoder Reranker to refine retrieval results.
    Model: BAAI/bge-reranker-base (Lightweight & Powerful)
    """

    def __init__(self, model_id="BAAI/bge-reranker-base", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚖️ Initializing Reranker ({model_id}) on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def rerank(self, query, candidates, top_k=2):
        """
        Re-scores a list of candidate dictionaries based on the query.
        candidates: List of dicts [{'question': '...', 'answer': '...'}, ...]
        """
        if not candidates:
            return []

        # 1. Prepare Pairs (Query, Candidate Question)
        # We only rerank based on the TEXT question similarity
        pairs = [[query, doc['question']] for doc in candidates]

        # 2. Tokenize
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(
                self.device)

            # 3. Predict Scores
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()

        # 4. Sort & Filter
        # Combine score with original candidate object
        scored_candidates = []
        for i, score in enumerate(scores):
            # Sigmoid not strictly needed for ranking, but good for debug
            candidates[i]['rerank_score'] = score.item()
            scored_candidates.append(candidates[i])

        # Sort descending by new rerank score
        scored_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

        return scored_candidates[:top_k]
