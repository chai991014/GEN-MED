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

    def rerank(self, query, candidates, top_k=2, visual_weight=0.3):
        """
        Re-scores candidates by fusing Text Reranker Score + Original Visual Score.

        Args:
            visual_weight (float): How much to trust the original CLIP/Visual score.
                                   0.0 = Text Only (Standard Reranker behavior)
                                   0.3 = Balanced Hybrid (Recommended)
        """
        if not candidates:
            return []

        # 1. Prepare Pairs (Query, Candidate Question)
        pairs = [[query, doc['question']] for doc in candidates]

        # 2. Compute Deep Text Scores (Cross-Encoder)
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(
                self.device)
            logits = self.model(**inputs, return_dict=True).logits.view(-1, ).float()

            # Normalize logits to 0-1 range (Sigmoid) so they match CLIP score scale
            text_scores = torch.sigmoid(logits).cpu().numpy()

        # 3. Hybrid Score Fusion
        scored_candidates = []
        for i, t_score in enumerate(text_scores):
            # Retrieve the original CLIP score (which includes image similarity)
            # We assume 'score' key exists from the Retriever
            original_score = candidates[i].get('score', 0.0)

            # FUSION FORMULA:
            # Final = (Deep_Text_Score) + (Weight * Original_Visual_Score)
            # This boosts items that are textually accurate AND visually similar
            final_score = t_score + (visual_weight * original_score)

            candidates[i]['rerank_text_score'] = float(t_score)
            candidates[i]['final_score'] = float(final_score)
            scored_candidates.append(candidates[i])

            # 4. Sort by the NEW Hybrid Final Score
        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)

        return scored_candidates[:top_k]
