import torch
from reranker import Reranker


class RAGPipeline:
    """
    Orchestrates the Retrieval-Augmented Generation pipeline.
    Combines a Retrieval Engine with a VQA Model Adapter.
    """
    def __init__(self,
                 llm_adapter,
                 retrieval_engine,
                 k=2,
                 alpha=0.5,
                 use_reranker=False,
                 reranker_model="BAAI/bge-reranker-base",
                 rerank_k=20,
                 device="cuda" if torch.cuda.is_available() else "cpu"):

        self.device = device
        self.llm = llm_adapter
        self.retriever = retrieval_engine
        self.k = k
        self.alpha = alpha

        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.rerank_k = rerank_k
        if self.use_reranker:
            self.reranker = Reranker(model_id=self.reranker_model, device=self.device)

    def load(self):
        """Ensures the underlying LLM is loaded."""
        if self.llm.model is None:
            self.llm.load()

    def build_index(self, dataset):
        """Delegates indexing to the retrieval engine."""
        self.retriever.build_index(dataset)

    def generate(self, image, question, context=""):
        """
        Executes the RAG flow: Retrieve -> Contextualize -> Generate.
        """
        # If external context isn't provided, use RAG to find it
        if not context:
            # If reranking, fetch 20 candidates
            initial_k = self.rerank_k if self.use_reranker else self.k

            retrieved_items = self.retriever.retrieve(
                query_text=question,
                query_image=image,
                k=initial_k,
                alpha=self.alpha
            )

            if self.use_reranker:
                retrieved_items = self.reranker.rerank(
                    query=question,
                    candidates=retrieved_items,
                    top_k=self.k
                )

            if retrieved_items:
                context = self.retriever.format_prompt(retrieved_items)
            else:
                context = ""

        # Pass to the underlying llm adapter
        return self.llm.generate(image, question, context=context)

    def judge_answer(self, image, question, raw_answer):
        """Pass-through to the LLM (Judges don't use retrieval)."""
        return self.llm.judge_answer(image, question, raw_answer)
