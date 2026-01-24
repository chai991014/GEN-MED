from inference.prompt_template import (
    get_reflexion_critique_prompt,
    get_reflexion_critique_context,
    get_reflexion_refine_prompt
)


class RAGPipeline:
    """
    Orchestrates the Retrieval-Augmented Generation pipeline.
    Combines a Retrieval Engine with a VQA Model Adapter.
    """

    def __init__(self,
                 llm_adapter,
                 retrieval_engine,
                 k=2,
                 alpha=0.5):

        self.llm = llm_adapter
        self.retriever = retrieval_engine
        self.xai_retriever = None
        self.k = k
        self.alpha = alpha

    def load(self):
        """Ensures the underlying LLM is loaded."""
        if self.llm.model is None:
            self.llm.load()

    def build_index(self, dataset, index_name="default"):
        """Delegates indexing to the retrieval engine."""
        self.retriever.build_index(dataset, index_name=index_name)

    def _retrieve_context(self, image, question):
        """Helper to retrieve and format context string."""

        retrieved_items = self.retriever.retrieve(
            query_text=question,
            query_image=image,
            k=self.k,
            alpha=self.alpha
        )

        context = ""
        retrieved_ids = []
        retrieved_scores = []

        if retrieved_items:
            context = self.retriever.format_prompt(retrieved_items)
            retrieved_ids = [item.get('idx') for item in retrieved_items]
            retrieved_scores = [round(item.get('score', 0), 4) for item in retrieved_items]
        return context, retrieved_ids, retrieved_scores

    def generate(self, image, question, context="", do_sample=False, temperature=1.0):
        """
        Executes the RAG flow: Retrieve -> Contextualize -> Generate.
        """
        retrieved_ids, retrieved_scores = [], []
        # If external context isn't provided, use RAG to find it
        if not context:
            context, retrieved_ids, retrieved_scores = self._retrieve_context(image, question)
        # Pass to the underlying llm adapter
        result = self.llm.generate(image, question, context=context, do_sample=do_sample, temperature=temperature)
        result["retrieved_ids"] = retrieved_ids
        result["retrieved_scores"] = retrieved_scores
        return result

    def reflexion_generate(self, image, question):
        """
        Implements Reflexion (Draft -> Critique -> Refine) WITH RAG Context.
        """
        # Retrieve Context ONCE
        rag_context, retrieved_ids, retrieved_scores = self._retrieve_context(image, question)

        # Draft (Fast Thinking + RAG)
        draft_dict = self.llm.generate(image, question, context=rag_context)
        draft_answer = draft_dict["prediction"]

        # Critique (Self-Correction)
        critique_prompt = get_reflexion_critique_prompt(draft_answer)

        # The critique context helps maintain conversation history
        critique_context = get_reflexion_critique_context(question, rag_context)

        critique_dict = self.llm.generate(image, critique_prompt, context=critique_context)
        critique_response = critique_dict["prediction"]

        # Refine (Final Answer + RAG)
        refine_prompt = get_reflexion_refine_prompt(question, draft_answer, critique_response)
        final_dict = self.llm.generate(image, refine_prompt, context=rag_context)

        # We pass the RAG context again to ensure the final answer is grounded
        return {
            "prediction": final_dict["prediction"],
            "reflexion_draft": draft_answer,
            "reflexion_critique": critique_response,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores
        }
