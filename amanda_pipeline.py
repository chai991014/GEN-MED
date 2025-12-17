import string
from utils import normalize_text
from prompt_template import (
    get_amanda_perceiver_prompt,
    get_amanda_reasoner_prompt,
    get_amanda_evaluator_prompt,
    get_amanda_retriever_query_prompt,
    get_retrieval_context_string
)


class AMANDAPipeline:
    """
    Implementation of AMANDA (Agentic Medical Knowledge Augmentation).
    """

    def __init__(self,
                 llm_adapter,
                 retrieval_engine,
                 reranker_engine=None,
                 k=3,
                 alpha=0.5,
                 rerank_k=20):

        self.llm = llm_adapter
        self.retriever = retrieval_engine
        self.reranker = reranker_engine
        self.k = k
        self.alpha = alpha
        self.rerank_k = rerank_k

    def build_index(self, dataset):
        """Builds the Knowledge Base (Extrinsic Memory)."""
        self.retriever.build_index(dataset)

    def generate(self, image, question):
        """
        Executes the AMANDA Agentic Loop
        (Perceiver -> Reasoner -> Evaluator -> Final Answer / Conditional Med-KA (Retriever -> Rerank) -> Reasoner).
        """
        # ... (STEP 1: EXPLORER AGENT - remains the same) ...
        explorer_prompt = get_amanda_perceiver_prompt(question)
        sub_questions = self.llm.generate(image, explorer_prompt)

        # ... (STEP 2: RETRIEVER AGENT) ...
        query_prompt = get_amanda_retriever_query_prompt(sub_questions)
        search_query = self.llm.generate(image, query_prompt)

        # Determine initial K for retrieval: Rerank K if reranker is used, else final K
        initial_k = self.rerank_k if self.reranker is not None else self.k

        candidates = self.retriever.retrieve(
            query_text=search_query,
            query_image=image,
            k=initial_k,
            alpha=self.alpha
        )

        # ==========================================
        # 2.5. RERANKING STEP (Passive Filtering)
        # ==========================================
        if self.reranker is not None:
            # Reorder candidates based on Cross-Encoder score
            # Note: We rerank using the LLM-generated 'search_query'
            candidates = self.reranker.rerank(
                query=search_query,
                candidates=candidates,
                top_k=self.k * 3  # We still over-fetch here for the Evaluator
            )

        # ==========================================
        # 3. EVALUATOR AGENT (Active Filtering - remains the same)
        # ==========================================
        verified_knowledge = []

        for item in candidates:
            # ... (Evaluation logic remains the same) ...
            eval_prompt = get_amanda_evaluator_prompt(question, item)
            decision = self.llm.judge_answer(image, eval_prompt, raw_answer="")

            clean_decision = normalize_text(decision)

            if "yes" in clean_decision:
                verified_knowledge.append(item)

            if len(verified_knowledge) >= self.k:
                break

        # Fallback (remains the same)
        if not verified_knowledge and candidates:
            verified_knowledge.append(candidates[0])

        # ... (STEP 4. REASONER AGENT - remains the same) ...
        context_str = get_retrieval_context_string(verified_knowledge)
        final_prompt = get_amanda_reasoner_prompt(question, sub_questions, context_str)
        return self.llm.generate(image, final_prompt)

    def judge_answer(self, image, question, raw_answer):
        return self.llm.judge_answer(image, question, raw_answer)
