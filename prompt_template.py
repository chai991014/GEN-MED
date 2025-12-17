def get_inference_prompt(question, context=""):
    """
    Standard VQA Prompt.
    Used by: llm_adapter.generate, rag_pipeline.generate
    """
    if context:
        return (
            f"{context}\n"
            f"Based on the examples above and the image, answer this:\n"
            f"Question: {question}"
        )
    return question


def get_judge_prompt(question, raw_answer):
    """
    Judge / Evaluation Prompt.
    Used by: llm_adapter.judge_answer
    """
    return (
        f"Question: {question}\n"
        f"Answer: {raw_answer}\n"
        "Task: Classify the Model Answer as 'Yes' or 'No'.\n"
        "Constraint: Output ONLY the label 'Yes' or 'No'. Do not explain.\n\n"
        "Verdict:"
    )


def get_api_judge_prompt(question, raw_answer):
    """
    External Judge / Evaluation Prompt.
    Used by: llm_judge.judge
    """
    return (
        f"""
        Task: Classify the Medical AI Response into "YES" or "NO" relative to the question.

        Question: {question}
        Model Response: {raw_answer}

        Rules:
        1. If the response indicates a positive finding/agreement, output "YES".
        2. If the response indicates a negative finding/absence/disagreement, output "NO".
        3. If it's ambiguous or requires checking the image, output "VISION_REQUIRED".
        4. Output ONLY the label.
        """
    )


def get_retrieval_context_string(examples):
    """
    Formats retrieved examples into a single block.
    Used by: retriever.format_prompt
    """
    if not examples:
        return ""
    context = "Reference Examples:\n"
    for i, ex in enumerate(examples):
        context += f"Ex {i + 1}: Q: {ex['question']} -> A: {ex['answer']}\n"
    return context


def get_reflexion_critique_prompt(draft_answer):
    """
    Critique the initial draft.
    """
    return (
        f"You previously answered: '{draft_answer}'. "
        f"Look at the image again. Are there any visual details (lesions/fractures) "
        f"you missed that contradict this? Briefly describe them."
    )


def get_reflexion_critique_context(question, rag_context=""):
    """
    Context passed during the critique phase.
    """
    base_context = f"Previous Question: {question}"
    if rag_context:
        return f"{base_context}\n{rag_context}"
    return base_context


def get_reflexion_refine_prompt(question, draft_answer, critique_response):
    """
    Refine the answer based on critique.
    """
    return (
        f"The original question was: '{question}'. "
        f"Your initial answer was: '{draft_answer}'. "
        f"Your critique was: '{critique_response}'. "
        f"Based on the image and this critique, provide the final correct answer."
    )
