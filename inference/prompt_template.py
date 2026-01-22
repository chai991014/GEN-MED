def get_inference_prompt(question, context=""):
    """
    Standard VQA Prompt.
    Used by: llm_adapter.generate, rag_pipeline.generate
    """
    if context:
        if "Reference Examples:" in context:
            return (
                f"{context}\n"
                f"Based on the examples above and the image, answer this:\n"
                f"Question: {question}"
            )
        return (
            f"{context}\n"
            f"Based on the hypothesis above and the image, answer this:\n"
            f"Question: {question}"
        )
    return question


def get_instruct_inference_prompt(question, context=""):
    """
    Enhanced VQA Prompt with Persona and Task Specification.
    """
    # 1. System Instruction (The "Persona")
    system_instruction = (
        "You are an expert radiologist and medical AI assistant. "
        "Your task is to analyze the provided medical image and answer the question "
        "truthfully based ONLY on the visual evidence."
    )

    # 2. Constraints (To prevent hallucinations/verbosity)
    constraints = (
        "Instructions:\n"
        "- Provide a concise, factual answer.\n"
        "- If the question demands a Yes/No answer, output the label first, then a brief reason.\n"
        "- Do not make up information not visible in the scan."
    )

    # 3. Dynamic Context Handling
    if context:
        # heuristic: Check if context is RAG Examples or Reflexion History
        if "Reference Examples:" in context:
            # Case A: RAG Context
            connector = (
                "Instructions: The examples above are for reference style only. "
                "Analyze the CURRENT image to answer the question below."
            )
        else:
            # Case B: Reflexion/Critique Context
            connector = (
                "Instructions: Review the context above (Previous Answer/Critique) "
                "and re-evaluate the image to provide the correct answer."
            )

        return (
            f"{system_instruction}\n\n"
            f"{context}\n\n"
            f"{connector}\n"
            f"Question: {question}\n"
            f"Answer:"
        )

    # Zero-Shot Case: Strong System Prompting
    return (
        f"{system_instruction}\n"
        f"{constraints}\n\n"
        f"Question: {question}\n"
        f"Answer:"
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
        f"You are a critical medical evaluator. You previously answered: '{draft_answer}'.\n"
        f"Task: Re-examine the image to verify this answer. \n"
        f"1. Check for laterality errors (Left vs Right).\n"
        f"2. Check if the finding is actually present or hallucinated.\n"
        f"3. Are there contradictory visual details you missed?\n"
        f"Output a brief critique listing ONLY the errors or visual contradictions. "
        f"If the answer is visually supported, say 'Verified'."
    )


def get_reflexion_critique_context(question, rag_context=""):
    """
    Context passed during the critique phase.
    """
    base_context = f"Original Question: {question}"
    if rag_context:
        return f"{base_context}\n{rag_context}"
    return base_context


def get_reflexion_refine_prompt(question, draft_answer, critique_response):
    """
    Refine the answer based on critique.
    """
    return (
        f"Task: Final Diagnosis.\n"
        f"Original Question: '{question}'\n"
        f"Initial Hypothesis: '{draft_answer}'\n"
        f"Visual Critique: '{critique_response}'\n\n"
        f"Instruction: Based on the image and the critique above, provide the corrected final answer. "
        f"If the critique confirmed the hypothesis, repeat the hypothesis concisely. "
        f"If errors were found, correct them."
    )
