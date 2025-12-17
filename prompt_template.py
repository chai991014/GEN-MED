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


def get_cot_inference_prompt(question, context=""):
    """
    Use this if the model struggles with reasoning.
    """
    # 1. Base Persona
    system_instruction = (
        "You are an expert radiologist. Analyze the image step-by-step to reach a conclusion."
    )

    # 2. Context Handling
    connector = ""
    if context:
        if "Reference Examples:" in context:
            connector = "Reference the examples above for medical terminology, but focus strictly on the provided image."
        else:
            connector = "Re-evaluate the case considering the critique/history above."

    # 3. Structure with CoT Steps
    base_prompt = (
        f"Question: {question}\n"
        "Step 1: Describe the key visual findings in the image (e.g., opacity, fracture, location).\n"
        "Step 2: Relate these findings to the question.\n"
        "Step 3: Conclude with the final concise answer.\n\n"
        "Answer:"
    )

    if context:
        return f"{system_instruction}\n\n{context}\n\n{connector}\n{base_prompt}"

    return f"{system_instruction}\n{base_prompt}"


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
