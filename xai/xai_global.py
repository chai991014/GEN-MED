import os
import sys
import time
import json
import pandas as pd
import glob
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import google.generativeai as genai
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# ==========================================
# UNIFIED EXECUTION ENGINE (The "Base")
# ==========================================
def call_gemini_judge(prompt, api_key, temperature=0.0):
    """
    Generic function to handle ALL LLM Judge calls.
    - Handles Auth
    - Enforces JSON output
    - Cleans Markdown formatting
    - Returns a Python Dictionary or None if failure
    """
    try:
        genai.configure(api_key=api_key)
        # Use a low temperature by default for consistent grading
        model = genai.GenerativeModel("gemini-2.5-flash-lite", generation_config={"temperature": temperature})
        generation_config = {"response_mime_type": "application/json"}

        # Enforce JSON mode
        resp = model.generate_content(prompt, generation_config=generation_config)

        text = resp.text.strip()
        # Clean Markdown fences (common issue)
        if text.startswith("```json"):
            text = text[7:-3]
        elif text.startswith("```"):
            text = text[3:-3]

        return json.loads(text)

    except Exception as e:
        print(f"Judge Error: {e}")
        return None


def call_gemini_judge_multimodal(prompt_parts, api_key, temperature=0.0):
    """
    Dedicated function for Multimodal calls (Images + Text).
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            genai.configure(api_key=api_key)
            # specific model for vision tasks
            model = genai.GenerativeModel("gemini-2.5-flash", generation_config={"temperature": temperature})
            generation_config = {"response_mime_type": "application/json"}

            # generate_content accepts a list [text, image, text...]
            resp = model.generate_content(prompt_parts, generation_config=generation_config)

            text = resp.text.strip()
            # Clean Markdown fences
            if text.startswith("```json"):
                text = text[7:-3]
            elif text.startswith("```"):
                text = text[3:-3]

            return json.loads(text)

        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} Multimodal Judge Error: {e}")
            if attempt < max_retries - 1:
                continue

    print(f"❌ All {max_retries} attempts failed.")
    return None


# ==========================================
# SPECIFIC JUDGE WRAPPERS (The "Logic")
# ==========================================
def check_answer_correctness(question, ground_truth, candidate_answer, api_key):
    """
    Wrapper for Outcome Verification.
    Returns: Boolean (True/False)
    """
    # Fast Fail
    if not isinstance(candidate_answer, str) or not candidate_answer.strip() or candidate_answer.lower() == 'nan':
        return False

    prompt = f"""
    ### Role: Senior Medical Radiologist Grader
    ### Task: Verify Medical Equivalence

    Context:
    - Question: "{question}"
    - Ground Truth: "{ground_truth}"
    - Candidate: "{candidate_answer}"

    Rules:
    1. Synonyms are Correct (Opacity = Consolidation).
    2. Laterality Mismatch is FALSE (Left != Right).
    3. Hallucination is FALSE (Don't invent masses).

    Output JSON: {{ "correct": boolean, "reason": "string" }}
    """

    # Call the Unified Engine
    result = call_gemini_judge(prompt, api_key, temperature=0.0)

    # Fallback to string matching if LLM fails completely
    if result is None:
        is_match = str(candidate_answer).lower() in str(ground_truth).lower()
        return is_match, {"reason": "LLM API Failure - Fallback to substring match", "correct": is_match}

    return result.get("correct", False), result


def get_coherence_score(question, draft, critique, final_answer, api_key):
    """
    Wrapper for Process Evaluation.
    Returns: Integer (0-10) or None
    """
    prompt = f"""
    ### Role: Logic Consistency Auditor
    ### Task: Rate the logical coherence of the self-correction process on a scale of 0 to 10.

    Trace:
    - Q: {question}
    - Draft: {draft}
    - Critique: {critique}
    - Final: {final_answer}

    ### Grading Spectrum (0-10):

    **0 (Total Failure / Incoherent):**
    - The Final Answer completely ignores the Critique.
    - OR The Critique says "Verified" but the Final Answer changes the meaning significantly.
    - OR The Critique hallucinates an error that didn't exist, breaking a correct Draft.

    **10 (Perfect Coherence / Logical):**
    - If Critique found an error: The Final Answer fixes exactly that error without introducing new ones.
    - If Critique said "Verified": The Final Answer remains identical to the Draft (or fixes only typos).
    - The reasoning chain is flawless and strictly followed.

    **Instructions:**
    - Use the full range (e.g., 3, 7, 8) to reflect partial success.
    - A score of 5 means "Attempted but failed to fully execute."
    - A score of 8 means "Good correction but missed a minor detail."

    Output strictly JSON: {{ "score": int, "reason": "brief explanation" }}
    """

    # Call the Unified Engine
    # Temperature 0.0 is still best to keep the grading "fair" across samples
    result = call_gemini_judge(prompt, api_key, temperature=0.0)

    if result is None:
        return None, {"error": "API Failure"}

    return int(result.get("score", 0)), result


def check_relevancy_single(query_text, query_img, retrieved_text, retrieved_img, api_key):
    """
    Multimodal Judge: Compares User Query (Img+Txt) vs Retrieved Item (Img+Txt).
    Uses the NEW call_gemini_multimodal function.
    """
    prompt_parts = [
        "### Role: Medical Retrieval Auditor",
        "### Task: Determine if the Retrieved Case is clinically relevant to the Query Case.",
        "### Input Data:",
        "**User Query Image:**", query_img,
        f"**User Query Text:** {query_text}",
        "**Retrieved Case Image:**", retrieved_img,
        f"**Retrieved Case Text:** {retrieved_text}",
        """
        ### Guidelines:
        1. **Anatomy:** Is the body part/modality compatible? (e.g., Chest X-Ray vs Chest X-Ray).
        2. **Pathology:** Does the retrieved case show similar findings or useful differential diagnoses?
        3. **Irrelevant:** If the retrieved image is a different body part or completely unrelated pathology, return FALSE.

        Output JSON: { "relevant": boolean, "reason": "brief explanation" }
        """
    ]

    # [CHANGE] Use the specialized multimodal engine
    result = call_gemini_judge_multimodal(prompt_parts, api_key, temperature=0.0)

    if result is None:
        return False, {"error": "API Failure"}

    return result.get("relevant", False), result


def check_faithfulness(question, context_text_list, final_answer, api_key):
    """
    Text Judge: Checks if Final Answer is supported by Retrieved Context.
    Uses the EXISTING call_gemini_judge (Text-Only).
    """
    context_block = "\n\n".join([f"Case {i + 1}: {txt}" for i, txt in enumerate(context_text_list)])

    prompt = f"""
    ### Role: Fact-Checking Auditor
    ### Task: Verify if the Final Answer is grounded in the Retrieved Context.

    **Retrieved Context:**
    {context_block}

    **User Question:** {question}
    **Final Answer:** {final_answer}

    ### Rules:
    1. **Faithful (True):** All medical claims in the Answer are supported by the Context.
    2. **Hallucinated (False):** The Answer claims specific facts (e.g., "Mass in left lung") that appear NOWHERE in the Context.
    3. **General Knowledge:** Basic definitions are allowed, but specific patient findings must come from Context.

    Output JSON: {{ "faithful": boolean, "reason": "string" }}
    """

    # [CHANGE] Use the existing text-only engine
    result = call_gemini_judge(prompt, api_key, temperature=0.0)

    if result is None:
        return False, {"error": "API Failure"}

    return result.get("faithful", False), result


# ==========================================
# MAIN EVALUATION FUNCTION
# ==========================================
def run_reflexion_eval(input_csv_path):
    # 1. Setup
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("❌ CRITICAL: No Google API_KEY found.")
        sys.exit(1)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv_name = f"./result/global_reflexion_report_{timestamp}.csv"

    df_results = pd.read_csv(input_csv_path)
    eval_results = []

    # Accumulators
    total_coherence_score = 0
    count_valid_scores = 0
    count_reflexion_active = 0

    # Matrix Buckets
    reflexion_outcomes = {
        "Fixed": 0,  # Wrong -> Right (Success)
        "Broke": 0,  # Right -> Wrong (Harmful)
        "Stable": 0,  # Right -> Right
        "Inert": 0  # Wrong -> Wrong
    }
    count_wrong_start = 0  # Denominator for Success Rate

    print(f"\n▶️ Starting Global Reflexion Analysis on {len(df_results)} samples...")

    for index, row in tqdm(df_results.iterrows(), total=len(df_results)):
        row_eval = row.to_dict()

        # Extract Data
        question = row['question']
        gt = str(row.get('ground_truth', ''))
        draft = str(row.get('reflexion_draft', ''))
        critique = str(row.get('reflexion_critique', ''))
        final_pred = str(row.get('raw_prediction', ''))  # This is the Final Answer

        # Only evaluate if Reflexion actually ran
        if draft and draft.lower() != 'nan' and draft.strip():

            # --- PHASE 1: Outcome Verification (Consistent Ruler) ---
            # Judge BOTH Draft and Final with the same prompt logic
            is_draft_correct, draft_raw = check_answer_correctness(question, gt, draft, api_key)
            is_final_correct, final_raw = check_answer_correctness(question, gt, final_pred, api_key)

            row_eval["xai_draft_correct"] = is_draft_correct
            row_eval["xai_final_correct"] = is_final_correct
            row_eval["xai_draft_judge_raw"] = str(draft_raw)
            row_eval["xai_final_judge_raw"] = str(final_raw)

            # --- PHASE 2: Process Evaluation (Coherence) ---
            # Evaluate logical consistency of the trace using your existing class
            coherence_score, coherence_raw = get_coherence_score(question, draft, critique, final_pred, api_key)

            row_eval["reflexion_coherence_score"] = coherence_score
            row_eval["reflexion_coherence_raw"] = str(coherence_raw)

            if coherence_score is not None:
                total_coherence_score += coherence_score
                count_valid_scores += 1

            count_reflexion_active += 1

            # --- PHASE 3: Categorization (Safety Matrix) ---
            category = "Inert"

            # Count Denominator for Success Rate (Drafts that started wrong)
            if not is_draft_correct:
                count_wrong_start += 1

                # Assign Categories
            if not is_draft_correct and is_final_correct:
                category = "Fixed"  # SUCCESS
                reflexion_outcomes["Fixed"] += 1
            elif is_draft_correct and not is_final_correct:
                category = "Broke"  # HARMFUL
                reflexion_outcomes["Broke"] += 1
            elif is_draft_correct and is_final_correct:
                category = "Stable"
                reflexion_outcomes["Stable"] += 1
            else:
                category = "Inert"
                reflexion_outcomes["Inert"] += 1

            row_eval["reflexion_category"] = category

        else:
            row_eval["reflexion_category"] = "Skipped"

        eval_results.append(row_eval)
        pd.DataFrame(eval_results).to_csv(output_csv_name, index=False)

    avg_coherence = total_coherence_score / count_valid_scores if count_valid_scores > 0 else 0
    # Success Rate: % of Wrong Drafts that got Fixed
    success_rate = (reflexion_outcomes["Fixed"] / count_wrong_start * 100) if count_wrong_start > 0 else 0
    # Harmful Rate: % of All Attempts that Broke a Correct Answer
    harmful_rate = (reflexion_outcomes["Broke"] / count_reflexion_active * 100) if count_reflexion_active > 0 else 0

    summary_csv_path = output_csv_name.replace("global_reflexion_report", "global_reflexion_summary")

    summary_data = [
        {"Metric": "Total Samples Processed", "Value": len(df_results)},
        {"Metric": "Reflexion Active Samples", "Value": count_reflexion_active},
        {"Metric": "Valid Coherence Scores Count", "Value": count_valid_scores},
        {"Metric": "Avg Coherence Score (0-10)", "Value": round(avg_coherence, 2)},
        {"Metric": "Self-Correction Success Rate (%)", "Value": round(success_rate, 2)},
        {"Metric": "Harmful Correction Rate (%)", "Value": round(harmful_rate, 2)},
        {"Metric": "Draft Incorrect Count (Start Wrong)", "Value": count_wrong_start},
        {"Metric": "Matrix - Fixed", "Value": reflexion_outcomes["Fixed"]},
        {"Metric": "Matrix - Broke", "Value": reflexion_outcomes["Broke"]},
        {"Metric": "Matrix - Stable", "Value": reflexion_outcomes["Stable"]},
        {"Metric": "Matrix - Inert", "Value": reflexion_outcomes["Inert"]}
    ]

    pd.DataFrame(summary_data).to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 60)
    print("🧠 GLOBAL REFLEXION ANALYSIS")
    print("=" * 60)
    print(f"• Active Samples:      {count_reflexion_active}")
    print(f"• Avg Coherence Score: {avg_coherence:.2f} / 10 (Process Quality)")
    print(f"• Correction Success:  {success_rate:.1f}% (Utility: Wrong -> Right)")
    print(f"• Harmful Corrections: {harmful_rate:.1f}% (Safety Risk: Right -> Wrong)")
    print(f"• Matrix: {reflexion_outcomes}")
    print(f"• Details Saved: {output_csv_name}")

    # Final Save
    pd.DataFrame(eval_results).to_csv(output_csv_name, index=False)


def run_rag_eval(input_csv_path):
    # 1. Setup
    api_key = os.getenv("API_KEY")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv_name = f"./result/global_rag_report_{timestamp}.csv"

    # Load Knowledge Base
    ds = load_dataset("flaviagiammarino/vqa-rad")
    ds_kb = ds['train']
    ds_query = ds['test']

    df_results = pd.read_csv(input_csv_path)
    eval_results = []

    # Accumulators
    total_faithfulness = 0
    total_relevancy = 0  # Avg of (Relevant Items / k)
    total_confidence = 0  # Avg of System Confidence Scores
    count_rag_active = 0

    print(f"\n▶️ Starting Global RAG Analysis on {len(df_results)} samples...")

    for index, row in tqdm(df_results.iterrows(), total=len(df_results)):
        row_eval = row.to_dict()

        # 1. Get string, default to empty
        id_str = str(row.get('retrieved_ids', ''))
        score_str = str(row.get('retrieved_scores', ''))

        # 2. Strip brackets and split
        # This turns "[1, 2]" -> "1, 2" -> ["1", " 2"]
        id_list_raw = id_str.replace('[', '').replace(']', '').split(',')
        score_list_raw = score_str.replace('[', '').replace(']', '').split(',')

        # 3. Convert to Int/Float (filtering out empty strings)
        retrieved_ids = [int(x) for x in id_list_raw if x.strip().isdigit()]
        retrieved_scores = [float(x) for x in score_list_raw if x.strip()]

        # Only process if retrieval happened
        if retrieved_ids:

            # --- 1. SYSTEM CONFIDENCE (Calibration) ---
            sys_conf = sum(retrieved_scores) / len(retrieved_scores)
            row_eval["rag_system_confidence"] = sys_conf
            total_confidence += sys_conf

            # --- 2. PREPARE MULTIMODAL DATA ---
            try:
                # Assuming simple index mapping. Adjust 'query_idx' if your CSV uses specific IDs.
                query_idx = index
                query_item = ds_query[query_idx]
                query_img = query_item['image'].convert("RGB")
                query_txt = row['question']

                relevant_count = 0
                retrieved_texts = []
                relevancy_details = []

                # --- 3. CONTEXTUAL RELEVANCY (Per Item) ---
                for r_id in retrieved_ids:
                    # Fetch Retrieved Item using ID
                    r_item = ds_kb[int(r_id)]
                    r_img = r_item['image'].convert("RGB")
                    r_txt = f"Q: {r_item['question']} A: {r_item['answer']}"
                    retrieved_texts.append(r_txt)

                    # Judge Pair
                    is_rel, rel_raw = check_relevancy_single(query_txt, query_img, r_txt, r_img, api_key)

                    if is_rel:
                        relevant_count += 1
                    relevancy_details.append(rel_raw)

                # Calculate Score for this Sample
                sample_relevancy = relevant_count / len(retrieved_ids)
                row_eval["rag_context_relevancy"] = sample_relevancy
                row_eval["rag_relevancy_details"] = str(relevancy_details)
                total_relevancy += sample_relevancy

                # --- 4. FAITHFULNESS (Text Only) ---
                final_ans = str(row.get('raw_prediction', ''))
                is_faithful, faith_raw = check_faithfulness(query_txt, retrieved_texts, final_ans, api_key)

                row_eval["rag_faithfulness"] = 1.0 if is_faithful else 0.0
                row_eval["rag_faithfulness_raw"] = str(faith_raw)
                if is_faithful: total_faithfulness += 1

                count_rag_active += 1

            except Exception as e:
                row_eval["rag_error"] = str(e)

        else:
            row_eval["rag_status"] = "No Retrieval"

        eval_results.append(row_eval)
        pd.DataFrame(eval_results).to_csv(output_csv_name, index=False)

    avg_relevancy = (total_relevancy / count_rag_active * 100) if count_rag_active else 0
    avg_faithfulness = (total_faithfulness / count_rag_active * 100) if count_rag_active else 0
    avg_confidence = (total_confidence / count_rag_active) if count_rag_active else 0

    summary_csv_path = output_csv_name.replace("global_rag_report", "global_rag_summary")

    summary_data = [
        {"Metric": "Total RAG Samples", "Value": count_rag_active},
        {"Metric": "Avg Context Relevancy (%)", "Value": round(avg_relevancy, 2)},
        {"Metric": "Avg Faithfulness (%)", "Value": round(avg_faithfulness, 2)},
        {"Metric": "Avg System Confidence (0-1)", "Value": round(avg_confidence, 4)},
    ]
    pd.DataFrame(summary_data).to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 60)
    print("🔎 GLOBAL RAG ANALYSIS")
    print("=" * 60)
    print(f"• Active Samples:    {count_rag_active}")
    print(f"• Context Relevancy: {avg_relevancy:.1f}% (Retrieval Quality)")
    print(f"• Faithfulness:      {avg_faithfulness:.1f}% (Hallucination Check)")
    print(f"• System Confidence: {avg_confidence:.4f}")
    print(f"• Details Saved:     {output_csv_name}")

    # Final Save
    pd.DataFrame(eval_results).to_csv(output_csv_name, index=False)


def run_global_summary(input_pattern="./result/global_rag_summary_part*.csv"):
    """
    Aggregates multiple summary CSV parts into a single global report using weighted averages.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"./result/global_rag_summary_final_{timestamp}.csv"
    # 1. Find all summary parts
    file_paths = glob.glob(input_pattern)
    if not file_paths:
        print("❌ No summary files found.")
        return

    print(f"found {len(file_paths)} parts. Aggregating...")

    total_count = 0
    weighted_relevancy_sum = 0
    weighted_faithfulness_sum = 0
    weighted_confidence_sum = 0

    # 2. Iterate and accumulate weighted sums
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)

            # Extract values safely by Metric name
            count = df.loc[df['Metric'] == 'Total RAG Samples', 'Value'].values[0]
            avg_relevancy = df.loc[df['Metric'] == 'Avg Context Relevancy (%)', 'Value'].values[0]
            avg_faithfulness = df.loc[df['Metric'] == 'Avg Faithfulness (%)', 'Value'].values[0]
            avg_confidence = df.loc[df['Metric'] == 'Avg System Confidence (0-1)', 'Value'].values[0]

            # Add to totals (Value * Count = Sum of all items in that batch)
            total_count += count
            weighted_relevancy_sum += (avg_relevancy * count)
            weighted_faithfulness_sum += (avg_faithfulness * count)
            weighted_confidence_sum += (avg_confidence * count)

        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")

    # 3. Calculate Global Weighted Averages
    if total_count > 0:
        global_avg_relevancy = weighted_relevancy_sum / total_count
        global_avg_faithfulness = weighted_faithfulness_sum / total_count
        global_avg_confidence = weighted_confidence_sum / total_count
    else:
        global_avg_relevancy = 0
        global_avg_faithfulness = 0
        global_avg_confidence = 0

    # 4. Format Output
    summary_data = [
        {"Metric": "Total RAG Samples", "Value": int(total_count)},
        {"Metric": "Avg Context Relevancy (%)", "Value": round(global_avg_relevancy, 2)},
        {"Metric": "Avg Faithfulness (%)", "Value": round(global_avg_faithfulness, 2)},
        {"Metric": "Avg System Confidence (0-1)", "Value": round(global_avg_confidence, 4)},
    ]

    # 5. Save
    df_final = pd.DataFrame(summary_data)
    df_final.to_csv(output_file, index=False)

    print("\n" + "=" * 40)
    print("🌍 GLOBAL AGGREGATED SUMMARY")
    print("=" * 40)
    print(df_final.to_string(index=False))
    print(f"\n✅ Saved to: {output_file}")


def generate_matrix_map(csv_path):
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    output_image = "./result/matrix_global_reflexion_summary_20260120_203328.png"

    # 2. Extract Matrix Values safely
    def get_val(metric_name):
        row = df[df['Metric'] == metric_name]
        return int(row['Value'].values[0]) if not row.empty else 0

    fixed = get_val('Matrix - Fixed')  # Wrong -> Right
    broke = get_val('Matrix - Broke')  # Right -> Wrong
    stable = get_val('Matrix - Stable')  # Right -> Right
    inert = get_val('Matrix - Inert')  # Wrong -> Wrong

    # 3. Construct 2x2 Matrix Data
    # Rows: Initial Draft Quality (Correct, Incorrect)
    # Cols: Final Answer Quality (Correct, Incorrect)
    matrix_counts = np.array([
        [stable, broke],  # Draft Correct
        [fixed, inert]  # Draft Incorrect
    ])

    # 4. Define Custom Annotations (Label + Count)
    labels = np.array([
        [f"STABLE\n(Right → Right)\n{stable}", f"BROKE\n(Right → Wrong)\n{broke}"],
        [f"FIXED\n(Wrong → Right)\n{fixed}", f"INERT\n(Wrong → Wrong)\n{inert}"]
    ])

    # 5. Plot Heatmap
    plt.figure(figsize=(8, 7))
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.2)

    # Custom Color Map (Green for Good, Red for Bad, Grey for Neutral)
    # Note: Using a single cmap 'Blues' is cleaner for counts, but you can customize.
    ax = sns.heatmap(
        matrix_counts,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=2,
        linecolor='black',
        square=True,
        xticklabels=["Correct", "Incorrect"],
        yticklabels=["Correct", "Incorrect"]
    )

    # 6. Formatting
    plt.xlabel("Final Answer Quality", fontsize=13, labelpad=10, fontweight='bold')
    plt.ylabel("Initial Draft Quality", fontsize=13, labelpad=10, fontweight='bold')
    plt.title("Reflexion Safety Matrix", fontsize=15, pad=20, fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # 7. Save
    plt.savefig(output_image, dpi=300)
    print(f"✅ Matrix Map generated: {output_image}")


if __name__ == "__main__":
    # csv_path = "./result/results_RAG+Reflexion+Instruct_QLoRA-Qwen3-4B_20260114_071430.csv"
    # run_reflexion_eval(csv_path)
    # run_rag_eval(csv_path)
    # run_global_summary()
    generate_matrix_map("./result/global_reflexion_summary_20260120_203328.csv")
