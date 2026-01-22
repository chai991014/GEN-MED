import pandas as pd
import os
import time
from datetime import datetime
import glob
from tqdm import tqdm
from sklearn.metrics import f1_score
import google.generativeai as genai
from groq import Groq
from openai import OpenAI
import evaluate  # pip install evaluate bert-score
import re
import string
import json

# ==========================================
# 1. KEYS & CLIENT INITIALIZATION
# ==========================================
KEYS = {
    "GOOGLE": "",
    "GROQ": "",
    "DEEPSEEK": ""
}

# Local BioBERTScore Setup
BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.2"
bert_metric = evaluate.load("bertscore")


def normalize_text(text):
    """Normalizes text for Exact Match (Lower, strip punctuation)."""
    if not isinstance(text, str):
        return str(text).lower().strip()
    text = text.lower().strip()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def check_exact_match(pred, gt):
    """Returns 1 if normalized prediction matches ground truth, else 0."""
    return 1 if normalize_text(pred) == normalize_text(gt) else 0


def extract_binary_verdict(text):
    if not text:
        return 0

    # 1. Look for the very last digit in the text (often where the verdict is)
    digits = re.findall(r'\d', text)
    if digits:
        last_digit = int(digits[-1])
        if last_digit in [0, 1]:
            return last_digit

    # 2. Fallback: Map common medical/eval keywords
    text_clean = text.lower().strip()
    positive_keywords = ['correct', 'match', 'yes', 'true', 'accurate']
    negative_keywords = ['incorrect', 'mismatch', 'no', 'false', 'wrong']

    for word in positive_keywords:
        if word in text_clean:
            return 1
    for word in negative_keywords:
        if word in text_clean:
            return 0

    # 3. Default to 0 if completely ambiguous
    return 0


def parse_json_verdicts(text, expected_keys):
    """Fuzzy matches LLM keys back to original engine names to fix Llama/JSON bugs."""
    if not isinstance(text, str) or not text.strip():
        return {k: 0 for k in expected_keys}
    try:
        # Extract JSON block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match: return {k: 0 for k in expected_keys}
        data = json.loads(match.group())

        aligned = {}
        for original in expected_keys:
            # Match logic: remove 'pred_', symbols, and lowercase everything
            norm_orig = original.lower().replace("pred_", "").replace("+", "").replace("_", "").replace("-", "")
            val = 0
            for k_llm, v_llm in data.items():
                norm_llm = str(k_llm).lower().replace("pred_", "").replace("+", "").replace("_", "").replace("-", "")
                if norm_llm == norm_orig or norm_llm in norm_orig or norm_orig in norm_llm:
                    val = v_llm
                    break
            aligned[original] = val
        return aligned
    except:
        return {k: 0 for k in expected_keys}


class EnsembleEngine:
    def __init__(self):
        # Judge 1: Gemini 3 Flash (1,500 RPD)
        genai.configure(api_key=KEYS["GOOGLE"])
        self.gemini = genai.GenerativeModel("gemini-2.5-flash-lite")

        # Judge 2: Llama 3.3 70B (1,000 RPD)
        self.groq = Groq(api_key=KEYS["GROQ"])

        # Judge 3: DeepSeek-R1 (Official API)
        self.deepseek_client = OpenAI(
            api_key=KEYS["DEEPSEEK"],
            base_url="https://api.deepseek.com"
        )

    def get_prompt(self, question, gt, pred, q_type):
        """Your improved differentiated prompt logic"""
        if q_type == "CLOSED":
            task_desc = "For this Yes/No question, does the model response signify the same answer as the ground truth?"
        else:
            task_desc = "For this medical description, is the core clinical finding in the model response the same as the ground truth?"

        return (
            f"System: You are a medical evaluation expert.\n"
            f"Question: {question}\n"
            f"Ground Truth: {gt}\n"
            f"Model Response: {pred}\n\n"
            f"Task: {task_desc}\n"
            f"Constraint: Output ONLY '1' for Correct or '0' for Incorrect."
        )

    def judge_all(self, q, gt, p, t):
        prompt = self.get_prompt(q, gt, p, t)
        results = {'Gemini_Raw': "", 'Gemini': 0, 'Llama_Raw': "", 'Llama': 0, 'Deepseek_Raw': "", 'Deepseek': 0}

        # Llama Call
        try:
            r = self.groq.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            results['Llama_Raw'] = r.choices[0].message.content
            results['Llama'] = extract_binary_verdict(r.choices[0].message.content)
        except Exception as e:
            print(f"❌ Llama Error: {e}")

        # Gemini Call
        try:
            r = self.gemini.generate_content(prompt)
            results['Gemini_Raw'] = r.text
            results['Gemini'] = extract_binary_verdict(r.text)
        except Exception as e:
            print(f"❌ Gemini Error: {e}")

        # Deepseek Call
        try:
            r = self.deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}]
            )
            content = r.choices[0].message.content
            # Extract verdict after the <think> tags
            results["Deepseek_Raw"] = content
            final_ans = content.split("</think>")[-1]
            results['Deepseek'] = extract_binary_verdict(final_ans)
        except Exception as e:
            print(f"❌ Deepseek Error: {e}")

        return results

    def get_batch_prompt(self, q, gt, preds_dict, q_type):

        if q_type == "CLOSED":
            task_desc = "For this Yes/No question, do these model responses signify the same answer as the ground truth?"
        else:
            task_desc = "For these medical descriptions, is the core clinical finding in the model response the same as the ground truth?"

        preds_list = "\n".join([f"- {k}: {v}" for k, v in preds_dict.items()])

        return (
            f"System: You are a medical evaluation expert.\n"
            f"Question: {q}\n"
            f"Ground Truth: {gt}\n\n"
            f"Evaluate these model responses:\n{preds_list}\n\n"
            f"Task: {task_desc}\n"
            f"Constraint: Respond ONLY with a JSON object where keys are the engine names and values are 1 (Correct) or 0 (Incorrect)."
        )

    def judge_batch(self, q, gt, preds_dict, q_type):
        prompt = self.get_batch_prompt(q, gt, preds_dict, q_type)
        raw_responses = {'Gemini_Raw': "", 'Llama_Raw': "", 'Deepseek_Raw': ""}
        final_verdicts = {name: {'Gemini': 0, 'Llama': 0, 'Deepseek': 0} for name in preds_dict.keys()}

        # 1. Gemini Judge
        try:
            res = self.gemini.generate_content(prompt).text
            raw_responses['Gemini_Raw'] = res
            v = parse_json_verdicts(res, preds_dict.keys())
            for k in preds_dict.keys():
                final_verdicts[k]['Gemini'] = v.get(k, 0)
        except Exception as e:
            print(f"Gemini Err: {e}")

        # 2. Llama Judge
        try:
            res = self.groq.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            ).choices[0].message.content
            raw_responses['Llama_Raw'] = res
            v = parse_json_verdicts(res, preds_dict.keys())
            for k in preds_dict.keys():
                final_verdicts[k]['Llama'] = v.get(k, 0)
        except Exception as e:
            print(f"Llama Err: {e}")

        # 3. DeepSeek Judge
        max_retries = 5
        for attempt in range(max_retries):
            try:
                res = self.deepseek_client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[{"role": "user", "content": prompt}]
                ).choices[0].message.content
                raw_responses['Deepseek_Raw'] = res
                content_after_think = res.split("</think>")[-1]
                v = parse_json_verdicts(content_after_think, preds_dict.keys())
                for k in preds_dict.keys():
                    final_verdicts[k]['Deepseek'] = v.get(k, 0)
                break
            except Exception as e:
                print(f"DeepSeek Connection attempt {attempt + 1} failed: {e}")
                time.sleep(5)  # Wait 5 seconds before retrying
                if attempt == max_retries - 1:
                    raw_responses['Deepseek_Raw'] = ""  # Fail gracefully

        return final_verdicts, raw_responses


# ==========================================
# 2. PROCESSING PIPELINE
# ==========================================
def run_batch_pipeline():
    engine = EnsembleEngine()
    input_files = glob.glob("./result/result-mevf-1/results_*.csv")
    output_dir = "./result/processed-mevf-1"
    os.makedirs(output_dir, exist_ok=True)

    # --- STEP 1: MERGE ALL FILES ---
    merged_df = None
    for f in input_files:
        name = os.path.basename(f).replace("results_", "").replace(".csv", "")
        df = pd.read_csv(f)

        cols_to_drop = ['reflexion_draft', 'reflexion_critique', 'retrieved_ids', 'retrieved_scores']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

        df = df.rename(columns={'raw_prediction': f'pred_{name}'})
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df[['id', f'pred_{name}']], on='id', how='outer')

    # --- STEP 2: BATCH EVALUATION ---
    pred_cols = [c for c in merged_df.columns if c.startswith('pred_') and not any(x in c for x in ['_Gemini', '_Llama', '_Deepseek', '_final'])]

    # Pre-allocate columns to avoid fragmentation
    for c in pred_cols:
        for suffix in ['_Gemini', '_Llama', '_Deepseek', '_final_prediction']:
            if f"{c}{suffix}" not in merged_df.columns:
                merged_df[f"{c}{suffix}"] = None
    for raw_col in ['Gemini_Raw', 'Llama_Raw', 'Deepseek_Raw']:
        if raw_col not in merged_df.columns:
            merged_df[raw_col] = None

    for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Evaluating Batch"):
        preds_dict = {c: row[c] for c in pred_cols}
        batch_verdicts, batch_raw = engine.judge_batch(row['question'], row['ground_truth'], preds_dict, row['type'])

        merged_df.at[idx, 'Gemini_Raw'] = batch_raw['Gemini_Raw']
        merged_df.at[idx, 'Llama_Raw'] = batch_raw['Llama_Raw']
        merged_df.at[idx, 'Deepseek_Raw'] = batch_raw['Deepseek_Raw']

        for c in pred_cols:
            scores = batch_verdicts[c]
            merged_df.at[idx, f"{c}_Gemini"] = scores['Gemini']
            merged_df.at[idx, f"{c}_Llama"] = scores['Llama']
            merged_df.at[idx, f"{c}_Deepseek"] = scores['Deepseek']

            majority = 1 if (scores['Gemini'] + scores['Llama'] + scores['Deepseek']) >= 2 else 0
            merged_df.at[idx, f"{c}_final_prediction"] = majority

        merged_df.to_csv(os.path.join(output_dir, "batch_processed_results_checkpoint.csv"), index=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_path = os.path.join(output_dir, f"batch_processed_results_{timestamp}.csv")
    merged_df.to_csv(final_results_path, index=False)


def run_mevf_pipeline():
    engine = EnsembleEngine()
    # Updated path to match your request
    input_files = glob.glob("./result/result-mevf/results_*.csv")
    output_dir = "result/processed-mevf"

    # --- STEP 1: MERGE ALL FILES (Logic from CODE B) ---
    merged_df = None

    for f in input_files:
        # Create a unique name for the prediction column based on filename
        name = os.path.basename(f).replace("results_", "").replace(".csv", "")

        df = pd.read_csv(f)

        # Drop unnecessary columns if they exist
        cols_to_drop = ['reflexion_draft', 'reflexion_critique', 'retrieved_ids', 'retrieved_scores']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

        # Rename the raw prediction to a unique column name
        df = df.rename(columns={'raw_prediction': f'pred_{name}'})

        if merged_df is None:
            merged_df = df
        else:
            # Merge on ID (assuming all files share the same Question IDs)
            # We keep 'question', 'ground_truth', 'type' from the first file
            merged_df = pd.merge(merged_df, df[['id', f'pred_{name}']], on='id', how='outer')

    # --- STEP 2: PREPARE COLUMNS (Logic from CODE B) ---
    # Identify all prediction columns we just created (e.g., pred_file1, pred_file2)
    pred_cols = [c for c in merged_df.columns if c.startswith('pred_')]

    # Pre-allocate result columns to avoid fragmentation
    for c in pred_cols:
        # Create columns for Judge verdicts and Final Prediction
        for suffix in ['_Gemini', '_Llama', '_Deepseek', '_final_prediction']:
            col_name = f"{c}{suffix}"
            if col_name not in merged_df.columns:
                merged_df[col_name] = None

    # Pre-allocate Raw Judge Response columns (Global, not per prediction)
    for raw_col in ['Gemini_Raw', 'Llama_Raw', 'Deepseek_Raw']:
        if raw_col not in merged_df.columns:
            merged_df[raw_col] = None

    # Checkpoint path
    checkpoint_path = os.path.join(output_dir, "batch_processed_checkpoint.csv")

    # --- STEP 3: BATCH EVALUATION LOOP ---
    print("Starting Batch Evaluation...")

    for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Evaluating Batch"):
        q_type = str(row['type']).upper()

        # Gather all predictions for this row into a dictionary
        # Format: {'pred_file1': 'answer A', 'pred_file2': 'answer B'}
        preds_dict = {c: row[c] for c in pred_cols}

        if q_type == 'CLOSED':
            # --- LOGIC FROM CODE A (Modified for Batch) ---
            # STRATEGY A: EXACT MATCH (No API Call)
            # We must loop through EVERY prediction column locally

            for col_name, pred_value in preds_dict.items():
                # Perform Exact Match locally
                match_result = check_exact_match(pred_value, row['ground_truth'])

                # Update Final Prediction immediately
                merged_df.at[idx, f"{col_name}_final_prediction"] = match_result

                # Set Judge columns to None (or explicit value) as we didn't use them
                merged_df.at[idx, f"{col_name}_Gemini"] = None
                merged_df.at[idx, f"{col_name}_Llama"] = None
                merged_df.at[idx, f"{col_name}_Deepseek"] = None

            # Raw explanation columns can be left Empty or set to "CLOSED - NO JUDGE"
            merged_df.at[idx, 'Gemini_Raw'] = "CLOSED"
            merged_df.at[idx, 'Llama_Raw'] = "CLOSED"
            merged_df.at[idx, 'Deepseek_Raw'] = "CLOSED"

        else:
            # --- LOGIC FROM CODE B (LLM Judge) ---
            # STRATEGY B: BATCH LLM JUDGE (OPEN ONLY)
            # Sends all predictions to the judge in one prompt (if supported) or iterates internally

            # engine.judge_batch expects: question, gt, dict_of_predictions, type
            batch_verdicts, batch_raw = engine.judge_batch(row['question'], row['ground_truth'], preds_dict,
                                                           row['type'])

            # 1. Update Raw Explanations (Global for the row)
            merged_df.at[idx, 'Gemini_Raw'] = batch_raw.get('Gemini_Raw')
            merged_df.at[idx, 'Llama_Raw'] = batch_raw.get('Llama_Raw')
            merged_df.at[idx, 'Deepseek_Raw'] = batch_raw.get('Deepseek_Raw')

            # 2. Update Scores for each prediction column
            for c in pred_cols:
                scores = batch_verdicts.get(c, {'Gemini': 0, 'Llama': 0, 'Deepseek': 0})

                merged_df.at[idx, f"{c}_Gemini"] = scores['Gemini']
                merged_df.at[idx, f"{c}_Llama"] = scores['Llama']
                merged_df.at[idx, f"{c}_Deepseek"] = scores['Deepseek']

                # Majority Vote
                vote_sum = (scores['Gemini'] or 0) + (scores['Llama'] or 0) + (scores['Deepseek'] or 0)
                merged_df.at[idx, f"{c}_final_prediction"] = 1 if vote_sum >= 2 else 0

        # --- STEP 4: SAVE CHECKPOINT (Logic from CODE B) ---
        merged_df.to_csv(checkpoint_path, index=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_path = os.path.join(output_dir, f"batch_processed_results_{timestamp}.csv")
    merged_df.to_csv(final_results_path, index=False)
    print(f"\n✅ All files processed. Saved to {final_results_path}")


def process_batch_closed_logic():
    input_file = "result/processed-mevf/batch_processed_results_20260115_131934.csv"
    output_dir = "result/processed-mevf"

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)

    # 1. Identify Base Prediction Columns
    # These are columns starting with 'pred_' but NOT ending with judge suffixes
    all_cols = df.columns
    suffixes = ['_Gemini', '_Llama', '_Deepseek', '_final_prediction']
    base_pred_cols = []

    for col in all_cols:
        if col.startswith('pred_'):
            is_derived = False
            for suffix in suffixes:
                if col.endswith(suffix):
                    is_derived = True
                    break
            if not is_derived:
                base_pred_cols.append(col)

    print(f"Identified prediction sets: {base_pred_cols}")

    # 2. Iterate and Apply Logic
    for idx, row in df.iterrows():
        q_type = str(row['type']).upper()

        if q_type == 'CLOSED':
            # --- LOGIC FROM CODE A (Modified for Batch) ---
            # STRATEGY A: EXACT MATCH (No API Call)

            # Loop through EVERY prediction column locally
            for col_name in base_pred_cols:
                pred_value = row[col_name]

                # Perform Exact Match locally
                match_result = check_exact_match(pred_value, row['ground_truth'])

                # Update Final Prediction immediately
                df.at[idx, f"{col_name}_final_prediction"] = match_result

                # Set Judge columns to None (Clear them)
                df.at[idx, f"{col_name}_Gemini"] = None
                df.at[idx, f"{col_name}_Llama"] = None
                df.at[idx, f"{col_name}_Deepseek"] = None

            # Set Raw explanation columns to "CLOSED"
            df.at[idx, 'Gemini_Raw'] = "CLOSED"
            df.at[idx, 'Llama_Raw'] = "CLOSED"
            df.at[idx, 'Deepseek_Raw'] = "CLOSED"

    # 3. Save Result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_path = os.path.join(output_dir, f"batch_processed_results_{timestamp}.csv")
    df.to_csv(final_results_path, index=False)
    print(f"\n✅ All files processed. Saved to {final_results_path}")


def run_recovery():
    INPUT_FILE = "result/processed-agent/batch_processed_results_20260110_062157.csv"
    OUTPUT_DIR = "result/processed-agent/"

    df = pd.read_csv(INPUT_FILE)

    pred_cols = [c for c in df.columns if
                 c.startswith('pred_') and not any(x in c for x in ['_Gemini', '_Llama', '_Deepseek', '_final'])]

    print(f"Repairing {len(df)} rows for engines: {pred_cols}")

    # 1. RE-PARSE RAW DATA
    for idx, row in df.iterrows():
        # Parse each judge separately
        g_verdicts = parse_json_verdicts(row['Gemini_Raw'], pred_cols)
        l_verdicts = parse_json_verdicts(row['Llama_Raw'], pred_cols)
        d_verdicts = parse_json_verdicts(row['Deepseek_Raw'], pred_cols)

        for c in pred_cols:
            g, l, d = g_verdicts.get(c, 0), l_verdicts.get(c, 0), d_verdicts.get(c, 0)
            df.at[idx, f"{c}_Gemini"] = g
            df.at[idx, f"{c}_Llama"] = l
            df.at[idx, f"{c}_Deepseek"] = d
            # Recalculate Majority
            df.at[idx, f"{c}_final_prediction"] = 1 if (g + l + d) >= 2 else 0

    # 2. RE-CALCULATE SUMMARY (Using same logic as previous)
    summary_data = []
    for c in pred_cols:
        engine_name = c.replace("pred_", "")
        closed_df = df[df['type'] == 'CLOSED']
        open_df = df[df['type'] == 'OPEN']

        # for MEVF CLOSED ignore llm judge column
        # closed_acc = closed_df[f"{c}_final_prediction"].mean() * 100 if not closed_df.empty else 0

        # F1 Score
        gt_binary = closed_df['ground_truth'].str.lower().map({'yes': 1, 'no': 0}).fillna(0)
        f1_closed = f1_score(gt_binary, closed_df[f"{c}_final_prediction"].astype(int)) if not closed_df.empty else 0

        # BioBERTScore
        bio_score = 0
        if not open_df.empty:
            b_res = bert_metric.compute(
                predictions=open_df[c].astype(str).tolist(),
                references=open_df['ground_truth'].astype(str).tolist(),
                model_type=BIOBERT_MODEL, num_layers=9
            )
            bio_score = sum(b_res['f1']) / len(b_res['f1']) * 100

        summary_data.append({
            "File": f"results_{engine_name}.csv",
            "Closed_Acc_Gemini": closed_df[f"{c}_Gemini"].mean() * 100,
            "Closed_Acc_Llama": closed_df[f"{c}_Llama"].mean() * 100,
            "Closed_Acc_Deepseek": closed_df[f"{c}_Deepseek"].mean() * 100,
            "Closed_Avg_Acc": closed_df[[f"{c}_Gemini", f"{c}_Llama", f"{c}_Deepseek"]].mean(axis=1).mean() * 100,
            # "Closed_Avg_Acc": closed_acc,
            "Closed_F1_Majority": f1_closed * 100,
            "Open_Acc_Gemini": open_df[f"{c}_Gemini"].mean() * 100,
            "Open_Acc_Llama": open_df[f"{c}_Llama"].mean() * 100,
            "Open_Acc_Deepseek": open_df[f"{c}_Deepseek"].mean() * 100,
            "Open_Avg_Acc": open_df[[f"{c}_Gemini", f"{c}_Llama", f"{c}_Deepseek"]].mean(axis=1).mean() * 100,
            "Open_BioBERT": bio_score
        })

    # 3. SAVE
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_path = os.path.join(OUTPUT_DIR, f"batch_processed_results_{timestamp}.csv")
    df.to_csv(final_results_path, index=False)

    pd.DataFrame(summary_data).to_csv(os.path.join(OUTPUT_DIR, f"processed_results_summary_{timestamp}.csv"),
                                      index=False)
    print("Done. Check result/processed/ directory.")


def calculate_judge_error_summary():
    INPUT_FILE = "./result/judge-error-rate/batch_processed_results_*.csv"
    OUTPUT_DIR = "result/judge-error-rate"

    files = glob.glob(INPUT_FILE)
    if not files:
        print(f"No files found matching: {INPUT_FILE}")
        return

    print(f"Found {len(files)} files. Scanning for configurations...")

    results = []

    # Global accumulators for the final Summary row
    grand_totals = {
        'Gemini': {'err': 0, 'decision_count': 0},
        'Llama': {'err': 0, 'decision_count': 0},
        'Deepseek': {'err': 0, 'decision_count': 0},
        'questions': 0
    }

    # PROCESS EACH FILE
    for f_path in files:
        try:
            df = pd.read_csv(f_path)
            filename = os.path.basename(f_path)

            # Identify unique "base" prediction names (configurations)
            # We assume any column ending in '_final_prediction' represents a valid config group.
            base_cols = set()
            for c in df.columns:
                if c.endswith('_final_prediction'):
                    # The base name is everything before '_final_prediction'
                    # e.g. 'pred_RAG04+Basic_LLaVA-Med_20260108_050420'
                    base_name = c[:-len('_final_prediction')]
                    base_cols.add(base_name)

            sorted_base_cols = sorted(list(base_cols))

            # Loop through EACH configuration found in this single file
            for base in sorted_base_cols:
                # Construct column names
                col_g = f"{base}_Gemini"
                col_l = f"{base}_Llama"
                col_d = f"{base}_Deepseek"
                col_final = f"{base}_final_prediction"

                # Check if all required columns exist
                if not all(c in df.columns for c in [col_g, col_l, col_d, col_final]):
                    print(f"  ⚠️ Skipping configuration {base} in {filename} (missing judge columns)")
                    continue

                # Get values (fill NaNs with 0)
                g_vals = df[col_g].fillna(0).astype(int)
                l_vals = df[col_l].fillna(0).astype(int)
                d_vals = df[col_d].fillna(0).astype(int)
                final_pred = df[col_final].fillna(0).astype(int)

                count_rows = len(df)  # Questions in this batch

                # Calculate Errors (Compare Judge vs Final)
                err_g = (g_vals != final_pred).sum()
                err_l = (l_vals != final_pred).sum()
                err_d = (d_vals != final_pred).sum()

                # Build Row Data
                # First column is the "name of the row" (the configuration name)
                row_data = {
                    "Model Configuration": base,
                    "Total_Questions": count_rows
                }

                # Update Grand Totals (Summary Logic)
                grand_totals['questions'] += count_rows

                # Gemini Stats
                row_data["error_rate_Gemini"] = (err_g / count_rows) if count_rows > 0 else 0.0
                grand_totals['Gemini']['err'] += err_g
                grand_totals['Gemini']['decision_count'] += count_rows

                # Llama Stats
                row_data["error_rate_Llama"] = (err_l / count_rows) if count_rows > 0 else 0.0
                grand_totals['Llama']['err'] += err_l
                grand_totals['Llama']['decision_count'] += count_rows

                # Deepseek Stats
                row_data["error_rate_Deepseek"] = (err_d / count_rows) if count_rows > 0 else 0.0
                grand_totals['Deepseek']['err'] += err_d
                grand_totals['Deepseek']['decision_count'] += count_rows

                results.append(row_data)

        except Exception as e:
            print(f"Error processing {f_path}: {e}")

    # 3. CALCULATE SUMMARY ROW
    summary_row = {
        "Model Configuration": "Summary",
        "Total_Questions": grand_totals['questions']
    }

    for llm in ['Gemini', 'Llama', 'Deepseek']:
        g_count = grand_totals[llm]['decision_count']
        g_err = grand_totals[llm]['err']
        g_rate = (g_err / g_count) if g_count > 0 else 0.0
        summary_row[f"error_rate_{llm}"] = g_rate

    results.append(summary_row)

    # 4. SAVE OUTPUT
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_results_summary_judge_error_{timestamp}.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    out_df = pd.DataFrame(results)

    # Reorder columns: Name, Total_Questions, then the error rates
    cols = ["Model Configuration", "Total_Questions"] + [c for c in out_df.columns if
                                                         c not in ["Model Configuration", "Total_Questions"]]
    out_df = out_df[cols]

    out_df.to_csv(output_path, index=False)

    print(f"\n✅ Processing Complete.")
    print(f"   Saved to: {output_path}")


def calculate_judge_error_summary_v2():
    INPUT_DIR = "result/judge-error-rate"
    OUTPUT_DIR = "result/judge-error-rate"

    # 1. DEFINE PATTERNS
    patterns = [
        os.path.join(INPUT_DIR, "batch_processed_results_*.csv"),
        os.path.join(INPUT_DIR, "processed_results_results_*.csv")
    ]

    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = list(set(files))

    if not files:
        print(f"No files found in {INPUT_DIR}.")
        return

    # --- SPECIFIC FILES CONFIGURATION ---
    # These files ONLY run judge on OPEN questions.
    # All other files will be processed completely (OPEN + CLOSED).
    OPEN_ONLY_FILENAMES = {
        "batch_processed_results_20260116_105722.csv",
        "processed_results_results_MEVF+BAN_20251218_022816.csv",
        "processed_results_results_MEVF+SAN_20251218_031842.csv"
    }

    print(f"Found {len(files)} files. Processing...")

    results = []
    grand_totals = {
        'Gemini': {'err': 0, 'decision_count': 0},
        'Llama': {'err': 0, 'decision_count': 0},
        'Deepseek': {'err': 0, 'decision_count': 0},
        'questions': 0
    }

    for f_path in files:
        try:
            df = pd.read_csv(f_path)
            filename = os.path.basename(f_path)

            # --- CONDITIONAL FILTERING ---
            # Only filter for OPEN if the file is in the specific list
            if filename in OPEN_ONLY_FILENAMES:
                if 'type' in df.columns:
                    df = df[df['type'] == 'OPEN']
                # If 'type' column missing, we assume it's already filtered or process as is

            if df.empty:
                print(f"  ⚠️ Skipping {filename} (Empty after filter)")
                continue

            # --- DETECT MODE (BATCH vs SINGLE) ---
            batch_cols = [c for c in df.columns if c.startswith('pred_') and c.endswith('_final_prediction')]
            configs = []

            if batch_cols:
                # Mode A: Batch File
                base_names = set(c[:-len('_final_prediction')] for c in batch_cols)
                for base in sorted(base_names):
                    configs.append({
                        'name': base.replace("pred_", ""),
                        'col_g': f"{base}_Gemini",
                        'col_l': f"{base}_Llama",
                        'col_d': f"{base}_Deepseek",
                        'col_final': f"{base}_final_prediction"
                    })
            else:
                # Mode B: Single File
                config_name = filename.replace("processed_results_results_", "").replace(".csv", "")
                configs.append({
                    'name': config_name,
                    'col_g': "Gemini",
                    'col_l': "Llama",
                    'col_d': "Deepseek",
                    'col_final': "final_prediction"
                })

            # --- CALCULATE METRICS ---
            for cfg in configs:
                # Check columns
                required = [cfg['col_g'], cfg['col_l'], cfg['col_d'], cfg['col_final']]
                if not all(c in df.columns for c in required):
                    continue

                # Calculate Errors
                count = len(df)

                # FillNA with 0 safely
                g_vals = df[cfg['col_g']].fillna(0).astype(int)
                l_vals = df[cfg['col_l']].fillna(0).astype(int)
                d_vals = df[cfg['col_d']].fillna(0).astype(int)
                final_pred = df[cfg['col_final']].fillna(0).astype(int)

                err_g = (g_vals != final_pred).sum()
                err_l = (l_vals != final_pred).sum()
                err_d = (d_vals != final_pred).sum()

                # Add to Results
                results.append({
                    "Model Configuration": cfg['name'],
                    "Source File": filename,
                    "Total_Questions": count,
                    "error_rate_Gemini": (err_g / count) if count > 0 else 0,
                    "error_rate_Llama": (err_l / count) if count > 0 else 0,
                    "error_rate_Deepseek": (err_d / count) if count > 0 else 0
                })

                # Add to Totals
                grand_totals['questions'] += count
                grand_totals['Gemini']['err'] += err_g
                grand_totals['Gemini']['decision_count'] += count
                grand_totals['Llama']['err'] += err_l
                grand_totals['Llama']['decision_count'] += count
                grand_totals['Deepseek']['err'] += err_d
                grand_totals['Deepseek']['decision_count'] += count

        except Exception as e:
            print(f"Error processing {f_path}: {e}")

    # SUMMARY ROW
    summary_row = {
        "Model Configuration": "Summary",
        "Source File": "ALL",
        "Total_Questions": grand_totals['questions']
    }
    for llm in ['Gemini', 'Llama', 'Deepseek']:
        tot = grand_totals[llm]['decision_count']
        err = grand_totals[llm]['err']
        summary_row[f"error_rate_{llm}"] = (err / tot) if tot > 0 else 0

    results.append(summary_row)

    # SAVE
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"processed_results_summary_judge_error_{timestamp}.csv")

    out_df = pd.DataFrame(results)

    # Reorder columns
    desired_order = ["Model Configuration", "Total_Questions", "Source File",
                     "error_rate_Gemini", "error_rate_Llama", "error_rate_Deepseek"]
    final_cols = [c for c in desired_order if c in out_df.columns]
    out_df = out_df[final_cols]

    out_df.to_csv(output_path, index=False)
    print(f"\n✅ Processing Complete. Saved to: {output_path}")


def process_summary_file():
    INPUT_FILE = "result/judge-error-rate/processed_results_summary_judge_error_20260116_112611.csv"
    OUTPUT_DIR = "result/judge-error-rate"
    df = pd.read_csv(INPUT_FILE)

    def parse_configuration(row_val):
        """
        Parses 'pred_{tech}_{model}_{date}_{time}'
        into (method, prompt, model).
        """
        # 1. Handle Summary Row
        if row_val == "Summary":
            return pd.Series(["Summary", "-", "-"])

        parts = row_val.split('_')

        # 2. Handle MEVF / Short Format Cases (len == 4)
        # Format: pred_{MEVF+SAN}_{date}_{time}
        # Requirement: tech (method/prompt) empty, keep original name in model.
        if len(parts) == 3:
            full_content = parts[0]
            # method, prompt, model
            return pd.Series([None, None, full_content])

        # 3. Handle Standard Cases (len >= 5)
        # Format: pred_{tech}_{model_name}_{date}_{time}
        if len(parts) >= 4:
            raw_tech = parts[0]

            # Extract Model Name (everything between tech and the last 2 timestamp parts)
            model_name = parts[1]

            # Split Tech into Method + Prompt
            # Logic: locate the LAST +{prompt}
            if '+' in raw_tech:
                # rsplit with maxsplit=1 splits from the right side once
                method_part, prompt_part = raw_tech.rsplit('+', 1)
            else:
                # Fallback if no + exists (unlikely based on your description)
                method_part = raw_tech
                prompt_part = None

            return pd.Series([method_part, prompt_part, model_name])

        return pd.Series([None, None, None])

    # Apply the parsing logic
    print("Parsing Model Configurations...")
    df[['method', 'prompt', 'model']] = df['Model Configuration'].apply(parse_configuration)

    # Reorder columns: method, prompt, model, then the rest
    # We remove the original 'Model Configuration' column
    metric_cols = [c for c in df.columns if c not in ['Model Configuration', 'method', 'prompt', 'model']]
    final_cols = ['model', 'method', 'prompt'] + metric_cols

    df_final = df[final_cols]

    # Save output
    output_filename = f"processed_split_{os.path.basename(INPUT_FILE)}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    df_final.to_csv(output_path, index=False)

    print(f"✅ Processing Complete. Saved to: {output_path}")
    print(df_final.head())


def calculate_score():
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # 1. Load the data
    # Replace 'input.csv' with your actual file name
    file_path = 'performance-agent.csv'
    df = pd.read_csv(file_path)
    # df_no_slake = df.iloc[:-3]

    # 2. Initialize the Scaler (for Normalization)
    scaler = StandardScaler()
    final_scaler = MinMaxScaler()

    # 3. Create Normalized Columns (Scale 0 to 1)
    # We use these for calculation but don't need to save them
    norm_data = scaler.fit_transform(df[['ce_acc', 'ce_f1', 'oe_acc', 'biobert', 'time']])
    # norm_no_slake = scaler.fit_transform(df_no_slake[['ce_acc', 'ce_f1', 'oe_acc', 'biobert', 'time']])
    df_norm = pd.DataFrame(norm_data, columns=['n_ce', 'n_f1', 'n_oe', 'n_bio', 'n_t'])
    # df_norm_no_slake = pd.DataFrame(norm_no_slake, columns=['n_ce', 'n_f1', 'n_oe', 'n_bio', 'n_t'])

    # 4. Define Weights (The "Balanced Clinical Expert" Strategy)
    w_ce_acc = 0.20  # Low weight for Base Accuracy
    w_ce_f1 = 0.30  # High weight for Safety
    w_oe_acc = 0.25  # Highest weight for Understanding
    w_biobert = 0.20  # Medium weight for Text Quality
    w_time = 0.05

    # 5. Calculate the Composite Score

    # df['score_no_slake'] = (
    #     df_norm_no_slake['n_oe'] * w_oe_acc +
    #     df_norm_no_slake['n_f1'] * w_ce_f1 +
    #     df_norm_no_slake['n_bio'] * w_biobert +
    #     df_norm_no_slake['n_ce'] * w_ce_acc +
    #     df_norm_no_slake['n_t'] * w_time
    # )

    df['score'] = (
            df_norm['n_oe'] * w_oe_acc +
            df_norm['n_f1'] * w_ce_f1 +
            df_norm['n_bio'] * w_biobert +
            df_norm['n_ce'] * w_ce_acc +
            df_norm['n_t'] * w_time
    )

    # Rescale both score columns to strictly [0, 1]
    df[['score']] = final_scaler.fit_transform(df[['score']])
    # df['score_no_slake'] = final_scaler.fit_transform(df[['score_no_slake']])

    df['score'] = df['score'] * 100
    # df['score_no_slake'] = df['score_no_slake'] * 100

    # 6. Save the result
    df.to_csv(file_path, index=False)
    print(f"Success! Score calculated and saved to {file_path}")
    print(df.head())


if __name__ == "__main__":
    # run_batch_pipeline()
    # run_mevf_pipeline()
    # process_batch_closed_logic()
    # run_recovery()
    # calculate_judge_error_summary()
    # process_summary_file()
    # calculate_judge_error_summary_v2()
    calculate_score()

    # ==========================================
    # # SPLIT FILE for LLM as Judge
    # ==========================================
    # import numpy as np
    # file_path = [
    #     "./result/result-slake/results_RAG+Reflexion+Instruct_QLoRA-Qwen3-4B_20260115_042124.csv",
    #     "./result/result-slake/results_ZeroShot+Instruct_QLoRA-LLaVA-Med_20260115_035519.csv",
    #     "./result/result-slake/results_ZeroShot+Basic_DoRA-Qwen3-2B_20260115_083546.csv"
    # ]
    # for file in file_path:
    #     df = pd.read_csv(file)
    #     splits = np.array_split(df, 3)
    #     base_name = os.path.splitext(os.path.basename(file))[0]
    #
    #     for i, split_df in enumerate(splits):
    #         # Naming convention: filename_part_1.csv, filename_part_2.csv, etc.
    #         output_name = f"./result/result-slake/result-slake-{i + 1}/{base_name}_part_{i + 1}.csv"
    #
    #         split_df.to_csv(output_name, index=False)
    #         print(f"Saved: {output_name} ({len(split_df)} rows)")

    # ==========================================
    # MERGE FILE from LLM as Judge(file name already renamed remove _part_{i})
    # ==========================================
    # files = [
    #     './result/result-slake/processed-slake-1/batch_processed_results_20260115_165401.csv',
    #     './result/result-slake/processed-slake-2/batch_processed_results_20260115_165512.csv',
    #     './result/result-slake/processed-slake-3/batch_processed_results_20260115_170251.csv'
    # ]
    #
    # merged_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    #
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_filename = f"./result/processed-slake/batch_processed_results_{timestamp}.csv"
    # merged_df.to_csv(output_filename, index=False)
    # print(f"Successfully merged into {output_filename} ({len(merged_df)} rows)")

    # ==========================================
    # MERGE FILE from LLM as Judge (rename col) -> run_recovery without reparse raw data
    # ==========================================
    # import re
    # part_files = [
    #     './result/result-mevf/processed-mevf-1/batch_processed_results_20260115_124354.csv',
    #     './result/result-mevf/processed-mevf-2/batch_processed_results_20260115_131326.csv',
    #     './result/result-mevf/processed-mevf-3/batch_processed_results_20260115_131452.csv'
    # ]
    # dfs = []
    #
    # print("--- Processing Files ---")
    # for f_path in part_files:
    #     try:
    #         # Load the CSV
    #         df = pd.read_csv(f_path)
    #
    #         # 2. Rename Columns using Regex
    #         # Pattern matches "_part_" followed by 1, 2, or 3
    #         # It replaces it with an empty string ""
    #         new_columns = [re.sub(r'_part_[123]', '', col) for col in df.columns]
    #
    #         # Print example of change (for verification)
    #         changed = [f"{old} -> {new}" for old, new in zip(df.columns, new_columns) if old != new]
    #         if changed:
    #             print(f"File: {f_path}")
    #             print(f"   Renamed {len(changed)} columns. Example: {changed[0]}")
    #
    #         # Apply new column names
    #         df.columns = new_columns
    #         dfs.append(df)
    #
    #     except FileNotFoundError:
    #         print(f"⚠️ File not found: {f_path}")
    #
    # # 3. Merge Vertically
    # if dfs:
    #     # Since columns are now identical (unified), they will stack correctly
    #     merged_df = pd.concat(dfs, ignore_index=True)
    #
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     output_filename = f"./result/processed-mevf/batch_processed_results_{timestamp}.csv"
    #     merged_df.to_csv(output_filename, index=False)
    #     print("\n" + "=" * 40)
    #     print("✅ MERGE SUCCESSFUL")
    #     print("=" * 40)
    #     print(f"Output File:   {output_filename}")
    #     print(f"Total Rows:    {len(merged_df)}")
    #     print(f"Total Columns: {len(merged_df.columns)}")
    #     print("Final Columns Sample:", merged_df.columns.tolist()[:])
    # else:
    #     print("❌ No dataframes to merge.")
