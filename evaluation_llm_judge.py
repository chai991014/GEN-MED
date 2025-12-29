import pandas as pd
import os
import time
import glob
from tqdm import tqdm
from sklearn.metrics import f1_score
import google.generativeai as genai
from groq import Groq
from openai import OpenAI
import evaluate  # pip install evaluate bert-score
import re
import string

# ==========================================
# 1. KEYS & CLIENT INITIALIZATION
# ==========================================
KEYS = {
    "GOOGLE": "AIzaSyBfGm6wPQb-ztNZA_heS8-bPumDfQtHObY",
    "GROQ": "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",
    "DEEPSEEK": "sk-f32f1f1b98bd4603a48c6c98ae451908"
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


# ==========================================
# 2. PROCESSING PIPELINE
# ==========================================
def run_pipeline():
    engine = EnsembleEngine()
    files = glob.glob("./raw_pre_result/results_*.csv")
    summary_data = []

    for f_path in files:
        if "processed_results_" in f_path:
            continue
        df = pd.read_csv(f_path)

        # UI Requirement: Delete original final_prediction
        if "final_prediction" in df.columns:
            df = df.drop(columns=["final_prediction"])

        data = {k: [] for k in ['Gemini_Raw', 'Gemini', 'Llama_Raw', 'Llama', 'Deepseek_Raw', 'Deepseek', 'final_prediction']}

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {os.path.basename(f_path)}"):
            res = engine.judge_all(row['question'], row['ground_truth'], row['raw_prediction'], row['type'])

            for k in ['Gemini_Raw', 'Gemini', 'Llama_Raw', 'Llama', 'Deepseek_Raw', 'Deepseek']:
                data[k].append(res[k])

            # Majority Logic (Ignoring None)
            majority = 1 if (res['Gemini'] + res['Llama'] + res['Deepseek']) >= 2 else 0
            data['final_prediction'].append(majority)
            time.sleep(3)

        for k, v in data.items():
            df[k] = v

        # --- CALCULATE SUMMARY METRICS ---
        closed_df = df[df['type'] == 'CLOSED']
        open_df = df[df['type'] == 'OPEN']

        # F1 Calculation for CLOSED
        gt_binary = closed_df['ground_truth'].str.lower().map({'yes': 1, 'no': 0}).fillna(0)
        f1_closed = f1_score(gt_binary, closed_df['final_prediction']) if not closed_df.empty else 0

        # BioBERTScore for OPEN
        bio_score = 0
        if not open_df.empty:
            b_res = bert_metric.compute(
                predictions=open_df['raw_prediction'].astype(str).tolist(),
                references=open_df['ground_truth'].astype(str).tolist(),
                model_type=BIOBERT_MODEL,
                num_layers=9
            )
            bio_score = sum(b_res['f1']) / len(b_res['f1']) * 100

        summary_data.append({
            "File": os.path.basename(f_path),
            "Closed_Acc_Gemini": closed_df['Gemini'].mean() * 100,
            "Closed_Acc_Llama": closed_df['Llama'].mean() * 100,
            "Closed_Acc_Deepseek": closed_df['Deepseek'].mean() * 100,
            "Closed_Avg_Acc": closed_df[['Gemini', 'Llama', 'Deepseek']].mean(axis=1).mean() * 100,
            "Closed_F1_Majority": f1_closed * 100,
            "Open_Acc_Gemini": open_df['Gemini'].mean() * 100,
            "Open_Acc_Llama": open_df['Llama'].mean() * 100,
            "Open_Acc_Deepseek": open_df['Deepseek'].mean() * 100,
            "Open_Avg_Acc": open_df[['Gemini', 'Llama', 'Deepseek']].mean(axis=1).mean() * 100,
            "Open_BioBERT": bio_score
        })

        # Save Processed Result File
        df.to_csv(f"processed_results_{os.path.basename(f_path)}", index=False)

    # Save Master Summary
    pd.DataFrame(summary_data).to_csv("processed_results_summary.csv", index=False)
    print("\n✅ All files processed. Check processed_results_summary.csv")


def run_mevf_pipeline():
    engine = EnsembleEngine()
    # Updated path to match your request
    files = glob.glob("./raw_pre_result/results_*.csv")
    summary_data = []

    for f_path in files:
        if "processed_results_" in f_path:
            continue
        df = pd.read_csv(f_path)

        if "final_prediction" in df.columns:
            df = df.drop(columns=["final_prediction"])

        # CHANGED: Initialize ALL columns including Raw data to preserve it
        data = {k: [] for k in
                ['Gemini_Raw', 'Gemini', 'Llama_Raw', 'Llama', 'Deepseek_Raw', 'Deepseek', 'final_prediction']}

        print(f"Evaluating {os.path.basename(f_path)}...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            q_type = str(row['type']).upper()

            if q_type == 'CLOSED':
                # STRATEGY A: EXACT MATCH (No API Call)
                match_result = check_exact_match(row['raw_prediction'], row['ground_truth'])

                # Fill Raw and Verdict columns with None for Closed questions
                for k in ['Gemini_Raw', 'Gemini', 'Llama_Raw', 'Llama', 'Deepseek_Raw', 'Deepseek']:
                    data[k].append(None)

                data['final_prediction'].append(match_result)

            else:
                # STRATEGY B: LLM JUDGE (OPEN)
                res = engine.judge_all(row['question'], row['ground_truth'], row['raw_prediction'], row['type'])

                # Store both Raw response and Verdict
                for k in ['Gemini_Raw', 'Gemini', 'Llama_Raw', 'Llama', 'Deepseek_Raw', 'Deepseek']:
                    data[k].append(res[k])

                # Majority Vote
                vote_sum = (res['Gemini'] or 0) + (res['Llama'] or 0) + (res['Deepseek'] or 0)
                data['final_prediction'].append(1 if vote_sum >= 2 else 0)

                # Update DataFrame
        for k, v in data.items():
            df[k] = v

        # --- METRICS CALCULATION ---
        c_df = df[df['type'] == 'CLOSED']
        o_df = df[df['type'] == 'OPEN']

        # A. CLOSED METRICS
        closed_acc = c_df['final_prediction'].mean() * 100 if not c_df.empty else 0

        f1_closed = 0
        if not c_df.empty:
            gt_binary = c_df['ground_truth'].apply(lambda x: 1 if normalize_text(x) in ['yes', 'true'] else 0)
            pred_binary = c_df['raw_prediction'].apply(lambda x: 1 if normalize_text(x) in ['yes', 'true'] else 0)
            f1_closed = f1_score(gt_binary, pred_binary, zero_division=0) * 100

        # B. OPEN METRICS
        open_acc_g = o_df['Gemini'].mean() * 100 if not o_df.empty else 0
        open_acc_l = o_df['Llama'].mean() * 100 if not o_df.empty else 0
        open_acc_d = o_df['Deepseek'].mean() * 100 if not o_df.empty else 0
        open_avg = (open_acc_g + open_acc_l + open_acc_d) / 3 if not o_df.empty else 0

        bio_score = 0
        if not o_df.empty:
            b_res = bert_metric.compute(
                predictions=o_df['raw_prediction'].astype(str).tolist(),
                references=o_df['ground_truth'].astype(str).tolist(),
                model_type=BIOBERT_MODEL,
                num_layers=9
            )
            bio_score = sum(b_res['f1']) / len(b_res['f1']) * 100

        summary_data.append({
            "File": os.path.basename(f_path),
            "Closed_Accuracy": closed_acc,
            "Closed_F1_Score": f1_closed,
            "Open_Acc_Gemini": open_acc_g,
            "Open_Acc_Llama": open_acc_l,
            "Open_Acc_Deepseek": open_acc_d,
            "Open_Avg_Accuracy": open_avg,
            "Open_BioBERT": bio_score
        })

        df.to_csv(f"processed_results_{os.path.basename(f_path)}", index=False)

    pd.DataFrame(summary_data).to_csv("processed_results_summary.csv", index=False)
    print("\n✅ All files processed. Check processed_results_summary.csv")


if __name__ == "__main__":
    run_mevf_pipeline()
