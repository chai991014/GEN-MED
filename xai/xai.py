import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import gaussian_filter
import google.generativeai as genai
import json
import re


class FastOcclusionExplainer:
    """
    Balanced Occlusion Explainer.
    Defaults to 6x6 Grid (36 passes) for better detail.
    Uses Dynamic Smoothing to turn blocks into organic blobs.
    """

    def __init__(self, adapter):
        self.adapter = adapter
        self.model = adapter.model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 1  # Keep at 1 to prevent VRAM lag

    def generate_heatmap(self, image, question, target_answer, grid_size=6):
        """
        grid_size=6 (36 patches).
        Run time: Approx 2-3 minutes.
        Result: Much more detailed than 4x4.
        """
        # 1. Clean Memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        image = image.convert("RGB")
        W, H = image.size

        # 2. Tokenize Target
        if "LLaVA" in str(type(self.adapter).__name__):
            target_ids = self.adapter.tokenizer(target_answer, add_special_tokens=False).input_ids
        else:
            if hasattr(self.adapter, 'processor'):
                target_ids = self.adapter.processor.tokenizer(target_answer, add_special_tokens=False).input_ids
            else:
                target_ids = self.adapter.tokenizer(target_answer, add_special_tokens=False).input_ids

        if len(target_ids) == 0:
            return np.zeros((H, W), dtype=np.float32)
        target_token_id = target_ids[0]

        # 3. Generate 6x6 Grid
        print(f"⚡ Running Balanced Grid ({grid_size}x{grid_size})...")
        step_x = int(W / grid_size)
        step_y = int(H / grid_size)

        patches = []
        coords = []
        img_np = np.array(image)

        # Generate Baseline
        baseline_score = self._run_inference_single(image, question, target_token_id)

        for y in range(0, H, step_y):
            for x in range(0, W, step_x):
                temp_img = img_np.copy()
                # Draw gray box
                cv2.rectangle(temp_img, (x, y), (x + step_x, y + step_y), (127, 127, 127), -1)
                patches.append(Image.fromarray(temp_img))
                coords.append((x, y))

        # 4. Inference Loop
        scores = []
        for patch in tqdm(patches, desc="Scanning Grid"):
            s = self._run_inference_single(patch, question, target_token_id)
            scores.append(s)

        # 5. Build Heatmap
        heatmap = np.zeros((H, W), dtype=np.float32)
        for i, score in enumerate(scores):
            x, y = coords[i]
            diff = max(0, baseline_score - score)

            # Fill the exact grid block
            # Ensure we don't go out of bounds
            y_end = min(y + step_y, H)
            x_end = min(x + step_x, W)
            heatmap[y:y_end, x:x_end] = diff

        # 6. DYNAMIC SMOOTHING (The Magic Fix)
        # Instead of fixed sigma=8, we calculate sigma based on block size.
        # sigma = step_size / 3 ensures blocks melt into round blobs.
        dynamic_sigma = step_x / 3.0

        heatmap = gaussian_filter(heatmap, sigma=dynamic_sigma)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap

    def _run_inference_single(self, image, question, target_token_id):
        self.model.eval()
        with torch.no_grad():
            if "LLaVA" in str(type(self.adapter).__name__):
                qs = self.adapter.tok_img + "\n" + question
                prompt = f"USER: {qs}\nASSISTANT:"
                input_ids = self.adapter.tokenizer_image_token(prompt, self.adapter.tokenizer, self.adapter.idx,
                                                               return_tensors='pt').unsqueeze(0).to(self.device)

                image_tensor = self.adapter.image_processor.preprocess(image, return_tensors='pt')[
                    'pixel_values'].half().to(self.device)

                outputs = self.model(input_ids, images=image_tensor)
                logits = outputs.logits[0, -1, :]
            else:
                # Qwen fallback
                from qwen_vl_utils import process_vision_info
                messages = [{"role": "user",
                             "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
                text_input = self.adapter.processor.apply_chat_template(messages, tokenize=False,
                                                                        add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.adapter.processor(text=[text_input], images=image_inputs, videos=video_inputs,
                                                padding=True, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]

            prob = torch.softmax(logits, dim=-1)[target_token_id].item()
            return prob


class CounterfactualExplainer:
    """
    Fast Text-Based Robustness Checks.
    """

    def __init__(self, adapter):
        self.adapter = adapter

    def evaluate(self, image, original_question, original_prediction):
        report = []
        score = 0

        # 1. Hallucination Check
        pathologies = ["fracture", "pneumonia", "tumor", "mass"]
        targets = [p for p in pathologies if p.lower() not in original_prediction.lower()]

        import random
        if targets:
            neg = random.choice(targets)
            q_neg = f"Is there a {neg}?"
            # Use direct generation
            ans = self.adapter.generate(image, q_neg, context="").get("prediction", "")

            report.append(f"- **Q:** {q_neg} \n - **A:** {ans}")
            if "no" in ans.lower() or "normal" in ans.lower():
                score += 1
            else:
                report.append("_(⚠️ Warning: Possible Hallucination)_")

        # 2. Consistency Check
        rephrased = original_question.replace("What is", "Identify").replace("is there", "do you see")
        ans_rep = self.adapter.generate(image, rephrased, context="").get("prediction", "")

        is_consistent = (original_prediction.lower() in ans_rep.lower()) or (
                    ans_rep.lower() in original_prediction.lower())
        report.append(f"\n- **Consistency:** {'✅ Pass' if is_consistent else '⚠️ Fail'}")
        if is_consistent:
            score += 1

        return "\n".join(report)


def apply_colormap(image, mask):
    if mask is None: return image
    mask[mask < 0.2] = 0
    try:
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        img_np = np.array(image)
        overlayed = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        return Image.fromarray(overlayed)
    except:
        return image


class ReflexionJudge:
    """
    Uses an external LLM (Gemini 2.5 Flash Lite) to act as a 'Supervisor'.
    It evaluates the quality of the Reflexion process:
    1. Was the Draft accurate?
    2. Did the Critique actually catch mistakes?
    3. Is the Final Answer better than the Draft?
    """

    def __init__(self, api_key):
        self.api_key = api_key
        if not self.api_key:
            print("⚠️ ReflexionJudge: No Google API Key provided.")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-2.5-flash-lite")
            except Exception as e:
                print(f"⚠️ ReflexionJudge Init Error: {e}")
                self.model = None

    def evaluate(self, question, gt, draft, critique, final_pred):
        if not self.model:
            return "⚠️ Judge not initialized (Check API Key)."

        prompt = f"""
        Act as a Senior Medical AI Supervisor. Evaluate the reasoning process of a junior AI model.

        --- CONTEXT ---
        **Question:** {question}
        **Ground Truth:** {gt}

        --- AI REASONING TRACE ---
        **1. Initial Draft:** {draft}
        **2. Self-Critique:** {critique}
        **3. Final Answer:** {final_pred}

        --- TASK ---
        Analyze the effectiveness of the 'Reflexion' (Self-Correction) process:
        1. **Critique Quality:** Was the critique accurate? Did it identify real hallucinations or errors in the draft?
        2. **Improvement:** Did the Final Answer improve upon the Draft? (Better/Worse/Same)
        3. **Verification:** Is the Final Answer consistent with the Ground Truth?

        Output a concise Markdown report with:
        - **Reflexion Score:** (0/10 - How well did it self-correct?)
        - **Analysis:** (Brief bullet points)
        - **Verdict:** (Safe/Unsafe)
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ Judge Execution Error: {e}"


class RAGJudge:
    """
    Multimodal RAG Evaluator (Transfer Learning from GeoXplain).
    Uses Gemini Vision to act as a 'Senior Radiologist' Judge.
    It compares the USER'S QUERY against the RETRIEVED IMAGE + TEXT.
    """

    def __init__(self, api_key):
        self.api_key = api_key
        if not self.api_key:
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-3-flash-preview")
            except Exception as e:
                print(f"⚠️ RAG Judge Init Error: {e}")
                self.model = None

    def evaluate_batch(self, user_query, rag_items):
        """
        Args:
            user_query (str): The original User Query.
            rag_items (list): List of dicts [{'image': PIL, 'q': str, 'a': str}, ...]
        """
        if not self.model:
            return "⚠️ Judge not initialized."

        if not rag_items:
            return "⚠️ No items to evaluate."

        # 1. Start the Interleaved Payload
        contents = []

        # System Intro
        intro_prompt = f"""
        ### ROLE: Senior Medical Data Auditor

        ### TASK:
        A RAG system retrieved {len(rag_items)} clinical cases (X-Rays + Reports) to answer a User Query.
        Your job is to evaluate the relevance of EACH retrieved item.

        ### USER QUERY:
        "{user_query}"

        --- RETRIEVED ITEMS BELOW ---
        """
        contents.append(intro_prompt)

        # 2. Iterate through the LIST and append to payload
        for i, item in enumerate(rag_items):
            if item is None:
                continue

            # Header for this item
            item_text = f"""

            ### ITEM #{i + 1}
            **Retrieved Question:** "{item['q']}"
            **Retrieved Answer:** "{item['a']}"
            **Retrieved Scan:** (See image below)
            """

            contents.append(item_text)  # Add Text
            contents.append(item['img'])  # Add PIL Image (Interleaved)

        # 3. Closing Instruction
        closing_prompt = """
        --- EVALUATION CRITERIA (HOW TO JUDGE) ---
        For EACH item, perform a strict medical assessment:
        
        1. **Visual Relevance Check:** - Does the X-Ray show the **same anatomical region** as the User Query? (e.g., if query is "Chest", reject "Knee" images).
           - Does it show a **relevant projection** or view?
           
        2. **Semantic Relevance Check:** - Does the Retrieved Question/Answer pair discuss the **same pathology** or medical concept?
           - Is the information actually useful for answering the User Query?

        3. **Verdict Logic:**
           - **RELEVANT:** If the item offers useful visual OR textual evidence.
           - **IRRELEVANT:** If it is a Hallucination, Wrong Body Part, or Unrelated Condition.

        --- OUTPUT TEMPLATE (JSON ONLY) ---
        You must return a SINGLE valid JSON object. 
        Do NOT use Markdown formatting (no ```json blocks).
        
        {
            "items": [
                {
                    "index": 1,
                    "visual_check": "Image confirms Chest X-Ray...",
                    "semantic_check": "Text discusses pneumonia...",
                    "verdict": "RELEVANT" (or "IRRELEVANT"),
                    "reasoning": "Strong anatomical match."
                },
                ... (Repeat for all items)
            ],
            "summary": "Brief overall assessment of retrieval quality."
        }
        """
        contents.append(closing_prompt)

        # 4. Single API Call
        try:
            response = self.model.generate_content(contents)
            raw_text = response.text

            # Clean Markdown (Gemini often adds ```json)
            clean_text = re.sub(r"```json|```", "", raw_text).strip()

            data = json.loads(clean_text)
            summary = data.get("summary", "No summary provided.")

            return data, summary

        except Exception as e:
            print(f"JSON Parse Error: {e}")
            return {}, f"❌ Error parsing Judge Output: {str(e)}"
