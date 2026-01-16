import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import gaussian_filter
import google.generativeai as genai
import json
import re
from collections import Counter


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


class RetrievalCounterfactual:
    """
    Generates Visual Counterfactuals using the RAG database.
    It finds a 'Negative' example (Healthy) that is visually similar to the 'Positive' input.
    """

    def __init__(self, inference_engine):
        self.engine = inference_engine  # This gives access to the RAG retriever

    def find_counterfactual(self, image, original_question, original_prediction_text):
        """
        1. Embed the query image.
        2. Search the RAG database (Knowledge Base).
        3. Filter for an image where the Ground Truth is OPPOSITE to the current prediction.
        """
        # Safety check for RAG
        if not hasattr(self.engine, 'xai_retriever') or self.engine.xai_retriever is None:
            return None, "XAI Retrieval module not active. Enable RAG to use Retrieval Counterfactuals."

        # Use the XAI Retriever (Super KB)
        retriever = self.engine.xai_retriever

        # 1. Retrieve top 10 similar images
        # We need more candidates (k=10) to increase odds of finding an OPPOSITE case
        candidates = retriever.retrieve(original_question, image, k=20, alpha=0.5)

        # 2. Heuristic: Determine the 'target' opposite label
        # If prediction is "Yes", we want a "No" (and vice versa)
        clean_pred = original_prediction_text.lower().strip()
        is_binary_pred = clean_pred in ["yes", "no"]

        cf_item = None

        # 3. Search Loop
        for cand in candidates:
            cand_ans = str(cand['answer']).lower().strip()

            # --- SCENARIO A: Binary Prediction (Yes/No) ---
            if is_binary_pred:
                # We strictly want the OPPOSITE answer
                target_gt = "no" if clean_pred == "yes" else "yes"
                if cand_ans == target_gt:
                    cf_item = cand
                    break

            # --- SCENARIO B: Open-Ended Prediction (e.g., "Pneumonia") ---
            else:
                # We want a case that is visually similar but has a DIFFERENT answer.

                # Rule 1: It cannot be the exact same answer (That's a supporting example, not a CF)
                if cand_ans == clean_pred:
                    continue

                # Rule 2: It should not be a "subset" string (e.g. "left lung" vs "lung")
                # This avoids weak counterfactuals where the difference is just specificity.
                if cand_ans in clean_pred or clean_pred in cand_ans:
                    continue

                # Rule 3: Ignore generic/unhelpful answers if the prediction was specific
                if cand_ans in ["chest x-ray", "pa", "ap", "film"]:
                    continue

                # If passed checks, we found a "Contrast Case"
                cf_item = cand
                break

        # 4. Handle "No Counterfactual Found"
        if not cf_item:
            msg = "Could not find a valid counterfactual (all retrieved neighbors had similar answers to your prediction)."
            return None, msg

        # 5. Fetch Image & Format Report
        ds = retriever.knowledge_base
        try:
            cf_image = ds[int(cf_item['idx'])]['image'].convert("RGB")
        except:
            return None, "Error loading counterfactual image."

        return cf_image, f"""
        ### 🔄 Visual Counterfactual Found
        **Analysis:**
        The model predicted **'{original_prediction_text}'**.
        To verify this, we retrieved a visually similar historical case where the Ground Truth was **'{cf_item['answer']}'**.

        **Similar Historical Case:**
        - **Question:** {cf_item['question']}
        - **Answer:** {cf_item['answer']}

        *If the model is robust, it should visually distinguish the pathology in the Input vs. the Healthy Counterfactual.*
        """


class ConceptExplainer:
    """
    Handles Concept-Based Explanations using two strategies:
    1. Zero-Shot Activation (Fixed List)
    2. Neighbor-Based Consensus (Open-Ended Discovery)
    """

    def __init__(self, inference_engine):
        # We need the XAI retriever (BioMedCLIP) for encoding and searching
        if hasattr(inference_engine, 'xai_retriever') and inference_engine.xai_retriever is not None:
            self.model = inference_engine.retriever.model
            self.processor = inference_engine.retriever.preprocess
            self.tokenizer = inference_engine.retriever.tokenizer
            self.device = inference_engine.retriever.device
            self.retriever = inference_engine.retriever
        else:
            self.model = None

    def evaluate(self, image, concepts=None):
        """
        Method 1: Zero-Shot Concept Activation.
        Uses BioMedCLIP to measure the similarity between the image and high-level
        medical concepts. This satisfies the 'Concept-based Explanation' rubric.
        """
        if not self.model:
            return None, "RAG/BioMedCLIP is not loaded. Cannot run Concept Analysis."

        if concepts is None:
            # Default medical concepts to test against
            concepts = [
                "Normal Chest X-Ray",
                "Pleural Effusion",
                "Pneumonia",
                "Consolidation",
                "Edema",
                "Atelectasis",
                "Cardiomegaly",
                "Pneumothorax",
                "Fracture",
                "Support Devices"
            ]

        # 1. Process Image
        image_input = self.processor(image).unsqueeze(0).to(self.device)

        # 2. Process Text Concepts
        text_inputs = self.tokenizer(concepts).to(self.device)

        # 3. Calculate Similarity (Concept Activation)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Dot Product = Similarity Score
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            scores = similarity[0].cpu().numpy()

        # 4. Format Output
        results = sorted(zip(concepts, scores), key=lambda x: x[1], reverse=True)

        # Create a Bar Chart text representation
        report = "### 🧠 Concept Activation Analysis (BioMedCLIP)\n"
        report += "This analysis measures which high-level medical concepts are most activated by this image.\n\n"

        for concept, score in results:
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            report += f"- **{concept}**:\n"
            report += f"`{bar}` ({score * 100:.1f}%)\n"

        return results, report

    def discover_concepts(self, image, question, k=30):
        """
        Method 2: Neighbor-Based Concept Consensus.
        Retrieves k similar cases and extracts the most common keywords (concepts)
        from their Ground Truth answers.
        """
        if not self.model:
            return "XAI Retrieval module not active."

        # 1. Retrieve Neighbors
        results = self.retriever.retrieve(question, image, k=k)

        # 2. Define Concept Mapping (The "Medical Dictionary")
        # This groups scattered words into meaningful categories
        concept_map = {
            "Lung Opacity/Pneumonia": ["opacity", "opacities", "consolidation", "infiltrate", "pneumonia", "airspace",
                                       "density"],
            "Atelectasis": ["atelectasis", "collapse", "volume loss"],
            "Pleural Effusion": ["effusion", "fluid", "blunting", "costophrenic"],
            "Pneumothorax": ["pneumothorax", "air", "pleural space"],
            "Cardiomegaly/Heart": ["cardiomegaly", "enlarged", "heart", "cardiac", "silhouette"],
            "Edema/Congestion": ["edema", "congestion", "vascular", "hilar", "prominence"],
            "Fracture/Bone": ["fracture", "broken", "bone", "rib", "clavicle", "skeletal"],
            "Support Device": ["tube", "line", "catheter", "pacemaker", "clip", "wire"],
            "Normal/Healthy": ["normal", "unremarkable", "clear", "negative", "no acute"],
        }

        # Initialize Scores (Float, because we use weighted ranking)
        concept_scores = {key: 0.0 for key in concept_map}

        # 3. Process Neighbors with Weighted Voting
        for i, item in enumerate(results):
            text = str(item['answer']).lower()

            # Weight Decay: Rank 1 is worth 1.0, Rank 50 is worth ~0.2
            # This ensures the BEST visual matches matter more than the noisy tail
            weight = 1.0 / (1.0 + (i * 0.1))

            q_text = str(item['question']).lower()
            a_text = str(item['answer']).lower()

            # Text to analyze: Always check Answer, check Question if Answer is short/vague
            text_to_check = a_text
            if len(a_text) < 5 or a_text in ['yes', 'no']:
                text_to_check += " " + q_text

            # Check against dictionary
            matched_any = False
            for category, keywords in concept_map.items():
                if any(kw in text_to_check for kw in keywords):
                    concept_scores[category] += weight
                    matched_any = True

            # Fallback: If no specific pathology found, check for generic "Yes/No"
            # (Optional, but helps context)

        # 4. Sort & Normalize
        total_weight = sum(concept_scores.values()) + 1e-9  # Avoid div/0

        # Convert to percentages
        final_results = []
        for cat, score in concept_scores.items():
            if score > 0:
                percent = (score / total_weight) * 100
                final_results.append((cat, percent))

        # Sort by score
        final_results.sort(key=lambda x: x[1], reverse=True)

        # 5. Generate Report
        report = "### 🧬 Neighbor-Based Concept Consensus\n"
        report += f"Analyzed **{k} neighbors** using Semantic Grouping (weighted by similarity).\n"

        # Filter: Only show concepts with > 5% relevance
        active_concepts = [c for c in final_results if c[1] > 2.0]

        if not active_concepts:
            report += "*No specific medical pathology consensus found (mostly generic conversation).*"
        else:
            for term, conf in active_concepts[:5]:  # Top 5
                bar_len = int(min(20, (conf / 100) * 40))  # Scale bar
                bar = "█" * bar_len + "░" * (20 - bar_len)
                report += f"- **{term}**:\n"
                report += f"`{bar}` ({conf:.1f}%)\n"

        return report


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
