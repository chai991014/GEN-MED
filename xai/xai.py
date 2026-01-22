import torch
import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import gaussian_filter
import google.generativeai as genai
import json
import re
import csv
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
        self.batch_size = 64  # Keep at 1 to prevent VRAM lag

    def generate_heatmap(self, image, question, target_answer, grid_size=6):
        """
        grid_size=6 (36 patches).
        Run time: Approx 2-3 minutes.
        Result: Much more detailed than 4x4.
        """
        # 1. Clean Memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        max_dim = 768
        w, h = image.size

        # Only resize if the image is actually larger than the limit
        if max(w, h) > max_dim:
            # Calculate new size preserving aspect ratio
            ratio = max_dim / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            print(f"📉 Resized image for XAI: {w}x{h} -> {new_w}x{new_h}")
        else:
            image = image.copy()

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
            return np.zeros((h, w), dtype=np.float32)
        target_token_id = target_ids[0]

        # 3. Generate 6x6 Grid
        print(f"Running Balanced Grid ({grid_size}x{grid_size})...")
        import math
        step_x = int(math.ceil(W / grid_size))
        step_y = int(math.ceil(H / grid_size))

        patches = []
        coords = []
        img_np = np.array(image)

        # Generate Baseline
        baseline_score = self._run_inference_single(image, question, target_token_id)

        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate coordinates based on index
                y = i * step_y
                x = j * step_x

                # Handle edge cases (don't go past image bounds)
                y_end = min(y + step_y, H)
                x_end = min(x + step_x, W)

                # Copy & Occlude
                temp_img = img_np.copy()
                cv2.rectangle(temp_img, (x, y), (x_end, y_end), (127, 127, 127), -1)

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
        heatmap = cv2.resize(heatmap, (w, h))

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


class AttentionExplainer:
    """
    Extracts 'Intrinsic Attention' weights from the last transformer layer
    during a single forward pass.
    """

    def __init__(self, model_adapter):
        self.model = model_adapter.model
        self.processor = model_adapter.processor
        self.device = self.model.device

    def generate_heatmap(self, image, question, original_pred=None, grid_size=None):
        # Note: 'original_pred' and 'grid_size' are unused here but kept for API compatibility

        # 1. Prepare Inputs (Manually to capture grid info)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]}
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Process inputs (This handles Qwen's dynamic resizing)
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # 2. Forward Pass with Attention Capture
        # We only need to generate 1 token to see what the model is "looking at" to start its answer.
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                output_attentions=True,
                return_dict_in_generate=True
            )

        # 3. Extract Attention from the Last Layer
        # Structure: outputs.attentions[generated_token_step][layer_index]
        # We take: First generated token -> Last Layer
        # Shape: (Batch, Heads, Query_Len, Key_Len)
        last_layer_att = outputs.attentions[0][-1]

        # Average across all attention heads to get a robust signal
        # Shape: (Batch, Key_Len) -> We squeeze batch & query dims
        att_map = last_layer_att.mean(dim=1)[0, -1, :]

        # 4. Identify Visual Tokens in the Sequence
        # Qwen2-VL delimits images with <|vision_start|> and <|vision_end|>
        input_ids = inputs.input_ids[0]

        # Dynamic Token Lookup (Safer than hardcoding IDs)
        try:
            vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        except:
            # Fallback for some Qwen versions
            vision_start_id = 151653
            vision_end_id = 151654

        try:
            # Find indices
            start_idx = (input_ids == vision_start_id).nonzero(as_tuple=True)[0][0].item()
            end_idx = (input_ids == vision_end_id).nonzero(as_tuple=True)[0][0].item()

            # Extract the raw attention scores for image tokens ONLY
            # We skip the start/end tags themselves
            image_att = att_map[start_idx + 1: end_idx]

            # 5. Reshape based on Qwen's Dynamic Grid
            # The processor stores the exact grid size used (Time, Height, Width)
            # Shape: Tensor([1, h, w])
            grid_thw = inputs['image_grid_thw'][0]
            h, w = grid_thw[1].item(), grid_thw[2].item()

            actual_tokens = image_att.shape[0]

            # --- SCENARIO A: 2x2 Pooling (Standard Qwen3-VL) ---
            # The LLM sees (H/2) * (W/2) tokens.
            # Example: Grid 1408 (44x32) -> Actual 352 (22x16)
            if actual_tokens == (h // 2) * (w // 2):
                print(f"Detected 2x2 Pooling: {h}x{w} -> {h//2}x{w//2}")
                heatmap = image_att.view(h // 2, w // 2).float().cpu().numpy()

            # --- SCENARIO B: Exact Match (No Pooling) ---
            elif actual_tokens == h * w:
                heatmap = image_att.view(h, w).float().cpu().numpy()

            # --- SCENARIO C: Fallback (Aspect Ratio Preservation) ---
            # If Qwen3 or other variants use different stride, we calculate shape mathematically
            else:
                print(f"⚠️ Grid mismatch: Grid={h}x{w} ({h*w}), Actual={actual_tokens}. Attempting aspect-ratio reshape.")
                # Preserve aspect ratio: w/h = ratio
                ratio = w / h
                # h_new * w_new = actual_tokens  =>  h_new^2 * ratio = actual_tokens
                h_new = int(np.sqrt(actual_tokens / ratio))
                w_new = int(actual_tokens / h_new)

                if h_new * w_new == actual_tokens:
                    heatmap = image_att.view(h_new, w_new).float().cpu().numpy()
                else:
                    # Final safety net: Square reshape
                    side = int(np.sqrt(actual_tokens))
                    heatmap = image_att[:side * side].view(side, side).float().cpu().numpy()

            # 6. Robust Normalization (Percentile Clipping) & Resize to Original Image
            # Models often dump massive attention on one "sink" pixel (outlier).
            # We clip the top 2% of intensities so they don't wash out the rest.
            threshold = np.percentile(heatmap, 98)
            heatmap = np.clip(heatmap, 0, threshold)

            # Smoothing
            # Raw attention is spiky. We blur it slightly to reveal the "region".
            heatmap = gaussian_filter(heatmap, sigma=1.0)

            # Standard Min-Max Normalize
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            # Resize heatmap to match original image dimensions
            original_w, original_h = image.size
            heatmap = cv2.resize(heatmap, (original_w, original_h))

            return heatmap

        except Exception as e:
            print(f"❌ Attention Map Error: {e}")
            # Return empty mask on failure
            return np.zeros((image.size[1], image.size[0]))


class RetrievalCounterfactual:
    """
    Generates Visual Counterfactuals using the XAI Super-KB.
    Enforces 'Topic Consistency' to ensure the retrieved question
    is relevant to the user's question.
    """

    def __init__(self, inference_engine, csv_path="medical_concepts_stats_processed.csv"):
        self.engine = inference_engine  # This gives access to the RAG retriever
        self.vocab = set()

        # --- LOAD VOCABULARY FROM CSV ---
        # We load the valid medical terms to know what counts as a "Topic".
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        term = row.get("Concept", "").strip().lower()
                        if term and len(term) > 2:
                            self.vocab.add(term)
            except Exception as e:
                print(f"⚠️ Counterfactual Vocab Error: {e}")
        else:
            # Fallback: Basic set if CSV is missing
            self.vocab = {"pneumonia", "opacity", "effusion", "mass", "nodule", "fracture", "edema", "cardiomegaly",
                          "atelectasis"}

    def _extract_keywords(self, text):
        """
        Extracts only the medically relevant words present in our CSV.
        This automatically filters out noise like "view", "image", "side".
        """
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        # Filter: Only keep words that exist in our Curated CSV
        return {w for w in words if w in self.vocab}

    def find_counterfactual(self, image, original_question, original_prediction_text):
        """
        1. Embed the query image.
        2. Search the RAG database (Knowledge Base).
        3. Filter for an image where the Ground Truth is OPPOSITE to the current prediction.
        """
        # Safety check for RAG & Medical Vocabulary Dictionary
        if not hasattr(self.engine, 'xai_retriever') or self.engine.xai_retriever is None:
            return None, "XAI Retrieval module not active. Enable RAG to use Retrieval Counterfactuals."

        # Use the XAI Retriever (Super KB)
        retriever = self.engine.xai_retriever

        # 1. Retrieve top 10 similar images
        # We need more candidates (k=10) to increase odds of finding an OPPOSITE case
        candidates = retriever.retrieve(original_question, image, k=50, alpha=0.6)

        # 2. Heuristic: Determine the 'target' opposite label
        # If prediction is "Yes", we want a "No" (and vice versa)
        clean_pred = original_prediction_text.lower().strip()
        is_binary_pred = clean_pred in ["yes", "no"]

        # Extract Core Concept from User Question (e.g. "Tumor")
        user_topic_keywords = self._extract_keywords(original_question)

        cf_item = None

        # 3. Search Loop
        for cand in candidates:
            cand_q = str(cand['question']).lower()
            cand_ans = str(cand['answer']).lower().strip()

            # 2. Extract Core Concept from Neighbor
            cand_topic_keywords = self._extract_keywords(cand_q)

            # 3. STRICT TOPIC MATCHING
            # If user has medical concepts, the neighbor MUST share at least one.
            # This prevents matching "Tumor" with "Effusion" just because both are 'No'.
            if user_topic_keywords and not user_topic_keywords.intersection(cand_topic_keywords):
                continue

            # 4. Check for Contrast (Opposite Answer)
            if is_binary_pred:
                target = "no" if clean_pred == "yes" else "yes"
                if cand_ans == target:
                    cf_item = cand
                    break
            else:
                if cand_ans != clean_pred and cand_ans not in clean_pred and clean_pred not in cand_ans:
                    cf_item = cand
                    break

        # 4. Handle "No Counterfactual Found"
        if not cf_item:
            msg = f"No valid counterfactual found (Checked 50 neighbors, none matched topic '{list(user_topic_keywords)}' with opposite answer)."
            return None, msg

        # 5. Fetch Image & Format Report
        ds = retriever.knowledge_base
        try:
            cf_image = ds[int(cf_item['idx'])]['image'].convert("RGB")
        except:
            return None, "Error loading counterfactual image."

        report = f"""
        ### 📊 Case Comparison
        **Model Prediction:** {original_prediction_text}\n
        **Counterfactual Truth:** {cf_item['answer']}\n
        **Matched Details:**\n
        - **Common Topic:** {list(user_topic_keywords.intersection(cand_topic_keywords))}
        - **Historical Question:** *{cf_item['question']}*
        - **Historical Outcome:** {cf_item['answer']} (Confirmed by Radiologist)
        """

        return cf_image, report


class ConceptExplainer:
    """
    Handles Concept-Based Explanations using two strategies:
    1. Zero-Shot Activation (Fixed List): Checks image against fixed medical terms.
    2. Neighbor-Based Consensus (Open-Ended Discovery): Dynamically finds concepts from RAG neighbors.
    """

    # Standard CheXpert-style list
    CONCEPTS = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices"
    ]

    def __init__(self, inference_engine, csv_path="./xai/medical_concepts_stats_processed.csv"):
        # We need the XAI retriever (BioMedCLIP) for encoding and searching
        if hasattr(inference_engine, 'xai_retriever') and inference_engine.xai_retriever is not None:
            self.model = inference_engine.retriever.model
            self.processor = inference_engine.retriever.preprocess
            self.tokenizer = inference_engine.retriever.tokenizer
            self.device = inference_engine.retriever.device
            self.retriever = inference_engine.retriever
            self.csv_list = csv_path
        else:
            self.model = None

    def evaluate(self, image, concepts=None):
        """
        Method 1: Zero-Shot Concept Activation (Direct CSV Load).
        Uses BioMedCLIP to measure the similarity between the image and high-level
        medical concepts. This satisfies the 'Concept-based Explanation' rubric.
        """
        if not self.model:
            return None, "RAG/BioMedCLIP is not loaded. Cannot run Concept Analysis."

        if concepts is None:
            # Default medical concepts to test against
            concepts = []
            # --- DIRECT LOAD FROM CURATED CSV ---
            if os.path.exists(self.csv_list):
                try:
                    with open(self.csv_list, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        raw_candidates = []

                        for row in reader:
                            term = row.get("Concept", "").strip()
                            if term:  # Only check if not empty
                                raw_candidates.append(term.title())

                except Exception as e:
                    print(f"⚠️ CSV Error: {e}")

            existing_set = set(c.lower() for c in raw_candidates)

            for addon in self.CONCEPTS:
                if addon.lower() not in existing_set:
                    raw_candidates.append(addon)

            concepts = raw_candidates

            # If CSV is missing or empty, use this standard CheXpert-style list
            if not concepts:
                concepts = self.CONCEPTS

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
        all_results = sorted(zip(concepts, scores), key=lambda x: x[1], reverse=True)
        results = all_results[:10]

        # Create a Bar Chart text representation
        report = ""
        if os.path.exists(self.csv_list):
            report += f"✅ Scanned **{len(concepts)} concepts** (Combined CSV + CheXpert List).\n\n"

        has_output = False
        for concept, score in results:
            # Show anything with > 1% activation
            if score < 0.01:
                continue
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            report += f"- **{concept}**:\n"
            report += f"`{bar}` ({score * 100:.1f}%)\n"
            has_output = True

        if not has_output:
            report += "No concept activated. (0% Activation on all concepts)"

        return results, report

    def discover_concepts(self, image, question, k=30):
        """
        Method 2: Neighbor-Based Concept Consensus (Dynamic CSV-Driven).
        Retrieves k similar cases and extracts curated CSV file (concepts)
        from their Ground Truth answers.
        """
        if not self.model:
            return "XAI Retrieval module not active."

        # 1. Load Vocabulary from CSV
        # We build a set of valid terms to look for (e.g., {'Pneumonia', 'Mass', 'Opacity'})
        valid_vocab = set()
        if os.path.exists(self.csv_list):
            try:
                with open(self.csv_list, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        term = row.get("Concept", "").strip().lower()
                        if term and len(term) > 2:
                            valid_vocab.add(term)
            except Exception as e:
                return f"Error reading CSV: {e}"

        for addon in self.CONCEPTS:
            valid_vocab.add(addon.lower())

        # 2. Retrieve Neighbors
        results = self.retriever.retrieve(question, image, k=k)

        # 3. Dynamic Scanning
        concept_counts = Counter()

        for i, item in enumerate(results):
            # Weight Decay: Rank 1 matters more than Rank 50
            weight = 1.0 / (1.0 + (i * 0.1))

            # Context Building: Combine Answer + Question if Answer is vague
            a_text = str(item['answer']).lower()
            text_to_scan = a_text
            if len(a_text) < 5 or a_text in ['yes', 'no']:
                text_to_scan += " " + str(item['question']).lower()

            # Check for matches from our Combined Vocabulary
            # We check if the full phrase exists in the text (e.g. "pleural effusion")
            for concept in valid_vocab:
                if concept in text_to_scan:
                    concept_counts[concept.title()] += weight

        # 4. Normalize & Format
        total_weight = sum(concept_counts.values()) + 1e-9
        final_results = []
        for term, score in concept_counts.items():
            percent = (score / total_weight) * 100
            final_results.append((term, percent))

        final_results.sort(key=lambda x: x[1], reverse=True)

        # 5. Generate Report
        report = ""
        if os.path.exists(self.csv_list):
            report += f"✅ Aggregated data from **{k} similar neighbours** (Checked against CSV + CheXpert list).\n"
        else:
            report += f"✅ Aggregated data from **{k} similar neighbours** (Checked against CheXpert list only).\n"

        # Filter: Only show concepts with > 5% relevance
        active_concepts = [c for c in final_results if c[1] > 2.0]

        if not active_concepts:
            report += "*No specific medical pathology consensus found (mostly generic conversation).*"
        else:
            for term, conf in active_concepts[:5]:  # Top 5
                visual_conf = min(100, conf * 2.5)
                bar_len = int((visual_conf / 100) * 20)  # Scale bar
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
                self.generation_config = {
                    "response_mime_type": "application/json"
                }
                self.model = genai.GenerativeModel(
                    "gemini-2.5-flash",
                    generation_config=self.generation_config
                )
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

        --- OUTPUT REQUIREMENT ---
        Output a JSON object with this schema:
        {
            "items": [
                {
                    "index": int,
                    "visual_check": "string",
                    "semantic_check": "string",
                    "verdict": "RELEVANT" or "IRRELEVANT",
                    "reasoning": "string"
                }
            ],
            "summary": "string"
        }
        
        --- OUTPUT TEMPLATE EXAMPLE (JSON ONLY) ---
        
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
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(contents)
                raw_text = response.text

                # Clean Markdown (Gemini often adds ```json)
                clean_text = re.sub(r"```json|```", "", raw_text).strip()

                data = json.loads(clean_text)
                summary = data.get("summary", "No summary provided.")

                return data, summary

            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} Multimodal Judge Error: {e}")
                if attempt < max_retries - 1:
                    continue

        print(f"❌ All {max_retries} attempts failed.")
        fallback_data = {"items": [{"verdict": "ERROR", "visual_check": "N/A", "semantic_check": "N/A", "reasoning": "API Error"} for _ in range(len(rag_items))]}
        return fallback_data, f"❌ All {max_retries} attempts failed."
