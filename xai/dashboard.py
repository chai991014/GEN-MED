import os
import torch
import gradio as gr
import base64
import gc
from datasets import load_dataset, load_from_disk, concatenate_datasets
from llm_adapter import get_llm_adapter
from rag_pipeline import RAGPipeline
from retriever import MultimodalRetriever
from xai import FastOcclusionExplainer, ConceptExplainer, RetrievalCounterfactual, ReflexionJudge, RAGJudge, apply_colormap


# --- Default Config ---
CONFIG = {
    "TECH_TAG": None,
    "OUTPUT_FILE": None,
    "DATASET_ID": None,
    "LLAVA_REPO_PATH": "../LLaVA-Med",
    "MODEL_TYPE": None,
    "ADAPTER_PATH": None,
    "PROMPT": "Instruct",
    "USE_REFLEXION": True,
    "USE_RAG": True,
    "RAG_K": 3,
    "RAG_ALPHA": 0.5,
}


# --- Model & Adapter / Dataset Mapping ---
MODEL_OPTIONS = {
    "LLaVA-Med 1.5 Mistral 7B (Base)": {"model": "microsoft/llava-med-v1.5-mistral-7b", "adapter": None},
    "LLaVA-Med 1.5 Mistral 7B (QLoRA Fine-tuned)": {"model": "../finetune/qlora-llava", "adapter": None},
    "LLaVA-Med 1.5 Mistral 7B (Dora Fine-tuned)": {"model": "../finetune/dora-llava", "adapter": None},
    "Qwen3-VL-2B-Instruct (Base)": {"model": "Qwen/Qwen3-VL-2B-Instruct", "adapter": None},
    "Qwen3-VL-2B-Instruct (QLoRA Fine-tuned)": {"model": "Qwen/Qwen3-VL-2B-Instruct",
                                                "adapter": "../finetune/qlora-qwen3-2b/checkpoint-600"},
    "Qwen3-VL-2B-Instruct (Dora Fine-tuned)": {"model": "Qwen/Qwen3-VL-2B-Instruct",
                                               "adapter": "../finetune/dora-qwen3-2b/checkpoint-580"},
    "Qwen3-VL-4B-Instruct (Base)": {"model": "Qwen/Qwen3-VL-4B-Instruct", "adapter": None},
    "Qwen3-VL-4B-Instruct (QLoRA Fine-tuned)": {"model": "Qwen/Qwen3-VL-4B-Instruct",
                                                "adapter": "../finetune/qlora-qwen3-4b/checkpoint-390"},
    "Qwen3-VL-4B-Instruct (Dora Fine-tuned)": {"model": "Qwen/Qwen3-VL-4B-Instruct",
                                               "adapter": "../finetune/dora-qwen3-4b/checkpoint-390"},
}

DATASET_OPTIONS = {
    "VQA-RAD": {
        "path": "flaviagiammarino/vqa-rad",
        "size": 451
    },
    "SLAKE": {
        "path": "../slake_vqa_rad_format",
        "size": 1407
    },
}


# --- Global State ---
inference_engine = None
LOADED_CONFIG = None
MAX_RAG_K = 10
GOOGLE_API_KEY = "AIzaSyBfGm6wPQb-ztNZA_heS8-bPumDfQtHObY"


def get_img_html(img_path):
    try:
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f'<img src="data:image/png;base64,{encoded_string}" width="40" style="display: inline-block; margin-right: 10px; vertical-align: bottom;" />'
    except Exception as e:
        print(f"Could not load icon: {e}")
        return ""


def load_data_source(path, split):
    if os.path.exists(path) and os.path.isdir(path):
        full_ds = load_from_disk(path)
        return full_ds[split]
    else:
        return load_dataset(path, split=split)


def get_status_msg(config_dict=None):
    if config_dict is None:
        status_log = """
            ## ⚪ System Status: Idle\n
            Please load a model.
        """
        return status_log

    prompt_txt = config_dict['prompt']
    rag_txt = f"✅ Enabled (K={config_dict['k']}, α={config_dict['alpha']})" if config_dict["rag"] else "❌ Disabled"
    ref_txt = "✅ Enabled" if config_dict["reflexion"] else "❌ Disabled"

    status_log = f"""
        ## 🟢 System Status: Ready\n
        **Model:** `{config_dict['model']}`\n
        **Prompt style:** {prompt_txt}\n
        **Reflexion:** {ref_txt}\n
        **RAG:** {rag_txt}
    """

    return status_log


def create_md_accordion(label_md, initial_visible=False, visibility=True):
    # 1. Local State for this specific accordion
    is_open = gr.State(initial_visible)

    # 2. The Header Row
    with gr.Row(visible=visibility, elem_classes="accordion-header") as header_row:
        with gr.Column(scale=9, min_width=0):
            gr.Markdown(label_md)
        # Unique toggle button
        with gr.Column(scale=1, min_width=0):
            btn_label = "Hide 🔼" if initial_visible else "Show 🔽"
            btn = gr.Button(btn_label, size="sm", scale=0, min_width=1)

    # 3. The Content Group
    content_group = gr.Column(visible=initial_visible)

    # 4. Internal Toggle Logic (Self-contained)
    def toggle(open_state):
        new_state = not open_state
        btn_lbl = "Hide 🔼" if new_state else "Show 🔽"
        return new_state, gr.update(visible=new_state), btn_lbl

    btn.click(
        fn=toggle,
        inputs=is_open,
        outputs=[is_open, content_group, btn]
    )

    return header_row, content_group


def load_engine(model_key, use_rag, use_reflexion, prompt_style, k_val, alpha_val):
    global inference_engine, LOADED_CONFIG

    # 1. Update local config based on user input
    selected_paths = MODEL_OPTIONS[model_key]
    config = CONFIG.copy()
    config["MODEL_TYPE"] = selected_paths["model"]
    config["ADAPTER_PATH"] = selected_paths["adapter"]
    config["PROMPT"] = prompt_style
    config["USE_RAG"] = use_rag
    config["USE_REFLEXION"] = use_reflexion
    config["RAG_K"] = k_val
    config["RAG_ALPHA"] = alpha_val

    # Clean memory for new model load
    if inference_engine is not None:
        del inference_engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2. Base LLM Initialization
    llm_params = {
        "repo_path": os.path.abspath(config["LLAVA_REPO_PATH"]),
        "model_path": config["MODEL_TYPE"],
        "prompt": config["PROMPT"],
        "adapter_path": config["ADAPTER_PATH"]
    }

    llm = get_llm_adapter(config["MODEL_TYPE"], **llm_params)
    llm.load()

    # 3. RAG Wrapper (Applied if RAG is toggled, regardless of Reflexion)
    if config["USE_RAG"]:
        retriever_engine = MultimodalRetriever(device="cpu")
        # Load Knowledge Base
        ds_rad = load_dataset("flaviagiammarino/vqa-rad", split="train")
        ds_rad = ds_rad.add_column("idx", range(len(ds_rad)))

        inference_engine = RAGPipeline(
            llm,
            retriever_engine,
            k=config["RAG_K"],
            alpha=config["RAG_ALPHA"],
        )
        inference_engine.build_index(ds_rad)

        # XAI RETRIEVER (Robust, VQA-RAD + SLAKE)
        retriever_xai = MultimodalRetriever(device="cpu")
        try:
            ds_slake = load_data_source(DATASET_OPTIONS["SLAKE"]["path"], "train")

            common_cols = ["image", "question", "answer"]
            ds_rad_clean = ds_rad.select_columns(common_cols)
            ds_slake_clean = ds_slake.select_columns(common_cols)

            ds_super_kb = concatenate_datasets([ds_rad_clean, ds_slake_clean])
        except Exception as e:
            print(f"⚠️ Merge failed (using VQA-RAD only for XAI): {e}")
            ds_super_kb = ds_rad

        # Check if 'idx' exists (from fallback or accidental carry-over) and remove it first
        if "idx" in ds_super_kb.column_names:
            ds_super_kb = ds_super_kb.remove_columns("idx")

        # Add index for the Super KB
        ds_super_kb = ds_super_kb.add_column("idx", range(len(ds_super_kb)))

        # Build the XAI Index
        retriever_xai.build_index(ds_super_kb, index_name="xai_super_kb")

        # --- C. ATTACH XAI RETRIEVER TO ENGINE ---
        # We attach it as a separate attribute so xai.py can find it
        inference_engine.xai_retriever = retriever_xai

    else:
        # Standard flow without RAG
        inference_engine = llm

    LOADED_CONFIG = {
        "model": model_key,
        "rag": use_rag,
        "reflexion": use_reflexion,
        "prompt": prompt_style,
        "k": k_val,
        "alpha": alpha_val
    }

    return get_status_msg(LOADED_CONFIG), LOADED_CONFIG, gr.update(interactive=False)


def run_gui_inference(dataset_input, idx, model_key, use_rag, use_reflexion, prompt_style, k_val, alpha_val):
    global inference_engine, LOADED_CONFIG

    empty_rag = [gr.update(visible=False), None, None, gr.update(visible=False)] * MAX_RAG_K
    empty_xai = "**Run XAI Analysis** to generate the report."
    empty_report = "**Run LLM as Judge Evaluation** to generate the report."
    reset_results = (None, None, "", "", "", "", "No retrieval", None, empty_xai, empty_xai, None, empty_xai, empty_report, empty_report, None, *empty_rag)

    # --- CHECK 1: Is engine loaded? ---
    if inference_engine is None or LOADED_CONFIG is None:
        error_msg = """
            ## ⚠️ **Error:** No model loaded.\n
            Please select a model and click 'Initialize Engine'.
        """
        return error_msg, *reset_results[1:]

    # --- CHECK 2: Do settings match? ---
    current_settings = {
        "model": model_key,
        "rag": use_rag,
        "reflexion": use_reflexion,
        "prompt": prompt_style,
        "k": k_val,
        "alpha": alpha_val
    }

    if current_settings != LOADED_CONFIG:
        changes = [k for k, v in current_settings.items() if LOADED_CONFIG[k] != v]
        error_msg = f"""
            ## ⚠️ **Settings Mismatch**\n
            You changed: `{', '.join(changes)}`\n
            Please click **'Initialize / Reload Engine'** to apply changes before running inference.
        """
        return error_msg, *reset_results[1:]

    # --- INFERENCE LOGIC ---
    try:
        dataset_path = DATASET_OPTIONS[dataset_input]["path"]
        dataset = load_data_source(dataset_path, split="test")
        item = dataset[int(idx)]
        image = item['image'].convert("RGB")
        question = item['question']
        gt = str(item['answer'])

        if use_reflexion:
            raw_pred = inference_engine.reflexion_generate(image, question)
        else:
            raw_pred = inference_engine.generate(image, question)

        prediction_text = raw_pred.get('prediction', '').strip()

        # res_text = "### **Question**\n"
        # res_text += f"- {question}\n"
        # res_text += "---\n"
        # res_text += "### **Ground Truth**\n"
        # res_text += f"- {gt}\n"
        # res_text += "---\n"
        # res_text += "### **Prediction**\n"
        # res_text += f"{prediction_text}"

        draft_text = raw_pred.get('reflexion_draft', 'N/A')
        critique_text = raw_pred.get('reflexion_critique', 'N/A')

        reflex_intermediate_output = ""
        if use_reflexion:
            reflex_intermediate_output = f"""
                ### **Draft**\n
                - {draft_text}\n
                ### **Critique**\n
                - {critique_text}
            """
        else:
            reflex_intermediate_output = "None (Reflexion Disabled)"

        retrieved_ids = raw_pred.get("retrieved_ids", [])
        retrieval_log = f"Retrieved Indices: {retrieved_ids}" if retrieved_ids else "None (RAG Disabled)"

        rag_updates = []
        saved_rag_items = []

        train_ds = None
        if retrieved_ids:
            train_ds = load_dataset("flaviagiammarino/vqa-rad", split="train")

        for i in range(MAX_RAG_K):
            if retrieved_ids and i < len(retrieved_ids):
                r_id = retrieved_ids[i]
                r_item = train_ds[int(r_id)]
                r_img = r_item['image'].convert("RGB")
                r_q = r_item['question']
                r_a = str(r_item['answer'])

                saved_rag_items.append({
                    "img": r_img,
                    "q": r_q,
                    "a": r_a
                })

                r_txt = f"""
                    ### **Question**\n
                    - {r_q}\n\n
                    ### **Ground Truth**\n
                    - {r_a}
                """

                rag_updates.extend([gr.update(visible=True), r_img, r_txt, gr.update(value="")])
            else:
                rag_updates.extend([gr.update(visible=False), None, None, gr.update(visible=False)])

        state_data = {
            "image": image,
            "question": question,
            "gt": gt,
            "prediction": prediction_text,
            "draft": draft_text,
            "critique": critique_text,
            "rag_items": saved_rag_items
        }

        q = f"- {question}"
        a = f"- {gt}"

        # Success Return
        valid_status = get_status_msg(LOADED_CONFIG)
        return valid_status, image, q, a, prediction_text, reflex_intermediate_output, retrieval_log, None, empty_xai, empty_xai, None, empty_xai, empty_report, empty_report, state_data, *rag_updates

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"## ❌ **System Error:** {str(e)}", *reset_results[1:]


def run_xai_only(state_data):
    global inference_engine

    # Validation
    if not state_data:
        return None, "⚠️ **Action Required:** Please run **Inference** first."

    image = state_data["image"]
    question = state_data["question"]
    prediction_text = state_data["prediction"]

    adapter = inference_engine.llm if hasattr(inference_engine, 'llm') else inference_engine

    # 1. Visual XAI (6x6 Grid)
    xai_img = None
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        explainer = FastOcclusionExplainer(adapter)
        # 6x6 Grid for good detail
        mask = explainer.generate_heatmap(image, question, prediction_text, grid_size=6)
        xai_img = apply_colormap(image, mask)
    except Exception as e:
        print(f"XAI Error: {e}")
        xai_img = None  # Return None on error

    # 2. Visual Retrieval Counterfactual
    cf_visual_img = None
    cf_visual_rpt = ""
    try:
        # Only works if RAG is on
        if hasattr(inference_engine, 'xai_retriever') and inference_engine.xai_retriever:
            visual_cf = RetrievalCounterfactual(inference_engine)
            cf_visual_img, cf_visual_rpt = visual_cf.find_counterfactual(image, question, prediction_text)
        else:
            cf_visual_rpt = "RAG is disabled. Enable RAG to generate Visual Counterfactuals."
    except Exception as e:
        cf_visual_rpt = f"Visual CF Failed: {e}"

    # 3. oncept Analysis (BioMedCLIP)
    concept_rpt_zs = ""
    concept_rpt_nc = ""
    try:
        if hasattr(inference_engine, 'retriever'):
            c_explainer = ConceptExplainer(inference_engine)
            # Strategy A: Zero-Shot (Fixed Check)
            _, concept_rpt_zs = c_explainer.evaluate(image)

            # Strategy B: Neighbor Consensus (Discovery)
            concept_rpt_nc = c_explainer.discover_concepts(image, question, k=50)
        else:
            concept_rpt_zs = "RAG disabled. Cannot run Concept Analysis."
            concept_rpt_nc = "RAG disabled. Cannot run Concept Analysis."
    except Exception as e:
        concept_rpt_zs = f"Concept Analysis Failed: {e}"
        concept_rpt_nc = f"Concept Analysis Failed: {e}"

    return xai_img, concept_rpt_zs, concept_rpt_nc, cf_visual_img, cf_visual_rpt


def run_full_evaluation(state_data):
    """
    Runs BOTH the Reflexion Judge (Text) and the RAG Judge (Visual) sequentially.
    Returns two separate report strings.
    """
    if not state_data:
        error_msg = "⚠️ **Action Required:** Please run **Inference** first."
        return error_msg, error_msg, *[gr.update(value="") for _ in range(MAX_RAG_K)]

    # --- PART A: REFLEXION JUDGE (Gemini 2.5) ---
    reflexion_report = ""
    try:
        # Extract Data
        question = state_data.get("question")
        gt = state_data.get("gt", "N/A")
        prediction_text = state_data.get("prediction")
        draft = state_data.get("draft", "N/A")
        critique = state_data.get("critique", "N/A")

        if draft == "N/A" or critique == "N/A":
            reflexion_report = """
            ## ⚠️ Evaluation Skipped
            Reflexion was not enabled. Enable **Reflexion** in Settings to test this.
            """
        else:
            judge = ReflexionJudge(api_key=GOOGLE_API_KEY)
            reflexion_report = judge.evaluate(question, gt, draft, critique, prediction_text)

    except Exception as e:
        reflexion_report = f"## ❌ Reflexion Judge Error\n{str(e)}"

    # --- PART B: RAG JUDGE (Gemini 2.0 Vision) ---
    rag_summary_text = ""
    rag_item_updates = []

    try:
        rag_items = state_data.get("rag_items", [])
        user_q = state_data.get("question", "")

        if not rag_items:
            rag_summary_text = """
            ## ⚠️ Evaluation Skipped
            No RAG items found. RAG was not enabled. Enable **RAG** in Settings to test this.
            """
            rag_item_updates = [gr.update(value="", visible=False) for _ in range(MAX_RAG_K)]
        else:
            rag_judge = RAGJudge(api_key=GOOGLE_API_KEY)
            json_data, summary = rag_judge.evaluate_batch(user_q, rag_items)

            rag_summary_text = summary
            evals = json_data.get("items", [])

            for i in range(MAX_RAG_K):
                if i < len(evals):
                    e = evals[i]

                    # Styled Verdict Card
                    color = "red" if "IRRELEVANT" in e['verdict'].upper() else "green"
                    icon = "✅" if color == "green" else "❌"

                    card = f"""
                        ### {icon} {e['verdict']}
                        * **Visual:** {e['visual_check']}
                        * **Semantic:** {e['semantic_check']}
                        * **Reason:** {e['reasoning']}
                    """
                    rag_item_updates.append(gr.update(value=card, visible=True))
                else:
                    rag_item_updates.append(gr.update(value="", visible=False))

    except Exception as e:
        rag_summary_text = f"## ❌ RAG Judge Error\n{str(e)}"
        rag_item_updates = [gr.update(value="") for _ in range(MAX_RAG_K)]

    # RETURN BOTH REPORTS
    return reflexion_report, rag_summary_text, *rag_item_updates


def reset_system():
    global inference_engine, LOADED_CONFIG

    # 1. Clear Global State
    inference_engine = None
    LOADED_CONFIG = None

    # 2. Force GPU Memory Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # 3. Garbage Collect
    gc.collect()

    # 4. Prepare "Empty" Return Values
    # RAG items: 4 items per row (Row, Img, Txt, Verdict) * 10 rows
    empty_rag = [gr.update(visible=False), None, None, gr.update(visible=False)] * MAX_RAG_K

    # Return values matching the UI outputs
    return (
        "## ⚪ System Status: Idle\nModel unloaded & Memory cleared.",  # status
        None,  # img_out
        "",  # question
        "",  # gt
        "",  # prediction_text
        "",  # reflex_intermediate_out
        "",  # retrieval_log_out
        None,  # occlusion_out
        "",  # concept_rpt_zs_out
        "",  # concept_rpt_nc_out
        None,  # cf_visual_out
        "",  # cf_visual_report_out
        "",  # reflex_judge_report_out
        "",  # rag_judge_summary_out
        None,  # prediction_state
        *empty_rag  # rag_atomic_components
    )


# --- Gradio Interface ---
# my_theme = gr.themes.Soft(
#     text_size="lg",   # Increases font size
#     spacing_size="lg" # Increases padding/clickable area
# )

with gr.Blocks(theme="Soft", title="GEN-MED") as demo:
    last_loaded_state = gr.State(None)
    prediction_state = gr.State()

    with gr.Sidebar(position="left", width="25%"):
        status = gr.Markdown("## ⚪ System Status: Idle")

        with gr.Tab("Model"):
            gr.Markdown("## ⚙️ Engine Setup")
            model_select = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()),
                label="Select Model / Adapter Configuration",
                value=list(MODEL_OPTIONS.keys())[7]
            )
            prompt_input = gr.Radio(["Basic", "Instruct"], label="Prompt Style", value=CONFIG["PROMPT"])

        with gr.Tab("Setting"):
            gr.Markdown("## 🧩 Techniques (Toggles)")
            rag_toggle = gr.Checkbox(label="Enable RAG", value=CONFIG["USE_RAG"])
            reflexion_toggle = gr.Checkbox(label="Enable Reflexion", value=CONFIG["USE_REFLEXION"])

            with gr.Group(visible=CONFIG["USE_RAG"]) as rag_settings_group:
                with gr.Accordion("RAG Settings", open=True):
                    k_slider = gr.Slider(1, MAX_RAG_K, value=CONFIG["RAG_K"], step=1, label="K (Neighbors)")
                    a_slider = gr.Slider(0.0, 1.0, value=CONFIG["RAG_ALPHA"], label="Alpha (Text similarity) : 1-Alpha (Image similarity)")

        with gr.Tab("Inference"):
            gr.Markdown("## 🔍 Inference")

            with gr.Row():
                dataset_input = gr.Dropdown(["VQA-RAD", "SLAKE"], value="VQA-RAD", label="Dataset")
                idx_input = gr.Number(
                    value=0,
                    label=f"Sample Index (0 - {DATASET_OPTIONS['VQA-RAD']['size'] - 1})",
                    precision=0,
                    minimum=0,
                    maximum=DATASET_OPTIONS["VQA-RAD"]["size"] - 1
                )

            load_btn = gr.Button("🔄 Initialize / Reload Engine", variant="primary")
            run_btn = gr.Button("🚀 Run Inference", variant="secondary")
            xai_btn = gr.Button("⚡ Run XAI Analysis", variant="secondary")
            judge_btn = gr.Button("⚖️ Run LLM as Judge Evaluation", variant="secondary")
            reset_btn = gr.Button("⚠️ Reset System (Clear Memory)", variant="stop")

    icon_html = get_img_html("../assets/GENMED_XAI.png")
    gr.Markdown(f"# {icon_html} GEN-MED Dashboard")
    gr.Markdown("---")
    result_head, result_body = create_md_accordion("## 📊 Result Analysis", initial_visible=True)
    # gr.Markdown("## 📊 Result Analysis")

    with result_body:
        gr.Markdown("---")
        with gr.Row(height=350):
            with gr.Column(scale=1):
                img_out = gr.Image(type="pil", show_label=False, height=350)
            with gr.Column(scale=2):
                with gr.Row(height=150):
                    with gr.Column():
                        gr.Markdown("### **Question**")
                        gr.Markdown("---")
                        question = gr.Markdown(height=150)
                    with gr.Column():
                        gr.Markdown("### **Ground Truth**")
                        gr.Markdown("---")
                        gt = gr.Markdown(height=150)

                gr.Markdown("---")

                with gr.Row(height=200):
                    with gr.Column():
                        gr.Markdown("### **Prediction**")
                        gr.Markdown("---")
                        prediction = gr.Markdown(height=200)

        with gr.Tab("Deep Analysis (XAI)"):
            gr.Markdown("## 🔬 Comprehensive Analysis")
            gr.Markdown("---")
            with gr.Row():
                with gr.Column():
                    occ_head, occ_body = create_md_accordion("### 👁️ Occlusion Sensitivity", initial_visible=False)
                    with occ_body:
                        occlusion_out = gr.Image(type="pil", height=350, show_label=False)
                with gr.Column():
                    cav_zs_head, cav_zs_body = create_md_accordion("### 📝 Concept Analysis (Zero-Shot)", initial_visible=False)
                    with cav_zs_body:
                        concept_rpt_zs_out = gr.Markdown(container=True)
                with gr.Column():
                    cav_nc_head, cav_nc_body = create_md_accordion("### 📝 Concept Analysis (Neighbour-Based)", initial_visible=False)
                    with cav_nc_body:
                        concept_rpt_nc_out = gr.Markdown(container=True)

            with gr.Row():
                with gr.Column():
                    cfv_head, cfv_body = create_md_accordion("### 📜️ Visual Counterfactual (Similar Opposite)", initial_visible=False)
                    with cfv_body:
                        with gr.Row():
                            with gr.Column(scale=1):
                                cf_visual_out = gr.Image(type="pil", height=350, label="Nearest 'Opposite' Case")
                            with gr.Column(scale=2):
                                cf_visual_report_out = gr.Markdown(container=True)

        with gr.Tab("Deep Analysis (RAG)"):
            gr.Markdown("## 📂 Visual RAG Retrieval Items & Evaluation")
            gr.Markdown("---")
            retrieval_log_out = gr.Textbox(show_label=False, lines=1, max_lines=1)

            rag_item_head, rag_item_body = create_md_accordion("### 🗃️ Retrieved Item List", initial_visible=False)
            with rag_item_body:
                rag_atomic_components = []
                rag_verdict_boxes = []

                for i in range(MAX_RAG_K):
                    with gr.Row(visible=False, variant="panel") as r:
                        # Column 1: Image
                        with gr.Column(scale=1):
                            r_img = gr.Image(type="pil", label=f"Retrieved Item {i + 1}", height=250)

                        # Column 2: Text Metadata
                        with gr.Column(scale=1):
                            r_txt = gr.Markdown(height=250, label="Metadata")

                        # Column 3: The Verdict Card (Populated by JSON)
                        with gr.Column(scale=1):
                            r_eval = gr.Markdown()

                    rag_atomic_components.extend([r, r_img, r_txt, r_eval])
                    rag_verdict_boxes.append(r_eval)

            rag_report_head, rag_report_body = create_md_accordion("### 📋 AI Judge Assessment (Gemini 3 Flash) - Overall Retrieval Health", initial_visible=False)
            with rag_report_body:
                rag_judge_summary_out = gr.Markdown(container=True)

        with gr.Tab("Deep Analysis (Reflexion)"):
            gr.Markdown("## 🧠 Reflexion Process & Evaluation")
            gr.Markdown("---")

            reflex_res_head, reflex_res_body = create_md_accordion("### 📝 Internal Monologue (Draft & Critique)", initial_visible=False)
            with reflex_res_body:
                reflex_intermediate_out = gr.Markdown(container=True)

            reflex_report_head, reflex_report_body = create_md_accordion("### 📋 AI Judge Assessment (Gemini 2.5 Flash Lite)", initial_visible=False)
            with reflex_report_body:
                reflex_judge_report_out = gr.Markdown(container=True)

    # Bindings
    all_inputs_and_buttons = [
        # Inputs
        dataset_input, idx_input,
        model_select, prompt_input,
        rag_toggle, reflexion_toggle,
        k_slider, a_slider,
        # Buttons
        load_btn, run_btn, xai_btn, judge_btn
    ]

    def lock_interface():
        # Returns "interactive=False" for every component in the list
        return [gr.update(interactive=False)] * len(all_inputs_and_buttons)

    def unlock_interface():
        # Returns "interactive=True" for every component in the list
        return [gr.update(interactive=True)] * len(all_inputs_and_buttons)

    load_btn.click(
        load_engine,
        inputs=[model_select, rag_toggle, reflexion_toggle, prompt_input, k_slider, a_slider],
        outputs=[status, last_loaded_state, load_btn]
    )

    run_btn.click(
        fn=lock_interface,
        inputs=None,
        outputs=all_inputs_and_buttons,
        queue=False
    ).then(
        run_gui_inference,
        inputs=[dataset_input, idx_input, model_select, rag_toggle, reflexion_toggle, prompt_input, k_slider, a_slider],
        outputs=[status, img_out, question, gt, prediction, reflex_intermediate_out, retrieval_log_out, occlusion_out, concept_rpt_zs_out, concept_rpt_nc_out,
                 cf_visual_out, cf_visual_report_out, reflex_judge_report_out, rag_judge_summary_out, prediction_state, *rag_atomic_components]
    ).then(
        fn=unlock_interface,
        inputs=None,
        outputs=all_inputs_and_buttons
    )

    xai_btn.click(
        fn=lock_interface,
        inputs=None,
        outputs=all_inputs_and_buttons,
        queue=False
    ).then(
        run_xai_only,
        inputs=[prediction_state],
        outputs=[occlusion_out, concept_rpt_zs_out, concept_rpt_nc_out, cf_visual_out, cf_visual_report_out]
    ).then(
        fn=unlock_interface,
        inputs=None,
        outputs=all_inputs_and_buttons
    )

    judge_btn.click(
        fn=lock_interface,
        inputs=None,
        outputs=all_inputs_and_buttons,
        queue=False
    ).then(
        run_full_evaluation,
        inputs=[prediction_state],
        outputs=[reflex_judge_report_out, rag_judge_summary_out, *rag_verdict_boxes]
    ).then(
        fn=unlock_interface,
        inputs=None,
        outputs=all_inputs_and_buttons
    )

    reset_btn.click(
        fn=lock_interface,
        inputs=None,
        outputs=all_inputs_and_buttons,
        queue=False
    ).then(
        fn=reset_system,
        inputs=None,
        # Updates ALL output visual components
        outputs=[status, img_out, question, gt, prediction, reflex_intermediate_out, retrieval_log_out, occlusion_out, concept_rpt_zs_out, concept_rpt_nc_out,
                 cf_visual_out, cf_visual_report_out, reflex_judge_report_out, rag_judge_summary_out, prediction_state, *rag_atomic_components]
    ).then(
        fn=unlock_interface,  # Force Unlock everything
        inputs=None,
        outputs=all_inputs_and_buttons
    )

    def check_for_changes(model, rag, reflexion, prompt, k, alpha, last_config):
        if last_config is None:
            return gr.update(interactive=True), gr.update()

        current_ui = {
            "model": model,
            "rag": rag,
            "reflexion": reflexion,
            "prompt": prompt,
            "k": k,
            "alpha": alpha
        }

        labels = {
            "model": "Model Selection",
            "rag": "RAG Technique",
            "reflexion": "Reflexion Workflow",
            "prompt": "Prompt Style",
            "k": "RAG K-Value",
            "alpha": "RAG Alpha-Weight"
        }

        changed_names = [labels[k] for k, v in current_ui.items() if v != last_config.get(k)]

        if changed_names:
            changes_str = ", ".join(changed_names)
            yellow_status = f"""## 🟡 System Status: Pending\n
                **Warning:** Settings changed. (`{changes_str}`)\n
                `Reload Engine` required to apply changes.
            """
            return gr.update(interactive=True), yellow_status
        else:
            rag_txt = f"✅ Enabled (K={last_config['k']}, α={last_config['alpha']})" if last_config["rag"] else "❌ Disabled"
            ref_txt = "✅ Enabled" if last_config["reflexion"] else "❌ Disabled"

            green_status = f"""## 🟢 System Status: Ready\n
                **Model:** `{last_config['model']}`\n
                **Reflexion:** {ref_txt}\n
                **RAG:** {rag_txt}
            """

            return gr.update(interactive=False), green_status

    settings_inputs = [model_select, rag_toggle, reflexion_toggle, prompt_input, k_slider, a_slider]
    for component in settings_inputs:
        component.change(
            fn=check_for_changes,
            inputs=settings_inputs + [last_loaded_state],
            outputs=[load_btn, status]
        )


    def update_dataset_info(dataset_name):
        size = DATASET_OPTIONS.get(dataset_name, {}).get("size", 1)
        return gr.update(
            maximum=size - 1,
            label=f"Sample Index (0 - {size - 1})",
            value=0
        )

    dataset_input.change(
        fn=update_dataset_info,
        inputs=dataset_input,
        outputs=idx_input
    )


    def toggle_rag_visibility(is_checked):
        return gr.update(visible=is_checked)

    rag_toggle.change(
        fn=toggle_rag_visibility,
        inputs=rag_toggle,
        outputs=rag_settings_group
    )


if __name__ == "__main__":
    demo.launch(favicon_path="../assets/GENMED_XAI.png", server_port=7860)
