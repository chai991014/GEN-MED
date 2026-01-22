import os
import torch
import gradio as gr
import base64
import gc
from datasets import load_dataset, load_from_disk, concatenate_datasets
from inference.llm_adapter import get_llm_adapter
from inference.rag_pipeline import RAGPipeline
from inference.retriever import MultimodalRetriever
from xai.xai import FastOcclusionExplainer, AttentionExplainer, ConceptExplainer, RetrievalCounterfactual, ReflexionJudge, RAGJudge, apply_colormap
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# --- Default Config ---
CONFIG = {
    "TECH_TAG": None,
    "OUTPUT_FILE": None,
    "DATASET_ID": None,
    "LLAVA_REPO_PATH": "./LLaVA-Med",
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
    "LLaVA-Med 1.5 Mistral 7B (QLoRA)": {"model": "./finetune/qlora-llava", "adapter": None},
    "LLaVA-Med 1.5 Mistral 7B (Dora)": {"model": "./finetune/dora-llava", "adapter": None},
    "Qwen3-VL-2B-Instruct (Base)": {"model": "Qwen/Qwen3-VL-2B-Instruct", "adapter": None},
    "Qwen3-VL-2B-Instruct (QLoRA)": {"model": "Qwen/Qwen3-VL-2B-Instruct", "adapter": "./finetune/qlora-qwen3-2b/checkpoint-600"},
    "Qwen3-VL-2B-Instruct (Dora)": {"model": "Qwen/Qwen3-VL-2B-Instruct", "adapter": "./finetune/dora-qwen3-2b/checkpoint-580"},
    "Qwen3-VL-4B-Instruct (Base)": {"model": "Qwen/Qwen3-VL-4B-Instruct", "adapter": None},
    "Qwen3-VL-4B-Instruct (QLoRA)": {"model": "Qwen/Qwen3-VL-4B-Instruct", "adapter": "./finetune/qlora-qwen3-4b/checkpoint-390"},
    "Qwen3-VL-4B-Instruct (Dora)": {"model": "Qwen/Qwen3-VL-4B-Instruct", "adapter": "./finetune/dora-qwen3-4b/checkpoint-390"},
}

DATASET_OPTIONS = {
    "VQA-RAD": {
        "path": "flaviagiammarino/vqa-rad",
        "size": 451
    },
    "SLAKE": {
        "path": "./slake_vqa_rad_format",
        "size": 1407
    },
}

custom_css = """
.accordion-header {
    padding: 10px;
    background-color: var(--background-fill-secondary);
    margin-bottom: 5px;
}
.accordion-header button {
    background: transparent;
    color: var(--body-text-color);
}
.padded-container {
    padding: 15px; 
}
"""


# --- Global State ---
inference_engine = None
LOADED_CONFIG = None
MAX_RAG_K = 10
GOOGLE_API_KEY = os.getenv("API_KEY")


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

    with gr.Group(visible=visibility):
        # 2. The Header Row
        with gr.Row(visible=visibility, elem_classes="accordion-header") as header_row:
            with gr.Column(scale=9, min_width=0):
                gr.Markdown(label_md, container=True)
            # Unique toggle button
            with gr.Column(scale=1, min_width=0):
                btn_label = "Hide 🔼" if initial_visible else "Show 🔽"
                btn = gr.Button(btn_label, size="sm", scale=0, min_width=1)

        # 3. The Content Group
        content_group = gr.Column(visible=initial_visible, elem_classes="padded-container")

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

    return header_row, content_group, is_open, btn


def get_open_acc():
    return gr.update(visible=True), True, "Hide 🔼"


def get_closed_acc():
    return gr.update(visible=False), False, "Show 🔽"


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

    print("\n🧠 Building XAI Super-Index (VQA-RAD + SLAKE)...")
    try:
        retriever_xai = MultimodalRetriever(device="cpu")  # Keep on CPU to save VRAM

        # Load VQA-RAD
        ds_rad = load_dataset("flaviagiammarino/vqa-rad", split="train")

        # Load SLAKE (Robust Merge)
        try:
            ds_slake = load_data_source(DATASET_OPTIONS["SLAKE"]["path"], "train")

            # Column Cleanup
            common_cols = ["image", "question", "answer"]
            ds_rad_clean = ds_rad.select_columns(common_cols)

            # Cast SLAKE image format if needed
            if "image" in ds_slake.features and ds_slake.features["image"] != ds_rad.features["image"]:
                from datasets import Image as DImage
                ds_slake = ds_slake.cast_column("image", DImage())

            ds_slake_clean = ds_slake.select_columns(common_cols)
            ds_super_kb = concatenate_datasets([ds_rad_clean, ds_slake_clean])
        except Exception as e:
            print(f"⚠️ SLAKE Merge failed (using VQA-RAD only for XAI): {e}")
            ds_super_kb = ds_rad

        # Create unique index for XAI lookup
        if "idx" in ds_super_kb.column_names:
            ds_super_kb = ds_super_kb.remove_columns("idx")
        ds_super_kb = ds_super_kb.add_column("idx", range(len(ds_super_kb)))

        # Build Index (with unique name to avoid conflict)
        retriever_xai.build_index(ds_super_kb, index_name="xai_super_kb")

    except Exception as e:
        print(f"❌ XAI Retriever Init Failed: {e}")
        retriever_xai = None

    # 3. RAG Wrapper (Applied if RAG is toggled, regardless of Reflexion)
    if config["USE_RAG"]:
        print("\n⚡ Building Inference Index (VQA-RAD)...")
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
    else:
        # Standard flow without RAG
        inference_engine = llm

    inference_engine.xai_retriever = retriever_xai

    LOADED_CONFIG = {
        "model": model_key,
        "rag": use_rag,
        "reflexion": use_reflexion,
        "prompt": prompt_style,
        "k": k_val,
        "alpha": alpha_val
    }

    return get_status_msg(LOADED_CONFIG), LOADED_CONFIG, gr.update(interactive=False)


def run_database_inference(dataset_input, idx, model_key, use_rag, use_reflexion, prompt_style, k_val, alpha_val):
    """
    outputs = [
    status, img_out, question, gt, prediction, reflex_intermediate_out, retrieval_log_out,
    attention_out, occlusion_out, concept_rpt_zs_out, concept_rpt_nc_out, cf_visual_out, cf_visual_report_out,
    reflex_judge_report_out, rag_judge_summary_out, prediction_state,
    *rag_atomic_components, *ref_updates, *rag_updates_acc,
    *att_updates, *occ_updates, *zs_updates, *nc_updates, *cf_updates,
    *ref_judge_updates, *rag_judge_updates,

    reflex_intermediate_output, retrieval_log,
    attention_out, occlusion_out, attention_out, occlusion_out,
    concept_rpt_zs_out, concept_rpt_nc_out, cf_visual_out, cf_visual_report_out,
    reflex_judge_report_out, rag_judge_summary_out,
    *rag_atomic_components, *ref_updates, *rag_updates_acc,
    *rag_atomic_components, *rag_updates_acc,
    *att_updates, *occ_updates, *stk_aud_vis_updates, *zs_updates, *nc_updates, *cf_updates,
    *ref_judge_updates, *rag_judge_updates,
    ]
    """
    global inference_engine, LOADED_CONFIG

    empty_rag = [gr.update(visible=False), None, None, gr.update(visible=False)] * MAX_RAG_K
    empty_xai = "**Run XAI Analysis** to generate the report."
    empty_report = "**Run LLM Judge Evaluation** to generate the report."
    closed_acc = get_closed_acc()
    reset_results = (
        None, None, "", "", "", "", "No retrieval",
        None, None, empty_xai, empty_xai, None, empty_xai,
        empty_report, empty_report, None,
        *empty_rag, *closed_acc, *closed_acc,
        *closed_acc, *closed_acc, *closed_acc, *closed_acc, *closed_acc,
        *closed_acc, *closed_acc,

        "", "No retrieval",
        None, None, None, None,
        empty_xai, empty_xai, None, empty_xai,
        empty_report, empty_report,
        *empty_rag, *closed_acc, *closed_acc,
        *empty_rag, *closed_acc,
        *closed_acc, *closed_acc, *closed_acc, *closed_acc, *closed_acc, *closed_acc,
        *closed_acc, *closed_acc
    )

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

        idx_int = int(idx)
        if idx_int < 0 or idx_int >= len(dataset):
            error_msg = f"⚠️ **Error:** Index {idx_int} is out of bounds.\nValid range: 0 to {len(dataset) - 1}."
            return error_msg, *reset_results

        item = dataset[int(idx)]
        image = item['image'].convert("RGB")
        question = item['question']
        gt = str(item['answer'])

        if use_reflexion:
            raw_pred = inference_engine.reflexion_generate(image, question)
        else:
            raw_pred = inference_engine.generate(image, question)

        prediction_text = raw_pred.get('prediction', '').strip()

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
        stk_rag_updates = []
        stk_aud_rag_updates = []
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

                rag_updates.extend([gr.update(visible=True), r_img, r_txt, gr.update(value="", visible=True)])
                stk_rag_updates.extend([gr.update(visible=True), r_img, r_txt, gr.update(value="", visible=True)])
                stk_aud_rag_updates.extend([gr.update(visible=True), r_img, r_txt, gr.update(value="", visible=True)])
            else:
                rag_updates.extend([gr.update(visible=False), None, None, gr.update(visible=False)])
                stk_rag_updates.extend([gr.update(visible=False), None, None, gr.update(visible=False)])
                stk_aud_rag_updates.extend([gr.update(visible=False), None, None, gr.update(visible=False)])

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

        # Reflexion: Open if enabled
        ref_updates = get_open_acc() if use_reflexion else closed_acc

        # RAG: Open if items were retrieved
        rag_updates_acc = get_open_acc() if retrieved_ids else closed_acc

        return (
            valid_status, image, q, a, prediction_text, reflex_intermediate_output, retrieval_log,
            None, None, empty_xai, empty_xai, None, empty_xai,
            empty_report, empty_report, state_data,
            *rag_updates, *ref_updates, *rag_updates_acc,
            *closed_acc, *closed_acc, *closed_acc, *closed_acc, *closed_acc,
            *closed_acc, *closed_acc,

            reflex_intermediate_output, retrieval_log,
            None, None, None, None,
            empty_xai, empty_xai, None, empty_xai,
            empty_report, empty_report,
            *stk_rag_updates, *ref_updates, *rag_updates_acc,
            *stk_aud_rag_updates, *rag_updates_acc,
            *closed_acc, *closed_acc, *closed_acc, *closed_acc, *closed_acc, *closed_acc,
            *closed_acc, *closed_acc
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"## ❌ **System Error:** {str(e)}", *reset_results[1:]


def run_xai(state_data):
    global inference_engine, LOADED_CONFIG

    closed_acc = get_closed_acc()

    # Validation
    if not state_data:
        error_msg = "⚠️ **Action Required:** Please run **Inference** first."
        return (
            None, None, error_msg, error_msg, None, error_msg, *closed_acc, *closed_acc, *closed_acc, *closed_acc, *closed_acc,
            None, None, None, None, error_msg, error_msg, None, error_msg, *closed_acc, *closed_acc, *closed_acc, *closed_acc, *closed_acc
        )

    image = state_data["image"]
    question = state_data["question"]
    prediction_text = state_data["prediction"]

    adapter = inference_engine.llm if hasattr(inference_engine, 'llm') else inference_engine

    # 1. Visual XAI
    # --- METHOD 1: Qwen Attention (Intrinsic) ---
    # Fast, checks internal weights. Only works for Qwen.
    img_att = None
    try:
        model_name = LOADED_CONFIG["model"]
        if "qwen" in str(model_name).lower():
            print("\n⚡ Running Intrinsic Attention...")
            explainer_att = AttentionExplainer(adapter)
            att_mask = explainer_att.generate_heatmap(image, question)
            img_att = apply_colormap(image, att_mask)
    except Exception as e:
        print(f"⚠️ Attention XAI Failed: {e}")

    # --- METHOD 2: Occlusion (Perturbation) ---
    # Slower, but proves causality (removing this region breaks the answer).
    img_occ = None
    try:
        print("🐢 Running Sliding Occlusion...")
        explainer_occ = FastOcclusionExplainer(adapter)
        occ_mask = explainer_occ.generate_heatmap(image, question, prediction_text, grid_size=8)
        img_occ = apply_colormap(image, occ_mask)
    except Exception as e:
        print(f"⚠️ Occlusion XAI Failed: {e}")

    # 2. Visual Retrieval Counterfactual
    print("🔄 Running Visual Retrieval Counterfactual...")
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

    # 3. Concept Analysis (BioMedCLIP)
    concept_rpt_zs = ""
    concept_rpt_nc = ""
    try:
        if hasattr(inference_engine, 'retriever'):
            c_explainer = ConceptExplainer(inference_engine)
            # Strategy A: Zero-Shot (Fixed Check)
            print("🧠 Running Concept Activation Analysis...")
            _, concept_rpt_zs = c_explainer.evaluate(image)

            # Strategy B: Neighbor Consensus (Discovery)
            print("🧬 Running Neighbor-Based Concept Consensus...")
            concept_rpt_nc = c_explainer.discover_concepts(image, question, k=50)
        else:
            concept_rpt_zs = "RAG disabled. Cannot run Concept Analysis."
            concept_rpt_nc = "RAG disabled. Cannot run Concept Analysis."
    except Exception as e:
        concept_rpt_zs = f"Concept Analysis Failed: {e}"
        concept_rpt_nc = f"Concept Analysis Failed: {e}"

    att_updates = get_open_acc() if img_att is not None else closed_acc
    occ_updates = get_open_acc() if img_occ is not None else closed_acc
    zs_updates = get_open_acc() if concept_rpt_zs and "RAG disabled" not in concept_rpt_zs else closed_acc
    nc_updates = get_open_acc() if concept_rpt_nc and "RAG disabled" not in concept_rpt_nc else closed_acc
    cf_updates = get_open_acc() if cf_visual_img is not None else closed_acc
    stk_aud_vis_updates = get_open_acc() if img_att is not None or img_occ is not None else closed_acc

    return (
        img_att, img_occ, concept_rpt_zs, concept_rpt_nc, cf_visual_img, cf_visual_rpt,
        *att_updates, *occ_updates, *zs_updates, *nc_updates, *cf_updates,
        img_att, img_occ, img_att, img_occ,
        concept_rpt_zs, concept_rpt_nc, cf_visual_img, cf_visual_rpt,
        *att_updates, *occ_updates, *stk_aud_vis_updates, *zs_updates, *nc_updates, *cf_updates
    )


def run_llm_judge(state_data):
    """
    Runs BOTH the Reflexion Judge (Text) and the RAG Judge (Visual) sequentially.
    Returns two separate report strings.
    """
    closed_acc = get_closed_acc()

    if not state_data:
        error_msg = "⚠️ **Action Required:** Please run **Inference** first."
        return (
            error_msg, error_msg, *[gr.update(value="") for _ in range(MAX_RAG_K)], *closed_acc, *closed_acc,
            error_msg, error_msg, *[gr.update(value="") for _ in range(MAX_RAG_K)], *closed_acc, *closed_acc, *closed_acc
        )

    # --- PART A: REFLEXION JUDGE (Gemini 2.5) ---
    reflexion_report = ""
    try:
        print("\n⚖️Start Reflexion Judge")
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
    rag_item_updates_deep = []
    rag_item_updates_stk = []

    try:
        print("⚖️Start RAG Judge")
        rag_items = state_data.get("rag_items", [])
        user_q = state_data.get("question", "")

        if not rag_items:
            rag_summary_text = """
            ## ⚠️ Evaluation Skipped
            No RAG items found. RAG was not enabled. Enable **RAG** in Settings to test this.
            """
            rag_item_updates_deep = [gr.update(value="", visible=False) for _ in range(MAX_RAG_K)]
            rag_item_updates_stk = [gr.update(value="", visible=False) for _ in range(MAX_RAG_K)]
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
                    rag_item_updates_deep.append(gr.update(value=card, visible=True))
                    rag_item_updates_stk.append(gr.update(value=card, visible=True))
                else:
                    rag_item_updates_deep.append(gr.update(value="", visible=False))
                    rag_item_updates_stk.append(gr.update(value="", visible=False))

    except Exception as e:
        rag_summary_text = f"## ❌ RAG Judge Error\n{str(e)}"
        rag_item_updates_deep = [gr.update(value="") for _ in range(MAX_RAG_K)]
        rag_item_updates_stk = [gr.update(value="") for _ in range(MAX_RAG_K)]

    ref_judge_updates = get_open_acc() if "Reflexion disabled" not in reflexion_report else closed_acc
    rag_judge_updates = get_open_acc() if "No RAG items" not in rag_summary_text else closed_acc

    # RETURN BOTH REPORTS
    return (
        reflexion_report, rag_summary_text, *rag_item_updates_deep, *ref_judge_updates, *rag_judge_updates,
        reflexion_report, rag_summary_text, *rag_item_updates_stk, *ref_judge_updates, *rag_judge_updates, *rag_judge_updates
    )


def reset_system():
    global inference_engine, LOADED_CONFIG

    print("🔄 System Reset. Model unloaded & Memory cleared.")

    # 1. Clear Global State
    inference_engine = None
    LOADED_CONFIG = None

    # 2. Force GPU Memory Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # 3. Garbage Collect
    gc.collect()

    closed_acc = get_closed_acc()

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

        None,  # attention_out
        None,  # occlusion_out
        "",  # concept_rpt_zs_out
        "",  # concept_rpt_nc_out
        None,  # cf_visual_out
        "",  # cf_visual_report_out

        "",  # reflex_judge_report_out
        "",  # rag_judge_summary_out
        None,  # prediction_state

        *empty_rag,  # rag_atomic_components
        *closed_acc, *closed_acc,  # Reflexion, RAG Items
        *closed_acc, *closed_acc, *closed_acc, *closed_acc, *closed_acc,  # XAI (5 items)
        *closed_acc, *closed_acc,  # Judges (2 items)

        # --- NEW: STAKEHOLDER CLEARS ---
        "", "", *empty_rag, *empty_rag,             # Text/RAG
        *closed_acc, *closed_acc,                   # RAG/Ref states
        None, None, None, None, "", "", None, "",   # XAI Images
        *closed_acc, *closed_acc, *closed_acc, *closed_acc, *closed_acc, *closed_acc,  # XAI States
        "", "", *closed_acc, *closed_acc, *closed_acc    # Judges
    )


with gr.Blocks(theme="Soft", title="GEN-MED-X", css=custom_css) as demo:
    last_loaded_state = gr.State(None)
    prediction_state = gr.State()
    icon_html = get_img_html("assets/GENMED.png")
    icon_html_live = get_img_html("assets/GENMED_LIVE.png")
    icon_html_xai = get_img_html("assets/GENMED_XAI.png")

    with gr.Sidebar(position="left", width="25%"):
        gr.Markdown(f"# {icon_html}GEN-MED-X")
        gr.Markdown("### Your personal medical specialist.")
        gr.Markdown("---")
        with gr.Column():
            status = gr.Markdown("## ⚪ System Status: Idle", height=150, max_height=150)

            with gr.Tab("Configuration Setting"):
                gr.Markdown("## ⚙️ Engine Setup")
                model_select = gr.Dropdown(
                    choices=list(MODEL_OPTIONS.keys()),
                    label="Select Model / Adapter Configuration",
                    value=list(MODEL_OPTIONS.keys())[7],
                    filterable=False
                )
                with gr.Accordion("Settings", open=False):
                    prompt_input = gr.Radio(["Basic", "Instruct"], label="Prompt Style", value=CONFIG["PROMPT"])
                    rag_toggle = gr.Checkbox(label="Enable RAG", value=CONFIG["USE_RAG"])
                    reflexion_toggle = gr.Checkbox(label="Enable Reflexion", value=CONFIG["USE_REFLEXION"])

                with gr.Group(visible=CONFIG["USE_RAG"]) as rag_settings_group:
                    with gr.Accordion("RAG Settings", open=False):
                        k_slider = gr.Slider(1, MAX_RAG_K, value=CONFIG["RAG_K"], step=1, label="K (Neighbors)")
                        a_slider = gr.Slider(0.0, 1.0, value=CONFIG["RAG_ALPHA"], label="Alpha (Text similarity) : 1-Alpha (Image similarity)")

                load_btn = gr.Button("🔄 Initialize / Reload Engine", variant="primary")
                reset_btn = gr.Button("⚠️ Reset System (Clear Memory)", variant="stop")

            with gr.Tab("Inference"):
                gr.Markdown("## 🔍 Inference")

                with gr.Accordion("Database Inference", open=True):
                    dataset_input = gr.Dropdown(["VQA-RAD", "SLAKE"], value="VQA-RAD", label="Dataset")
                    idx_input = gr.Number(
                        value=0,
                        label=f"Sample Index (0 - {DATASET_OPTIONS['VQA-RAD']['size'] - 1})",
                        precision=0,
                        minimum=0,
                    )

                run_btn = gr.Button("🚀 Run Inference", variant="secondary")
                xai_btn = gr.Button("⚡ Run XAI Analysis", variant="secondary")
                judge_btn = gr.Button("⚖️ Run LLM Judge Evaluation", variant="secondary")

    with gr.Tab("Live Diagnostic"):
        gr.Markdown(f"## {icon_html_live} Live Diagnostic ChatBot")
        gr.Markdown("---")
        with gr.Column():
            gr.Markdown(f"## {icon_html_xai} XAI Dashboard")
            gr.Markdown("---")

    with gr.Tab("Database Dashboard"):
        gr.Markdown(f"## {icon_html_xai} XAI Dashboard (DATABASE)")
        gr.Markdown("---")
        with gr.Column():
            with gr.Row(height=350):
                with gr.Column(scale=1):
                    img_out = gr.Image(type="pil", show_label=False, height=350)
                with gr.Column(scale=2):
                    with gr.Row(height=150):
                        with gr.Column():
                            gr.Markdown("### **Question**")
                            gr.Markdown("---")
                            question = gr.Markdown(height=150, max_height=150)
                        with gr.Column():
                            gr.Markdown("### **Ground Truth**")
                            gr.Markdown("---")
                            gt = gr.Markdown(height=150, max_height=150)

                    gr.Markdown("---")

                    with gr.Row(height=200):
                        with gr.Column():
                            gr.Markdown("### **Prediction**")
                            gr.Markdown("---")
                            prediction = gr.Markdown(height=200, max_height=200)

            with gr.Tab("Stakeholder Perspectives"):
                gr.Markdown("## 👥 Multi-Stakeholder Explainability")
                gr.Markdown("Select a stakeholder tab to see the specific Question & Answer relevant to their role.")
                gr.Markdown("---")

                # STAKEHOLDER 1: CLINICAL SPECIALIST
                with gr.Tab("👨‍⚕️ Clinical Specialist"):
                    gr.Markdown(
                        "### ❓ Stakeholder Question: *'Is the model focusing on valid pathological features or image artifacts?'*"
                    )
                    with gr.Row():
                        with gr.Column():
                            stk_att_head, stk_att_body, stk_att_state, stk_att_btn = create_md_accordion(
                                "### 👁️ Intrinsic Attention (White-Box)",
                                initial_visible=False
                            )
                            with stk_att_body:
                                gr.Markdown(
                                    "**Answer:** The heatmap above shows the model's raw attention weights. If the focus is on the correct organ (e.g., lungs), the reasoning is valid.",
                                    container=True
                                )
                                stk_attention_out = gr.Image(type="pil", height=350, show_label=False, interactive=False)

                        with gr.Column():
                            stk_occ_head, stk_occ_body, stk_occ_state, stk_occ_btn = create_md_accordion(
                                "### 🧱 Occlusion Sensitivity (Causal)",
                                initial_visible=False
                            )
                            with stk_occ_body:
                                gr.Markdown(
                                    "**Answer:** Red regions indicate pixels that *caused* the diagnosis. If blocking these pixels changes the prediction, they are the 'Evidence'.",
                                    container=True
                                )
                                stk_occlusion_out = gr.Image(type="pil", height=350, show_label=False, interactive=False)

                # STAKEHOLDER 2: JUNIOR PRACTITIONER
                with gr.Tab("🎓 Junior Practitioner"):
                    gr.Markdown(
                        "### ❓ Stakeholder Question: *'What historical evidence or similar cases support this diagnosis?'*"
                    )

                    # Counterfactuals
                    stk_cf_head, stk_cf_body, stk_cf_state, stk_cf_btn = create_md_accordion(
                        "### 🔄 Visual Counterfactual (Differential Diagnosis)",
                        initial_visible=False
                    )
                    with stk_cf_body:
                        with gr.Row():
                            gr.Markdown(
                                "**Answer:** This compares the current patient to a similar historical case with the *opposite* outcome, helping you understand the decision boundary.",
                                container=True
                            )
                        with gr.Row():
                            with gr.Column(scale=1):
                                stk_cf_visual_out = gr.Image(type="pil", height=350, label="Nearest 'Opposite' Case", interactive=False)
                            with gr.Column(scale=2):
                                stk_cf_visual_report_out = gr.Markdown(container=True)

                    # RAG Retrieval Items (Moved here from separate tab)
                    stk_rag_head, stk_rag_body, stk_rag_state, stk_rag_btn = create_md_accordion(
                        "### 📚 Retrieved Case Precedents (RAG)",
                        initial_visible=False
                    )
                    with stk_rag_body:
                        stk_retrieval_log_out = gr.Textbox(show_label=False, lines=1, max_lines=1, label="Retrieval IDs")
                        gr.Markdown(
                            "**Answer:** These are the verified historical cases the AI retrieved to support its decision. Reviewing them allows you to check if the AI is citing relevant medical precedents or irrelevant data.",
                            container=True
                        )
                        stk_rag_atomic_components = []
                        # stk_rag_verdict_boxes = []  # Keep track for outputs
                        for i in range(MAX_RAG_K):
                            with gr.Row(visible=False, variant="panel") as r:
                                with gr.Column(scale=1):
                                    r_img = gr.Image(type="pil", label=f"Case #{i + 1}", height=250)
                                with gr.Column(scale=1):
                                    r_txt = gr.Markdown(height=250, container=True)
                                with gr.Column(scale=1, visible=False):
                                    r_eval = gr.Markdown()  # Judge verdict
                            stk_rag_atomic_components.extend([r, r_img, r_txt, r_eval])
                            # stk_rag_verdict_boxes.append(r_eval)

                # STAKEHOLDER 3: PATIENT
                with gr.Tab("🏥 Patient / Non-Specialist"):
                    gr.Markdown("### ❓ Stakeholder Question: *'What did the AI find in simple, understandable terms?'*")
                    with gr.Row():
                        with gr.Column():
                            stk_zs_head, stk_zs_body, stk_zs_state, stk_zs_btn = create_md_accordion(
                                "### 📊 Concept Detection (Zero-Shot)",
                                initial_visible=False
                            )
                            with stk_zs_body:
                                gr.Markdown(
                                    "**Answer:** These bars show how strongly the image activates specific medical concepts.",
                                    container=True
                                )
                                stk_concept_zs_out = gr.Markdown(container=True)

                        with gr.Column():
                            stk_nc_head, stk_nc_body, stk_nc_state, stk_nc_btn = create_md_accordion(
                                "### 🧬 Similar Case Consensus",
                                initial_visible=False
                            )
                            with stk_nc_body:
                                gr.Markdown(
                                    "**Answer:** This shows the most common diagnoses found in 50 other patients who looked similar to you.",
                                    container=True
                                )
                                stk_concept_nc_out = gr.Markdown(container=True)

                # STAKEHOLDER 4: AUDITOR
                with gr.Tab("⚖️ Auditor / Regulator"):
                    gr.Markdown(
                        "### ❓ Stakeholder Question: *'Did the system follow safety protocols and verify its data?'*")

                    stk_aud_vis_head, stk_aud_vis_body, stk_aud_vis_state, stk_aud_vis_btn = create_md_accordion(
                        "### 👁️ Traceability: Visual Audit Trail",
                        initial_visible=False
                    )
                    with stk_aud_vis_body:
                        gr.Markdown(
                            "**Answer:** These saliency maps provide an immutable audit trail, verifying that the model's decision was based on relevant anatomical features and not spurious artifacts.",
                            container=True
                        )
                        with gr.Row():
                            stk_aud_att_out = gr.Image(type="pil", height=350, label="Intrinsic Attention", interactive=False)
                            stk_aud_occ_out = gr.Image(type="pil", height=350, label="Occlusion Sensitivity", interactive=False)

                    # Reflexion
                    stk_ref_res_head, stk_ref_res_body, stk_ref_res_state, stk_ref_res_btn = create_md_accordion(
                        "### 🧠 Self-Correction Trace (Reflexion)",
                        initial_visible=False
                    )
                    with stk_ref_res_body:
                        gr.Markdown(
                            "**Answer:** This trace demonstrates the model's self-correction capability, ensuring it proactively identifies and rectifies potential hallucinations before generating the final report.",
                            container=True
                        )
                        stk_reflex_out = gr.Markdown(container=True)

                    stk_ref_rpt_head, stk_ref_rpt_body, stk_ref_rpt_state, stk_ref_rpt_btn = create_md_accordion(
                        "### ✅ Reasoning Audit (LLM Judge)",
                        initial_visible=False
                    )
                    with stk_ref_rpt_body:
                        gr.Markdown(
                            "**Answer:** Independent evaluation certifying the logic is sound.",
                            container=True
                        )
                        stk_ref_judge_out = gr.Markdown(container=True)

                    # RAG
                    stk_rag_rpt_head, stk_rag_rpt_body, stk_rag_rpt_state, stk_rag_rpt_btn = create_md_accordion(
                        "### 🛡️ Data Integrity Audit (RAG Judge)",
                        initial_visible=False
                    )
                    with stk_rag_rpt_body:
                        gr.Markdown(
                            "**Answer:** Summary of external data reliability.",
                            container=True
                        )
                        stk_rag_judge_out = gr.Markdown(container=True)
                    stk_audit_head, stk_audit_body, stk_audit_state, stk_audit_btn = create_md_accordion(
                        "### 📋 Data Integrity: Detailed Verification Logs",
                        initial_visible=False
                    )
                    with stk_audit_body:
                        gr.Markdown(
                            "**Answer:** This itemized log provides a granular verification of every external data source used, flagging any irrelevant or potentially dangerous retrieval items.",
                            container=True
                        )
                        stk_audit_rag_atomic_components = []  # New list for Auditor
                        stk_audit_verdict_boxes = []  # Track verdicts to update later

                        for i in range(MAX_RAG_K):
                            with gr.Row(visible=False, variant="panel") as r:
                                with gr.Column(scale=1):
                                    r_img = gr.Image(type="pil", label=f"Item {i + 1}", height=200)
                                with gr.Column(scale=1):
                                    r_txt = gr.Markdown(height=200, container=True)
                                with gr.Column(scale=1):
                                    r_eval = gr.Markdown(container=True)
                            stk_audit_rag_atomic_components.extend([r, r_img, r_txt, r_eval])
                            stk_audit_verdict_boxes.append(r_eval)

            with gr.Tab("Deep Analysis"):
                gr.Markdown("## 💡 Technic-Based Analysis")
                gr.Markdown("Select a technical tab to see the specific analysis.")
                gr.Markdown("---")
                with gr.Tab("XAI"):
                    gr.Markdown("## 🔬 Comprehensive Analysis")
                    gr.Markdown("---")
                    with gr.Row():
                        with gr.Column():
                            att_head, att_body, att_state, att_btn = create_md_accordion("### 👁️ Internal Attention (Intrinsic)", initial_visible=False)
                            with att_body:
                                attention_out = gr.Image(type="pil", height=350, show_label=False)
                                gr.Markdown("### Map Type: Eye-Tracking (Correlation).", container=True)
                                gr.Markdown("---")
                                gr.Markdown("""
                                **How to read this:**\n
                                **🔴 Red/Yellow:** Areas the model **"looked at"** while generating the answer.\n
                                **🔵 Blue:** Areas the model ignored.\n
                                *Note: Just because the model looked here doesn't mean it was important.*
                                """, container=True)
                        with gr.Column():
                            occ_head, occ_body, occ_state, occ_btn = create_md_accordion("### 👁️ Causal Occlusion (Perturbation)", initial_visible=False)
                            with occ_body:
                                occlusion_out = gr.Image(type="pil", height=350, show_label=False)
                                gr.Markdown("### Map Type: Stress Test (Causation).", container=True)
                                gr.Markdown("---")
                                gr.Markdown("""
                                **How to read this:**\n
                                **🔴 Red Blob:** **Critical Evidence**. Blocking this area caused the model to **fail** or change its diagnosis.\n
                                **🔵 Blue:** Irrelevant background (Safe to remove).\n
                                *This confirms exactly which part of the image drove the decision.*
                                """, container=True)

                    with gr.Row():
                        with gr.Column():
                            cf_head, cf_body, cf_state, cf_btn = create_md_accordion("### 📜️ Visual Counterfactual (Similar Opposite)", initial_visible=False)
                            with cf_body:
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        cf_visual_out = gr.Image(type="pil", height=350, label="Nearest 'Opposite' Case")
                                    with gr.Column(scale=2):
                                        gr.Markdown("### 🔄 Retrieval Counterfactual Analysis", container=True)
                                        gr.Markdown("---")
                                        gr.Markdown("""
                                        **What is this?** This search finds a *real historical patient* who looks visually similar to your image but had the **opposite diagnosis**.\n\n
                                        **How to read:**\n
                                        * **Compare Images:** If the two images look identical but have opposite labels, the diagnosis might be subtle or ambiguous.\n
                                        * **Topic Check:** We ensured both cases discuss the same specific condition.\n
                                        """, container=True)
                                        cf_visual_report_out = gr.Markdown(container=True)

                    with gr.Row():
                        with gr.Column():
                            zs_head, zs_body, zs_state, zs_btn = create_md_accordion("### 📝 Zero-Shot Concept Activation", initial_visible=False)
                            with zs_body:
                                gr.Markdown("### 🧠 Visual Concept Check (BioMedCLIP)", container=True)
                                gr.Markdown("---")
                                gr.Markdown("""
                                **What is this?** The AI scanned your image for specific visual signs.\n
                                A high percentage means the image **looks like** it matches that medical condition.\n\n
                                """, container=True)
                                concept_rpt_zs_out = gr.Markdown(container=True)
                        with gr.Column():
                            nc_head, nc_body, nc_state, nc_btn = create_md_accordion("### 📝 Neighbour-Based Concept Consensus", initial_visible=False)
                            with nc_body:
                                gr.Markdown("### 🧬 Historical Similar Cases", container=True)
                                gr.Markdown("---")
                                gr.Markdown("""
                                **What is this?** The AI found **similar past cases** in the database.\n
                                A high percentage means this condition **appeared frequently** in those past cases.\n\n
                                """, container=True)
                                concept_rpt_nc_out = gr.Markdown(container=True)

                with gr.Tab("RAG"):
                    gr.Markdown("## 📂 Visual RAG Retrieval Items & Evaluation")
                    gr.Markdown("---")
                    retrieval_log_out = gr.Textbox(show_label=False, lines=1, max_lines=1)

                    rag_item_head, rag_item_body, rag_item_state, rag_item_btn = create_md_accordion("### 🗃️ Retrieved Item List", initial_visible=False)
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
                                    r_txt = gr.Markdown(height=250, label="Metadata", container=True)

                                # Column 3: The Verdict Card (Populated by JSON)
                                with gr.Column(scale=1):
                                    r_eval = gr.Markdown(container=True)

                            rag_atomic_components.extend([r, r_img, r_txt, r_eval])
                            rag_verdict_boxes.append(r_eval)

                    rag_rpt_head, rag_rpt_body, rag_rpt_state, rag_rpt_btn = create_md_accordion("### 📋 AI Judge Assessment (Gemini 2.5 Flash) - Overall Retrieval Health", initial_visible=False)
                    with rag_rpt_body:
                        rag_judge_summary_out = gr.Markdown(container=True)

                with gr.Tab("Reflexion"):
                    gr.Markdown("## 🧠 Reflexion Process & Evaluation")
                    gr.Markdown("---")

                    ref_res_head, ref_res_body, ref_res_state, ref_res_btn = create_md_accordion("### 📝 Internal Monologue (Draft & Critique)", initial_visible=False)
                    with ref_res_body:
                        reflex_intermediate_out = gr.Markdown(container=True)

                    ref_rpt_head, ref_rpt_body, ref_rpt_state, ref_rpt_btn = create_md_accordion("### 📋 AI Judge Assessment (Gemini 2.5 Flash Lite)", initial_visible=False)
                    with ref_rpt_body:
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
        fn=lock_interface,
        inputs=None,
        outputs=all_inputs_and_buttons,
        queue=False
    ).then(
        load_engine,
        inputs=[model_select, rag_toggle, reflexion_toggle, prompt_input, k_slider, a_slider],
        outputs=[status, last_loaded_state, load_btn]
    ).then(
        fn=unlock_interface,
        inputs=None,
        outputs=all_inputs_and_buttons
    )

    run_btn.click(
        fn=lock_interface,
        inputs=None,
        outputs=all_inputs_and_buttons,
        queue=False
    ).then(
        run_database_inference,
        inputs=[dataset_input, idx_input, model_select, rag_toggle, reflexion_toggle, prompt_input, k_slider, a_slider],
        outputs=[
            status, img_out, question, gt, prediction, reflex_intermediate_out, retrieval_log_out,
            attention_out, occlusion_out, concept_rpt_zs_out, concept_rpt_nc_out, cf_visual_out, cf_visual_report_out,
            reflex_judge_report_out, rag_judge_summary_out, prediction_state,
            *rag_atomic_components,
            ref_res_body, ref_res_state, ref_res_btn,
            rag_item_body, rag_item_state, rag_item_btn,
            att_body, att_state, att_btn,
            occ_body, occ_state, occ_btn,
            zs_body, zs_state, zs_btn,
            nc_body, nc_state, nc_btn,
            cf_body, cf_state, cf_btn,
            ref_rpt_body, ref_rpt_state, ref_rpt_btn,
            rag_rpt_body, rag_rpt_state, rag_rpt_btn,

            stk_reflex_out, stk_retrieval_log_out,
            stk_attention_out, stk_occlusion_out, stk_aud_att_out, stk_aud_occ_out,
            stk_concept_zs_out, stk_concept_nc_out, stk_cf_visual_out, stk_cf_visual_report_out,
            stk_ref_judge_out, stk_rag_judge_out,
            *stk_rag_atomic_components,
            stk_ref_res_body, stk_ref_res_state, stk_ref_res_btn,
            stk_rag_body, stk_rag_state, stk_rag_btn,
            *stk_audit_rag_atomic_components,
            stk_audit_body, stk_audit_state, stk_audit_btn,
            stk_att_body, stk_att_state, stk_att_btn,
            stk_occ_body, stk_occ_state, stk_occ_btn,
            stk_aud_vis_body, stk_aud_vis_state, stk_aud_vis_btn,
            stk_zs_body, stk_zs_state, stk_zs_btn,
            stk_nc_body, stk_nc_state, stk_nc_btn,
            stk_cf_body, stk_cf_state, stk_cf_btn,
            stk_ref_rpt_body, stk_ref_rpt_state, stk_ref_rpt_btn,
            stk_rag_rpt_body, stk_rag_rpt_state, stk_rag_rpt_btn
        ]
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
        run_xai,
        inputs=[prediction_state],
        outputs=[
            attention_out, occlusion_out, concept_rpt_zs_out, concept_rpt_nc_out, cf_visual_out, cf_visual_report_out,
            att_body, att_state, att_btn,
            occ_body, occ_state, occ_btn,
            zs_body, zs_state, zs_btn,
            nc_body, nc_state, nc_btn,
            cf_body, cf_state, cf_btn,

            # Stakeholder Analysis
            stk_attention_out, stk_occlusion_out, stk_aud_att_out, stk_aud_occ_out,
            stk_concept_zs_out, stk_concept_nc_out, stk_cf_visual_out, stk_cf_visual_report_out,
            stk_att_body, stk_att_state, stk_att_btn,
            stk_occ_body, stk_occ_state, stk_occ_btn,
            stk_aud_vis_body, stk_aud_vis_state, stk_aud_vis_btn,
            stk_zs_body, stk_zs_state, stk_zs_btn,
            stk_nc_body, stk_nc_state, stk_nc_btn,
            stk_cf_body, stk_cf_state, stk_cf_btn
        ]
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
        run_llm_judge,
        inputs=[prediction_state],
        outputs=[
            reflex_judge_report_out, rag_judge_summary_out, *rag_verdict_boxes,
            ref_rpt_body, ref_rpt_state, ref_rpt_btn,
            rag_rpt_body, rag_rpt_state, rag_rpt_btn,

            # Stakeholder Judge Outputs
            stk_ref_judge_out, stk_rag_judge_out, *stk_audit_verdict_boxes,
            stk_ref_rpt_body, stk_ref_rpt_state, stk_ref_rpt_btn,
            stk_rag_rpt_body, stk_rag_rpt_state, stk_rag_rpt_btn,
            stk_audit_body, stk_audit_state, stk_audit_btn
        ]
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
        outputs=[
            status, img_out, question, gt, prediction, reflex_intermediate_out, retrieval_log_out,
            attention_out, occlusion_out, concept_rpt_zs_out, concept_rpt_nc_out, cf_visual_out, cf_visual_report_out,
            reflex_judge_report_out, rag_judge_summary_out, prediction_state, *rag_atomic_components,
            ref_res_body, ref_res_state, ref_res_btn,
            rag_item_body, rag_item_state, rag_item_btn,
            att_body, att_state, att_btn,
            occ_body, occ_state, occ_btn,
            zs_body, zs_state, zs_btn,
            nc_body, nc_state, nc_btn,
            cf_body, cf_state, cf_btn,
            ref_rpt_body, ref_rpt_state, ref_rpt_btn,
            rag_rpt_body, rag_rpt_state, rag_rpt_btn,

            # STAKEHOLDER RESET OUTPUTS
            stk_reflex_out, stk_retrieval_log_out, *stk_rag_atomic_components, *stk_audit_rag_atomic_components,
            stk_ref_res_body, stk_ref_res_state, stk_ref_res_btn,
            stk_rag_body, stk_rag_state, stk_rag_btn,
            stk_attention_out, stk_occlusion_out, stk_aud_att_out, stk_aud_occ_out,
            stk_concept_zs_out, stk_concept_nc_out, stk_cf_visual_out, stk_cf_visual_report_out,
            stk_att_body, stk_att_state, stk_att_btn,
            stk_occ_body, stk_occ_state, stk_occ_btn,
            stk_aud_vis_body, stk_aud_vis_state, stk_aud_vis_btn,
            stk_zs_body, stk_zs_state, stk_zs_btn,
            stk_nc_body, stk_nc_state, stk_nc_btn,
            stk_cf_body, stk_cf_state, stk_cf_btn,
            stk_ref_judge_out, stk_rag_judge_out,
            stk_ref_rpt_body, stk_ref_rpt_state, stk_ref_rpt_btn,
            stk_rag_rpt_body, stk_rag_rpt_state, stk_rag_rpt_btn,
            stk_audit_body, stk_audit_state, stk_audit_btn
        ]
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

    def update_index_limits(dataset_name):
        # 1. Get dataset info safely
        ds_info = DATASET_OPTIONS.get(dataset_name, {})

        # 2. Get size (Default to 100 if missing to prevent crash)
        total_size = ds_info.get("size", 100)
        max_index = total_size - 1

        # 3. Return the UI update
        return gr.update(
            label=f"Sample Index (0 - {max_index})",
            value=0,
            interactive=True
        )

    dataset_input.change(
        fn=update_index_limits,
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
    demo.launch(favicon_path="assets/GENMED_XAI.png", server_port=7860)
