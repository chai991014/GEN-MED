import torch
import numpy as np
from inference.mevf.training import VQARADDataset, Args, VQAModel


class MEVFAdapter:
    def __init__(self, model_path, maml_path, ae_path, reasoning_model, device="cpu"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = Args()
        self.args.maml_path = maml_path
        self.args.ae_path = ae_path
        self.reasoning_model = reasoning_model

        print(f"Loading MEVF Adapter on {self.device}...")

        # 1. Build Vocabulary & Answer Mapping (Must match training!)
        # We load the training set momentarily just to rebuild the dictionary
        print("   • Rebuilding vocabulary from training set...")
        train_dset = VQARADDataset('train')
        self.dictionary = train_dset.dictionary
        self.ans2label = train_dset.ans2label
        self.label2ans = train_dset.label2ans
        self.vocab_size = train_dset.vocab_size
        self.num_ans_candidates = train_dset.num_ans_candidates

        # 2. Initialize Model
        # Using the wrapper VQAModel we defined earlier
        # Ensure this matches exactly what you trained
        self.model = VQAModel(train_dset, self.args, reasoning_module=self.reasoning_model)

        # 3. Load Weights
        print(f"   • Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image):
        """
        Converts PIL image to the specific MAML (84x84) and AE (128x128) tensors.
        """
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # MAML: 84x84 Grayscale
        maml_img = np.array(image.resize((84, 84)).convert('L')).astype('float32') / 255.0
        maml_tensor = torch.from_numpy(maml_img).unsqueeze(0).unsqueeze(0)  # [B, 1, 84, 84]

        # AE: 128x128 Grayscale
        ae_img = np.array(image.resize((128, 128)).convert('L')).astype('float32') / 255.0
        ae_tensor = torch.from_numpy(ae_img).unsqueeze(0).unsqueeze(0)  # [B, 1, 128, 128]

        return maml_tensor.to(self.device), ae_tensor.to(self.device)

    def preprocess_question(self, question):
        """
        Tokenizes the question using the fixed dictionary.
        """
        tokens = self.dictionary.tokenize(question, False)
        # Pad/Truncate to 12
        max_len = 12
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]

        q_tensor = torch.tensor(tokens).long().unsqueeze(0)  # [B, L]
        return q_tensor.to(self.device)

    def generate(self, image, question):
        """
        Standard interface to match LLM adapters.
        Returns: String (the predicted answer)
        """
        maml_in, ae_in = self.preprocess_image(image)
        q_in = self.preprocess_question(question)

        with torch.no_grad():
            logits = self.model(maml_in, ae_in, q_in)
            pred_idx = torch.argmax(logits, dim=1).item()

        return {
            "prediction": self.label2ans[pred_idx],
            "reflexion_draft": "",
            "reflexion_critique": "",
            "retrieved_ids": [],
            "retrieved_scores": []
        }

    def judge_answer(self, image, question, raw_pred):
        # MEVF is a classifier, so raw_pred is already the best class.
        # No separate judgement needed, just return the raw prediction.
        return raw_pred
