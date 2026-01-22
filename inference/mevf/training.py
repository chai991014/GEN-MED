import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.weight_norm import weight_norm
from datasets import load_dataset
import numpy as np
import pickle
import random
import time
from datetime import datetime
from inference.utils import setup_logger


# =============================================================================
# 1. CORE MODULES (SAN & BAN)
# =============================================================================

class StackedAttention(nn.Module):
    def __init__(self, num_stacks, v_dim, q_dim, hid_dim, dropout):
        super(StackedAttention, self).__init__()
        self.num_stacks = num_stacks
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        # Stack 1 Layers
        self.fc_v = nn.Linear(v_dim, hid_dim, bias=False)  # Projects Visual (128 -> 1024)
        self.fc_q = nn.Linear(q_dim, hid_dim, bias=True)  # Projects Question (1024 -> 1024)
        self.fc_h = nn.Linear(hid_dim, 1, bias=True)  # Attention Score

        # Subsequent Stack Layers
        self.layers = nn.ModuleList()
        for _ in range(num_stacks - 1):
            self.layers.append(nn.Linear(hid_dim, hid_dim, bias=True))  # 0: Query Update
            self.layers.append(nn.Linear(v_dim, hid_dim, bias=False))  # 1: Visual Projection (128 -> 1024)
            self.layers.append(nn.Linear(hid_dim, 1, bias=True))  # 2: Attention Score

    def forward(self, v, q):
        """
        v: [Batch, N, 128]
        q: [Batch, 1024]
        """
        B, N, _ = v.size()

        # --- Stack 1 ---
        # 1. Project features to hidden dimension (1024)
        q_proj = self.fc_q(q).unsqueeze(1).expand(B, N, -1)  # [B, N, 1024]
        v_proj = self.fc_v(v)  # [B, N, 1024]

        # 2. Attention Distribution
        h = self.tanh(q_proj + v_proj)
        h_drop = self.dropout(h)
        attn_score = self.fc_h(h_drop)  # [B, N, 1]
        attn_weight = self.softmax(attn_score)

        # 3. Weighted Sum (FIX: Use v_proj, NOT v)
        # We must use the projected visual features (1024 dim) so we can add them to q (1024 dim)
        v_weighted = (v_proj * attn_weight).sum(dim=1)  # [B, 1024]

        # 4. Update Query
        u = q + v_weighted  # [B, 1024] + [B, 1024] -> OK

        # --- Subsequent Stacks ---
        qs = [u]

        for i in range(self.num_stacks - 1):
            # Map updated query to hidden space
            u_proj = self.layers[3 * i + 0](qs[-1]).unsqueeze(1).expand(B, N, -1)

            # Re-project raw visual features to hidden space
            v_new_proj = self.layers[3 * i + 1](v)  # [B, N, 1024]

            h = self.tanh(u_proj + v_new_proj)
            h_drop = self.dropout(h)
            attn_score = self.layers[3 * i + 2](h_drop)
            attn_weight = self.softmax(attn_score)

            # FIX: Use v_new_proj (1024), NOT v (128)
            v_weighted = (v_new_proj * attn_weight).sum(dim=1)  # [B, 1024]

            # Refine query again
            u_next = qs[-1] + v_weighted
            qs.append(u_next)

        return qs[-1]


class BCNet(nn.Module):
    """Bilinear Connect Network (Low-rank bilinear pooling)"""

    def __init__(self, v_dim, q_dim, h_dim, h_out, k=3, dropout=[.2, .5]):
        super(BCNet, self).__init__()
        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = weight_norm(nn.Linear(v_dim, h_dim * k, bias=False), dim=None)
        self.q_net = weight_norm(nn.Linear(q_dim, h_dim * k, bias=False), dim=None)
        self.dropout = nn.Dropout(dropout[1])

        if 1 < k:
            self.p_net = nn.AvgPool1d(k, stride=k)

        if h_out is not None:
            self.h_net = weight_norm(nn.Linear(h_dim, h_out), dim=None)

    def forward(self, v, q):
        # v: [Batch, N, v_dim]
        # q: [Batch, L, q_dim]

        # 1. Handle Input Dimensions
        if v.dim() == 2:
            v = v.unsqueeze(1)
        if q.dim() == 2:
            q = q.unsqueeze(1)

        # 2. Linear Projection
        v_ = self.dropout(self.v_net(v))
        q_ = self.dropout(self.q_net(q))

        # 3. Bilinear Interaction
        v_ = v_.unsqueeze(2)  # [B, N, 1, H*K]
        q_ = q_.unsqueeze(1)  # [B, 1, L, H*K]
        logits = v_ * q_  # [B, N, L, H*K]

        # 4. Pooling
        if 1 < self.k:
            b, n, l, hk = logits.size()
            logits = logits.view(-1, 1, hk)
            logits = self.p_net(logits)
            logits = logits.view(b, n, l, -1)

        # 5. Output Projection
        if self.h_out is not None:
            logits = self.h_net(logits)  # [B, N, L, h_out]

        return logits


class BiAttention(nn.Module):
    """Bilinear Attention Module"""

    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2, .5]):
        super(BiAttention, self).__init__()
        self.glimpse = glimpse
        # No weight_norm wrapper here (BCNet has it internally)
        self.logits = BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3)

    def forward(self, v, q, v_mask=True, q_mask=None):
        p, logits = self.forward_all(v, q, v_mask, q_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True, q_mask=None):
        v_num = v.size(1)
        q_num = q.size(1)

        # Get Logits: [Batch, N, L, Glimpse]
        logits = self.logits(v, q)

        # Permute to [Batch, Glimpse, N, L] for Softmax
        logits = logits.permute(0, 3, 1, 2).contiguous()

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, -float('inf'))

        if q_mask is not None:
            # q_mask is [B, L]. Expand to [B, Glimpse, N, L]
            q_mask_ex = q_mask.unsqueeze(1).unsqueeze(2).expand(logits.size())
            logits.data.masked_fill_(q_mask_ex, -float('inf'))

        p = F.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits


class FCNet(nn.Module):
    """Simple Fully Connected Network with ReLU and Dropout"""

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())

        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


# =============================================================================
# 2. FEATURE EXTRACTORS (MAML & AE) - Must match pretrained weights exactly
# =============================================================================

class SimpleCNN(nn.Module):
    """
    The MAML backbone.
    Crucial: Weights loading logic must match the pickle format of 'pretrained_maml.weights'.
    """

    def __init__(self, weight_path):
        super(SimpleCNN, self).__init__()

        # Load pickle weights
        print(f"Loading MAML weights from {weight_path}...")
        try:
            with open(weight_path, 'rb') as f:
                weights = pickle.load(f, encoding='latin1')  # 'latin1' often needed for old pickles
        except Exception as e:
            print(f"Error loading pickle: {e}. Ensure file exists.")
            weights = {}  # Will crash later if empty, but handled for now

        self.conv1 = self.init_conv(1, 64, weights.get('conv1'), weights.get('b1'))
        # self.conv1_bn = nn.BatchNorm2d(64, momentum=0.05)
        self.conv2 = self.init_conv(64, 64, weights.get('conv2'), weights.get('b2'))
        # self.conv2_bn = nn.BatchNorm2d(64, momentum=0.05)
        self.conv3 = self.init_conv(64, 64, weights.get('conv3'), weights.get('b3'))
        # self.conv3_bn = nn.BatchNorm2d(64, momentum=0.05)
        self.conv4 = self.init_conv(64, 64, weights.get('conv4'), weights.get('b4'))
        # self.conv4_bn = nn.BatchNorm2d(64, momentum=0.05)

    def init_conv(self, inp, out, weight, bias):
        conv = nn.Conv2d(inp, out, 3, 2, 1, bias=True)
        if weight is not None and bias is not None:
            # Convert numpy (H, W, In, Out) -> Torch (Out, In, H, W)
            # The original repo uses a transpose: [3, 2, 0, 1]
            weight_torch = torch.from_numpy(np.transpose(weight, [3, 2, 0, 1])).float()
            bias_torch = torch.from_numpy(bias).float()
            conv.weight.data.copy_(weight_torch)
            conv.bias.data.copy_(bias_torch)
        return conv

    def forward(self, x):
        out = F.relu(self.conv1(x))
        # out = self.conv1_bn(out)
        out = F.relu(self.conv2(out))
        # out = self.conv2_bn(out)
        out = F.relu(self.conv3(out))
        # out = self.conv3_bn(out)
        out = F.relu(self.conv4(out))
        # out = self.conv4_bn(out)
        # Global pooling (approximate based on original repo logic)
        out = out.view(out.size(0), 64, -1)
        return torch.mean(out, 2)


class AutoEncoderModel(nn.Module):
    """The AE Backbone"""

    def __init__(self):
        super(AutoEncoderModel, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, padding=1, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 32, padding=1, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 16, padding=1, kernel_size=3)
        # Decoder (for reconstruction loss if needed)
        self.tran_conv1 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.tran_conv2 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward_pass(self, x):
        out = F.relu(self.conv1(x))
        out = self.max_pool1(out)
        out = F.relu(self.conv2(out))
        out = self.max_pool2(out)
        out = F.relu(self.conv3(out))
        return out  # returns spatial features


# =============================================================================
# 3. VISUAL ENCODER (MEVF)
# =============================================================================

class MEVF(nn.Module):
    """
    Mixture of Enhanced Visual Features (MEVF).
    Combines features from a MAML backbone (meta-learning) and an AutoEncoder.
    """

    def __init__(self, args):
        super(MEVF, self).__init__()

        # 1. MAML Backbone (SimpleCNN)
        self.maml = SimpleCNN(args.maml_path)

        # 2. AutoEncoder Backbone
        self.ae = AutoEncoderModel()
        self.ae.load_state_dict(torch.load(args.ae_path))

        # 3. AE Feature Projector
        # AE outputs 16 channels x 32 x 32 (flattened = 16384)
        # We project this to 64 dimensions to match MAML's output
        self.ae_convert = nn.Linear(16384, 64)

    def forward(self, maml_img, ae_img):
        """
        Input:
            maml_img: [B, 1, 84, 84]
            ae_img:   [B, 1, 128, 128]
        Output:
            v_emb:    [B, 1, 128] (Combined Feature Vector)
        """
        # --- MAML Branch ---
        # Output: [B, 64] -> unsqueeze to [B, 1, 64]
        maml_feat = self.maml(maml_img).unsqueeze(1)

        # --- AE Branch ---
        # Output: [B, 16, 32, 32]
        ae_feat_map = self.ae.forward_pass(ae_img)
        ae_feat = ae_feat_map.view(ae_feat_map.size(0), -1)  # Flatten
        ae_feat = self.ae_convert(ae_feat).unsqueeze(1)  # [B, 1, 64]

        # --- Concatenation ---
        # Combine MAML and AE features along the feature dimension
        # Result: [B, 1, 64+64] = [B, 1, 128]
        v_emb = torch.cat((maml_feat, ae_feat), 2)

        # --- L2 NORMALIZATION ---
        v_emb = F.normalize(v_emb, p=2, dim=2)

        return v_emb


# =============================================================================
# 4. REASONING MODEL (SAN & BAN)
# =============================================================================

class SAN(nn.Module):
    def __init__(self, dataset, args):
        super(SAN, self).__init__()
        self.args = args
        self.v_dim = 128  # 64 (MAML) + 64 (AE)
        self.q_dim = args.num_hid  # LSTM hidden size

        # 1. Text Encoder (LSTM)
        # Note: SAN uses the final hidden state, not the sequence
        self.w_emb = nn.Embedding(dataset.vocab_size, 300, padding_idx=0)
        self.q_rnn = nn.LSTM(input_size=300, hidden_size=self.q_dim, num_layers=1, batch_first=True)

        # 2. Stacked Attention
        # num_stacks is usually 2. We can re-use 'gamma' from args as 'num_stacks'
        # or add a new arg. We'll use args.gamma for convenience.
        self.san = StackedAttention(
            num_stacks=args.gamma,
            v_dim=self.v_dim,
            q_dim=self.q_dim,
            hid_dim=1024,  # Internal attention dimension
            dropout=0.5
        )

        # 3. Classifier
        # Input is the refined query vector (size q_dim)
        self.classifier = FCNet([self.q_dim, dataset.num_ans_candidates], dropout=0.5)

    def forward(self, v_emb, q_tokens):
        """
        v_emb: [B, 1, 128] (Visual Features)
        q_tokens: [B, L] (Question Indices)
        """
        # 1. Text Encoding
        w_emb = self.w_emb(q_tokens)  # [B, L, 300]
        self.q_rnn.flatten_parameters()
        _, (h_n, _) = self.q_rnn(w_emb)  # Take final hidden state
        q_emb = h_n[-1]  # [B, q_dim]

        # 2. Stacked Attention Reasoning
        # v_emb is [B, 1, 128]. SAN handles this fine (N=1).
        final_query = self.san(v_emb, q_emb)

        # 3. Classification
        logits = self.classifier(final_query)

        return logits


class BAN(nn.Module):
    def __init__(self, dataset, args):
        super(BAN, self).__init__()
        self.args = args
        self.v_dim = 128  # 64 (MAML) + 64 (AE)

        # Text Encoder (LSTM)
        self.w_emb = nn.Embedding(dataset.vocab_size, 300, padding_idx=0)
        self.q_rnn = nn.LSTM(input_size=300, hidden_size=args.num_hid, num_layers=1, batch_first=True)

        # Attention (BiAttention)
        self.v_att = BiAttention(self.v_dim, args.num_hid, args.num_hid, args.gamma)

        # Bilinear Residual Network (The reasoning steps)
        self.b_net = nn.ModuleList([
            BCNet(self.v_dim, args.num_hid, args.num_hid, None, k=3)
            for _ in range(args.gamma)
        ])
        self.q_prj = nn.ModuleList([
            FCNet([args.num_hid, args.num_hid], '', .2)
            for _ in range(args.gamma)
        ])

        # Classifier
        self.classifier = FCNet([args.num_hid, args.num_hid * 2, dataset.num_ans_candidates], dropout=0.5)

    def forward(self, v_emb, q_tokens):
        """
        Input:
            v_emb:    [B, 1, 128] (From MEVF)
            q_tokens: [B, L]
        """
        # 1. Create Mask (True where token is padding/0)
        q_mask = (q_tokens == 0)

        # 2. Text Feature Extraction
        w_emb = self.w_emb(q_tokens)  # [B, L, 300]
        self.q_rnn.flatten_parameters()
        q_out, _ = self.q_rnn(w_emb)
        q_emb = q_out  # [B, L, hid]

        # 3. BAN Loop
        b_emb = [0] * self.args.gamma
        att, logits = self.v_att.forward_all(v_emb, q_emb, v_mask=True, q_mask=q_mask)  # Calculate Attention map

        for g in range(self.args.gamma):
            # Bilinear Connect: Mix Visual (v_emb) and Question (q_emb)
            # b_emb[g] shape usually: [B, 1, L, hid] (since v has 1 object)
            b_emb[g] = self.b_net[g](v_emb, q_emb)

            # Apply Attention: Weighted sum over the Question sequence length
            # att shape: [B, gamma, 1, L] -> slice for current glimpse -> [B, 1, L]
            atten = att[:, g, :, :]

            # Apply weights: b_emb * attention
            weighted_feat = (b_emb[g] * atten.unsqueeze(-1)).sum(2).sum(1)  # Sum over L and V(1)

            # Residual connection for next hop
            q_emb_new = self.q_prj[g](weighted_feat.unsqueeze(1))
            q_emb = q_emb + q_emb_new

        # 4. Final Classification
        # Mask the padding positions before summing so we don't sum up noise
        q_emb_masked = q_emb.masked_fill(q_mask.unsqueeze(-1), 0.0)
        q_final = q_emb_masked.sum(1)

        logits = self.classifier(q_final)

        return logits


# =============================================================================
# 5. MODEL WRAPPER
# =============================================================================

class VQAModel(nn.Module):
    def __init__(self, dataset, args, reasoning_module='BAN'):
        super(VQAModel, self).__init__()

        # 1. Visual Module (Fixed)
        self.mevf = MEVF(args)

        # 2. Reasoning Module (Swappable)
        if reasoning_module == 'BAN':
            self.reasoning = BAN(dataset, args)
        elif reasoning_module == 'SAN':
            self.reasoning = SAN(dataset, args)
            pass
        else:
            raise ValueError(f"Unknown reasoning module: {reasoning_module}")

    def forward(self, maml_img, ae_img, q):
        # Step 1: Get Visual Features
        v_feat = self.mevf(maml_img, ae_img)

        # Step 2: Reason and Answer
        logits = self.reasoning(v_feat, q)

        return logits


# =============================================================================
# 6. DATASET & UTILS
# =============================================================================

class Dictionary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = ["<pad>", "<unk>"]

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower().replace('?', '').replace('.', '').replace(',', '')
        words = sentence.split()
        tokens = []
        for w in words:
            if add_word:
                if w not in self.word2idx:
                    self.word2idx[w] = len(self.idx2word)
                    self.idx2word.append(w)
                tokens.append(self.word2idx[w])
            else:
                tokens.append(self.word2idx.get(w, 1))  # 1 is <unk>
        return tokens

    @property
    def ntoken(self):
        return len(self.idx2word)


class VQARADDataset(Dataset):
    def __init__(self, split="train", dictionary=None, ans2label=None, data_list=None):
        if data_list is not None:
            self.hf_data = data_list
        else:
            self.hf_data = load_dataset("flaviagiammarino/vqa-rad", split="train" if split == "train" else "test")

        # Build Vocab
        if dictionary is None:
            self.dictionary = Dictionary()
            for item in self.hf_data: self.dictionary.tokenize(item['question'], True)
        else:
            self.dictionary = dictionary

        self.vocab_size = self.dictionary.ntoken

        # Build Answers
        if ans2label is None:
            self.ans2label = {}
            self.label2ans = []
            for item in self.hf_data:
                ans = str(item['answer']).lower()
                if ans not in self.ans2label:
                    self.ans2label[ans] = len(self.label2ans)
                    self.label2ans.append(ans)
        else:
            self.ans2label = ans2label

        self.num_ans_candidates = len(self.ans2label)

    def __len__(self):
        return len(self.hf_data)

    def __getitem__(self, idx):
        item = self.hf_data[idx]

        # Images (Keep as is)
        img = item['image'].convert('RGB')
        maml_img = np.array(img.resize((84, 84)).convert('L')).astype('float32') / 255.0
        maml_tensor = torch.from_numpy(maml_img).unsqueeze(0)
        ae_img = np.array(img.resize((128, 128)).convert('L')).astype('float32') / 255.0
        ae_tensor = torch.from_numpy(ae_img).unsqueeze(0)

        # Question (Keep as is)
        tokens = self.dictionary.tokenize(item['question'], False)
        if len(tokens) < 12:
            tokens += [0] * (12 - len(tokens))
        else:
            tokens = tokens[:12]
        q_tensor = torch.tensor(tokens).long()

        ans_str = str(item['answer']).lower()
        if ans_str in self.ans2label:
            ans_idx = self.ans2label[ans_str]
            target = torch.tensor(ans_idx).long()
        else:
            target = torch.tensor(-100).long()

        return maml_tensor, ae_tensor, q_tensor, target


# =============================================================================
# 7. TRAINING SCRIPT
# =============================================================================

class Args:
    maml_path = 'pretrained_maml.weights'  # Ensure this file exists
    ae_path = 'pretrained_ae.pth'  # Ensure this file exists
    reasoning_module = "BAN"
    num_hid = 1024
    gamma = 2
    lr = 5e-4
    epochs = 50
    batch_size = 16


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = Args()

    ACCUMULATION_STEPS = 4

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./model/MEVF_{args.reasoning_module}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    base_name = f"train_MEVF_{args.reasoning_module}_{timestamp}"
    setup_logger(save_dir, base_name)

    print("Preparing Data...")
    full_hf_dataset = load_dataset("flaviagiammarino/vqa-rad", split="train")

    dataset_list = list(full_hf_dataset)
    random.seed(42)
    random.shuffle(dataset_list)

    split_idx = int(len(dataset_list) * 0.9)
    train_subset = dataset_list[:split_idx]
    val_subset = dataset_list[split_idx:]

    print(f"✅ Splits aligned: Train={len(train_subset)}, Val={len(val_subset)}")

    train_dset = VQARADDataset(data_list=train_subset)
    val_dset = VQARADDataset(dictionary=train_dset.dictionary, ans2label=train_dset.ans2label, data_list=val_subset)

    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)

    print(f"Initializing Model (Vocab: {train_dset.vocab_size}, Ans: {train_dset.num_ans_candidates})...")
    model = VQAModel(train_dset, args, reasoning_module=Args.reasoning_module).to(device)

    # 1. Separate the parameters into two groups
    # Group A: The delicate visual backbone (MAML + AE)
    backbone_ids = list(map(id, model.mevf.parameters()))

    # Group B: Everything else (BAN, Classifier, Embeddings) - The "Head"
    head_params = list(filter(lambda p: id(p) not in backbone_ids, model.parameters()))

    # 2. Define Optimizer with different Learning Rates
    optimizer = torch.optim.Adamax([
        {'params': model.mevf.parameters(), 'lr': 0.0},  # Very slow updates for Vision (Safe)
        {'params': head_params, 'lr': args.lr}  # Normal updates for Reasoning
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    start_time = time.time()
    best_acc = 0.0

    print("Start Training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        for i, (maml_img, ae_img, q, label) in enumerate(train_loader):
            maml_img = maml_img.to(device)
            ae_img = ae_img.to(device)
            q = q.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            preds = model(maml_img, ae_img, q)
            loss = criterion(preds, label)
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            pred_indices = torch.argmax(preds, dim=1)
            correct += (pred_indices == label).sum().item()
            total_samples += label.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total_samples

        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for _, (maml_img, ae_img, q, label) in enumerate(val_loader):
                maml_img = maml_img.to(device)
                ae_img = ae_img.to(device)
                q = q.to(device)
                label = label.to(device)

                preds = model(maml_img, ae_img, q)

                # Calculate Validation Loss
                loss = criterion(preds, label)
                val_running_loss += loss.item()

                # Calculate accuracy only (ignore loss for speed)
                pred_indices = torch.argmax(preds, dim=1)
                val_correct += (pred_indices == label).sum().item()
                val_total += label.size(0)

        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        scheduler.step(avg_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            best_save_path = os.path.join(save_dir, f"{Args.reasoning_module}_best_model.pth")
            torch.save(model.state_dict(), best_save_path)
            print(f">>> New Best Accuracy! Saved to {best_save_path}")

        model.train()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Time: {total_time:.2f} sec")


if __name__ == "__main__":
    train()
