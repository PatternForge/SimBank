
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from SimBank.models.base import select_features


def emb_size(card):

    return int(min(50, round(card ** 0.5) + 1))


class TabularDataset(Dataset):

    def __init__(self, X_df, y_arr, cat_idx, num_idx):
        self.X = X_df.values.astype(np.float32)
        self.y = y_arr.astype(np.float32)
        self.cat_idx = cat_idx
        self.num_idx = num_idx

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        row = self.X[idx]
        cats = torch.tensor(row[self.cat_idx], dtype=torch.long)
        nums = torch.tensor(row[self.num_idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return cats, nums, label


class TabularNet(nn.Module):

    def __init__(self, cat_cardinalities, num_dim, hidden=[128, 64], p=0.2):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.Embedding(card, emb_size(card)) for card in cat_cardinalities])
        emb_total = sum([emb_size(card) for card in cat_cardinalities])
        self.bn_num = nn.BatchNorm1d(num_dim) if num_dim > 0 else nn.Identity()
        layers = []
        in_dim = emb_total + num_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, cats, nums):
        if len(self.emb_layers) > 0:
            embs = [emb(cats[:, i]) for i, emb in enumerate(self.emb_layers)]
            x_cat = torch.cat(embs, dim=1)
        else:
            x_cat = torch.empty((cats.size(0), 0), device=cats.device)
        x_num = self.bn_num(nums) if nums.numel() > 0 else nums
        x = torch.cat([x_cat, x_num], dim=1)
        logits = self.mlp(x).squeeze(1)
        return logits


class BCEWithLogitsLossSmooth(nn.Module):

    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


def run(df, sample_frac=0.25):

    df = df.sample(frac=sample_frac, random_state=0).reset_index(drop=True) if 0 < sample_frac < 1.0 else df
    df = df.copy()
    if "StageOfECL" not in df.columns:
        raise ValueError("StageOfECL not found")
    df["Stage3Flag"] = (df["StageOfECL"] == 3).astype(int)

    X, y, feats, _ = select_features(df, target="Stage3Flag")

    name_patterns = {"ARREAR", "STAGE", "ECL", "IMPAIR", "DEFAULT", "BUCKET", "LOSS", "PROVISION", "DELINQU", "PASTDUE", "DPD"}
    feats = [c for c in feats if c not in {"AccountID", "NumberID", "CustomerID", "DateOfPortfolio"}]
    feats = [c for c in feats if not any(p in c.upper() for p in name_patterns)]

    num_cols = df[feats].select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df[feats].select_dtypes(exclude=["number"]).columns.tolist()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))

        df[cat_cols] = df[cat_cols].fillna(-1)
        df[cat_cols] = (df[cat_cols].astype("int64") + 1).clip(lower=0)

    X = df[feats].copy()
    y = df["Stage3Flag"].values

    if len(np.unique(y)) < 2:
        raise ValueError("Stage3Flag has a single class")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)

    cat_idx = [i for i, c in enumerate(feats) if c in cat_cols]
    num_idx = [i for i, c in enumerate(feats) if c not in cat_cols]

    cat_cardinalities = [int(X.iloc[:, i].max()) + 1 for i in cat_idx]
    num_dim = len(num_idx)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl = DataLoader(TabularDataset(X_tr, y_tr, cat_idx, num_idx), batch_size=4096, shuffle=True, num_workers=0)
    test_dl = DataLoader(TabularDataset(X_te, y_te, cat_idx, num_idx), batch_size=8192, shuffle=False, num_workers=0)

    model = TabularNet(cat_cardinalities, num_dim, hidden=[128, 64], p=0.2).to(device)
    criterion = BCEWithLogitsLossSmooth(smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-4, max_lr=3e-3, step_size_up=200, mode="triangular2", cycle_momentum=False)

    EPOCHS = 10
    patience = 1
    best_auc = 0.0
    epochs_no_improve = 0
    train_losses = []
    valid_aucs = []

    def evaluate():
        model.eval()
        logits_list, y_list = [], []
        with torch.no_grad():
            for cats, nums, labels in test_dl:
                cats, nums, labels = cats.to(device), nums.to(device), labels.to(device)
                probs = torch.sigmoid(model(cats, nums)).cpu().numpy()
                logits_list.append(probs)
                y_list.append(labels.cpu().numpy())
        y_pred = np.concatenate(logits_list)
        y_true = np.concatenate(logits_list) * 0 + np.concatenate(y_list)
        mask = ~np.isnan(y_pred)
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        auc_roc = roc_auc_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        return {"auc": auc_roc, "pr_auc": pr_auc}, y_true, y_pred

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for cats, nums, labels in train_dl:
            cats, nums, labels = cats.to(device), nums.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(cats, nums), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        m, y_true, y_pred = evaluate()
        train_losses.append(epoch_loss)
        valid_aucs.append(m["auc"])
        print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | AUC: {m['auc']:.4f} | PR-AUC: {m['pr_auc']:.4f}")

    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_aucs, label="Valid AUC")
    plt.title("Training Progress – Stage 3 (NN)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR (AUC={auc(recall, precision):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall – Stage 3 (NN)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return model, m
