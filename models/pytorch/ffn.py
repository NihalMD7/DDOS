"""Feed-forward DDoS classifier (PyTorch parity with `models/ffn_tf.py`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[2]
ART = PROJECT / "artifacts"
REPORTS = PROJECT / "reports"
ART.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


class FFN(nn.Module):
    def __init__(self, in_dim: int, n_cls: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(epochs: int, seed: int) -> None:
    torch.manual_seed(seed); np.random.seed(seed)
    p = ART / "cicddos2019_clean.parquet"
    if not p.exists():
        raise SystemExit("run preprocessing/build_dataset.py first")
    df = pd.read_parquet(p)
    feature_cols = [c for c in df.columns if c != "Label"]
    X = df[feature_cols].astype("float32").values
    le = LabelEncoder()
    y = le.fit_transform(df["Label"].values)
    n_cls = len(le.classes_)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr).astype("float32"), scaler.transform(X_te).astype("float32")

    counts = pd.Series(y_tr).value_counts()
    target = int(counts.median())
    X_tr, y_tr = RandomUnderSampler(
        sampling_strategy={c: min(int(n), target) for c, n in counts.items()},
        random_state=seed,
    ).fit_resample(X_tr, y_tr)
    X_tr, y_tr = SMOTE(
        sampling_strategy={c: target for c in np.unique(y_tr)},
        random_state=seed, k_neighbors=3,
    ).fit_resample(X_tr, y_tr)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FFN(X_tr.shape[1], n_cls).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()),
        batch_size=512, shuffle=True,
    )
    for ep in range(epochs):
        model.train(); total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            total += loss.item() * xb.size(0)
        print(f"epoch {ep + 1}/{epochs}  loss={total / len(loader.dataset):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_te).to(dev)).cpu().numpy()
    preds = logits.argmax(1)
    rep = classification_report(y_te, preds, target_names=le.classes_, digits=3, zero_division=0)
    print(rep)
    (REPORTS / "ffn_pt_report.txt").write_text(rep)
    torch.save(model.state_dict(), ART / "ffn_pt.pt")
    joblib.dump(scaler, ART / "scaler.joblib")
    joblib.dump(le, ART / "label_encoder.joblib")
    (ART / "feature_columns.json").write_text(json.dumps(feature_cols))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args.epochs, args.seed)
