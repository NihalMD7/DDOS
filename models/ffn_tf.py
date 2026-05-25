"""Feed-forward DDoS classifier (TensorFlow/Keras)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import callbacks, layers, models

PROJECT = Path(__file__).resolve().parent.parent
ART = PROJECT / "artifacts"
REPORTS = PROJECT / "reports"
ART.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)
SEED = 42


def load_clean() -> pd.DataFrame:
    p = ART / "cicddos2019_clean.parquet"
    if not p.exists():
        from preprocessing.build_dataset import build  # type: ignore
        return build()
    return pd.read_parquet(p)


def build_model(in_dim: int, n_cls: int, hidden=(256, 128, 64), dropout=0.3) -> tf.keras.Model:
    m = models.Sequential([layers.Input(shape=(in_dim,))])
    for h in hidden:
        m.add(layers.Dense(h, activation="relu"))
        m.add(layers.Dropout(dropout))
    m.add(layers.Dense(n_cls, activation="softmax"))
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def train(epochs: int, gridsearch: bool) -> None:
    df = load_clean()
    feature_cols = [c for c in df.columns if c != "Label"]
    X = df[feature_cols].astype("float32").values
    le = LabelEncoder()
    y = le.fit_transform(df["Label"].values)
    n_cls = len(le.classes_)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr).astype("float32"), scaler.transform(X_te).astype("float32")

    counts = pd.Series(y_tr).value_counts()
    target = int(counts.median())
    X_tr, y_tr = RandomUnderSampler(
        sampling_strategy={c: min(int(n), target) for c, n in counts.items()},
        random_state=SEED,
    ).fit_resample(X_tr, y_tr)
    X_tr, y_tr = SMOTE(
        sampling_strategy={c: target for c in np.unique(y_tr)},
        random_state=SEED, k_neighbors=3,
    ).fit_resample(X_tr, y_tr)

    if gridsearch:
        from sklearn.linear_model import LogisticRegression
        # quick stability sweep on a proxy model; deep nets are far too slow for
        # a real CV grid in this notebook setting.
        grid = GridSearchCV(
            LogisticRegression(max_iter=1000),
            param_grid={"C": [0.1, 1.0, 10.0]},
            scoring="f1_macro", cv=3, n_jobs=-1,
        )
        grid.fit(X_tr[:30_000], y_tr[:30_000])
        print("[gridsearch] best C:", grid.best_params_, "F1:", round(grid.best_score_, 3))

    model = build_model(X_tr.shape[1], n_cls)
    model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=epochs, batch_size=512, verbose=2,
        callbacks=[callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    )
    preds = model.predict(X_te, verbose=0).argmax(1)
    rep = classification_report(y_te, preds, target_names=le.classes_, digits=3, zero_division=0)
    print(rep)
    (REPORTS / "ffn_tf_report.txt").write_text(rep)
    model.save(ART / "ffn_tf.keras")
    joblib.dump(scaler, ART / "scaler.joblib")
    joblib.dump(le, ART / "label_encoder.joblib")
    (ART / "feature_columns.json").write_text(json.dumps(feature_cols))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--gridsearch", action="store_true")
    args = p.parse_args()
    train(args.epochs, args.gridsearch)
