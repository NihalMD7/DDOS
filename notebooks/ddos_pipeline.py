# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DDoS Attack Detection — CICDDoS2019
#
# Multi-class classifier over 12 DDoS attack families plus benign traffic, trained on
# raw network-flow features from the **CICDDoS2019** dataset.
#
# Pipeline:
# 1. Load every `*-training.parquet` and `*-testing.parquet` shard from `data/`.
# 2. Clean numeric columns (drop inf/NaN, cap extremes).
# 3. Normalize the label space — collapse `DrDoS_X` and `X` variants into a single class.
# 4. Hybrid **SMOTE + random undersampling** to keep minority recall while
#    avoiding a benign-only collapse.
# 5. Train a **Feed-forward NN** and a **1D-CNN** in Keras (PyTorch ports live
#    in `models/pytorch/`).
# 6. Evaluate per-class precision / recall / F1, persist artifacts to `artifacts/`
#    for the FastAPI service in `api/main.py`.

# %%
import glob
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import layers, models, callbacks

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT = Path("/root/prep/job/ddos-detection")
DATA_DIR = PROJECT / "data"
ART_DIR = PROJECT / "artifacts"
REPORT_DIR = PROJECT / "reports"
ART_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Load every parquet shard

# %%
paths = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
print(f"discovered {len(paths)} shards")
frames = [pd.read_parquet(p) for p in paths]
df = pd.concat(frames, ignore_index=True)
print("raw shape:", df.shape)
print(df["Label"].value_counts().head(20))

# %% [markdown]
# ## 2. Clean + collapse label variants

# %%
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df["Label"] = df["Label"].str.replace("^DrDoS_", "", regex=True)
df["Label"] = df["Label"].str.replace("UDP-lag", "UDPLag", regex=False)
label_counts = df["Label"].value_counts()
print("collapsed label distribution:")
print(label_counts)

# drop classes that are too rare for stratified resampling
keep = label_counts[label_counts >= 20].index
df = df[df["Label"].isin(keep)].reset_index(drop=True)
print("after rare-class drop:", df.shape, "| classes:", df["Label"].nunique())

# %% [markdown]
# ## 3. Features / labels / scaling

# %%
feature_cols = [c for c in df.columns if c != "Label"]
X = df[feature_cols].astype("float32").values
le = LabelEncoder()
y = le.fit_transform(df["Label"].values)
n_classes = len(le.classes_)
print("features:", X.shape, "classes:", n_classes, "->", list(le.classes_))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype("float32")
X_test = scaler.transform(X_test).astype("float32")

# %% [markdown]
# ## 4. Hybrid SMOTE + random undersampling
#
# Cap the dominant classes, then oversample the long tail so every class lands at
# a similar count. This restores minority recall without collapsing on benign.

# %%
counts = pd.Series(y_train).value_counts()
print("pre-balance:", counts.to_dict())
target_high = int(counts.median())
under_strategy = {cls: min(int(c), target_high) for cls, c in counts.items()}
under = RandomUnderSampler(sampling_strategy=under_strategy, random_state=SEED)
X_tr, y_tr = under.fit_resample(X_train, y_train)

over_strategy = {cls: target_high for cls in np.unique(y_tr)}
smote = SMOTE(sampling_strategy=over_strategy, random_state=SEED, k_neighbors=3)
X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
print("post-balance:", pd.Series(y_tr).value_counts().to_dict())

# %% [markdown]
# ## 5a. Feed-forward NN (TF/Keras)

# %%
def build_ffn(in_dim: int, n_cls: int) -> tf.keras.Model:
    m = models.Sequential([
        layers.Input(shape=(in_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_cls, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

ffn = build_ffn(X_tr.shape[1], n_classes)
ffn.summary()
ffn_hist = ffn.fit(
    X_tr, y_tr,
    validation_data=(X_test, y_test),
    epochs=12, batch_size=512, verbose=2,
    callbacks=[callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
)

# %% [markdown]
# ## 5b. 1D-CNN (TF/Keras)
#
# Treat the 77-feature vector as a length-77 sequence with one channel. Conv1D
# learns local feature interactions; global pooling keeps the head small.

# %%
def build_cnn(in_dim: int, n_cls: int) -> tf.keras.Model:
    m = models.Sequential([
        layers.Input(shape=(in_dim, 1)),
        layers.Conv1D(64, 3, activation="relu", padding="same"),
        layers.Conv1D(64, 3, activation="relu", padding="same"),
        layers.MaxPool1D(2),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_cls, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

X_tr_c = X_tr[..., None]
X_te_c = X_test[..., None]
cnn = build_cnn(X_tr.shape[1], n_classes)
cnn.summary()
cnn_hist = cnn.fit(
    X_tr_c, y_tr,
    validation_data=(X_te_c, y_test),
    epochs=10, batch_size=512, verbose=2,
    callbacks=[callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
)

# %% [markdown]
# ## 6. Evaluation

# %%
def report(model, X, y, name, channel=False):
    p = model.predict(X[..., None] if channel else X, verbose=0).argmax(1)
    r = classification_report(y, p, target_names=le.classes_, digits=3, zero_division=0)
    print(f"\n=== {name} ===\n{r}")
    with open(REPORT_DIR / f"{name}_report.txt", "w") as fh:
        fh.write(r)
    return r

report(ffn, X_test, y_test, "ffn_tf")
report(cnn, X_test, y_test, "cnn_tf", channel=True)

# %% [markdown]
# ## 7. Persist artifacts for the FastAPI inference service

# %%
ffn.save(ART_DIR / "ffn_tf.keras")
cnn.save(ART_DIR / "cnn_tf.keras")
joblib.dump(scaler, ART_DIR / "scaler.joblib")
joblib.dump(le, ART_DIR / "label_encoder.joblib")
with open(ART_DIR / "feature_columns.json", "w") as fh:
    json.dump(feature_cols, fh)
print("artifacts written:", sorted(p.name for p in ART_DIR.iterdir()))
