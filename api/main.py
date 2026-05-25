"""FastAPI inference service for the trained DDoS classifier.

Loads the Keras CNN (preferred) or FFN, the fitted scaler, and the label
encoder from `artifacts/` at startup. Exposes:

    GET  /health
    POST /predict        # one flow vector
    POST /predict/batch  # list of flow vectors

Run from the project root:

    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT = Path(__file__).resolve().parent.parent
ART = PROJECT / "artifacts"

app = FastAPI(title="DDoS Detection API", version="1.0")


class Flow(BaseModel):
    features: List[float] = Field(..., description="77 CICDDoS2019 flow features in column order")


class BatchRequest(BaseModel):
    flows: List[Flow]


_state: dict = {}


@app.on_event("startup")
def _load() -> None:
    ffn = ART / "ffn_tf.keras"
    cnn = ART / "cnn_tf.keras"
    model_path = ffn if ffn.exists() else cnn
    if not model_path.exists():
        raise RuntimeError("no trained model found in artifacts/")
    _state["model"] = tf.keras.models.load_model(model_path)
    _state["arch"] = "ffn" if model_path.name.startswith("ffn") else "cnn"
    _state["scaler"] = joblib.load(ART / "scaler.joblib")
    _state["labels"] = joblib.load(ART / "label_encoder.joblib")
    _state["columns"] = json.loads((ART / "feature_columns.json").read_text())


def _predict(matrix: np.ndarray) -> list[dict]:
    s = _state["scaler"].transform(matrix).astype("float32")
    if _state["arch"] == "cnn":
        s = s[..., None]
    probs = _state["model"].predict(s, verbose=0)
    classes = _state["labels"].classes_
    out = []
    for row in probs:
        idx = int(row.argmax())
        out.append({"label": str(classes[idx]), "confidence": float(row[idx])})
    return out


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "arch": _state.get("arch"), "classes": list(_state["labels"].classes_)}


@app.post("/predict")
def predict(flow: Flow) -> dict:
    cols = _state["columns"]
    if len(flow.features) != len(cols):
        raise HTTPException(status_code=400, detail=f"expected {len(cols)} features, got {len(flow.features)}")
    return _predict(np.array([flow.features], dtype="float32"))[0]


@app.post("/predict/batch")
def predict_batch(req: BatchRequest) -> dict:
    cols = _state["columns"]
    rows = []
    for f in req.flows:
        if len(f.features) != len(cols):
            raise HTTPException(status_code=400, detail="bad feature length")
        rows.append(f.features)
    return {"predictions": _predict(np.array(rows, dtype="float32"))}
