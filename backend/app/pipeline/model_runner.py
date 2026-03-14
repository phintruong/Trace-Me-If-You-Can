"""Score transactions via local model (joblib) or MODEL_URL microservice."""

import json
from typing import List, Tuple

import numpy as np
import pandas as pd

from app.config import MODEL_PATH, MODEL_URL

# Feature column order must match training
FEATURE_COLUMNS = [
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Hour",
    "DayOfWeek",
    "Day",
    "Month",
]


def _get_model():
    """Load model from MODEL_PATH (joblib)."""
    import joblib
    return joblib.load(MODEL_PATH)


def _score_local(df: pd.DataFrame, model) -> pd.DataFrame:
    """Score using local sklearn-style model."""
    X = df[FEATURE_COLUMNS].fillna(0).astype(np.float64)
    risk_scores = model.predict_proba(X)[:, 1]
    risk_scores = np.clip(risk_scores, 0.0, 1.0)
    out = df.copy()
    out["risk_score"] = risk_scores
    # Optional: top features from tree feature_importances_
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        names = FEATURE_COLUMNS
        top_idx = np.argsort(imp)[::-1][:5]
        top_features = [(names[i], float(imp[i])) for i in top_idx]
        out["top_features"] = [top_features] * len(out)
    else:
        out["top_features"] = [[]] * len(out)
    return out


def _score_remote(df: pd.DataFrame) -> pd.DataFrame:
    """Score via MODEL_URL POST. Expects JSON body with rows, returns risk_scores and optional top_features."""
    import requests
    X = df[FEATURE_COLUMNS].fillna(0)
    payload = X.to_dict(orient="records")
    r = requests.post(
        MODEL_URL.rstrip("/") + "/predict",
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    risk_scores = np.array(data.get("risk_scores", data.get("scores", [])))
    risk_scores = np.clip(risk_scores, 0.0, 1.0)
    out = df.copy()
    out["risk_score"] = risk_scores
    out["top_features"] = data.get("top_features", [[]] * len(out))
    if len(out["top_features"]) != len(out):
        out["top_features"] = [[]] * len(out)
    return out


def score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run fraud model on preprocessed DataFrame.
    Uses MODEL_PATH (joblib) if set, else MODEL_URL (HTTP). Appends risk_score and top_features.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    if MODEL_PATH:
        model = _get_model()
        return _score_local(df, model)
    if MODEL_URL:
        return _score_remote(df)
    raise RuntimeError("Set MODEL_PATH or MODEL_URL to score transactions.")
