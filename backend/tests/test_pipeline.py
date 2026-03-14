"""Unit tests for pipeline: preprocess, score, graph_gen."""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Path setup
BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("DB_MODE", "sqlite")
os.environ.setdefault("DB_CONN_STRING", ":memory:")


@pytest.fixture
def sample_raw_df():
    """Minimal DataFrame with IBM required columns."""
    return pd.DataFrame({
        "Timestamp": ["2026-03-13 12:00:00", "2026-03-13 13:00:00", "2026-03-13 14:00:00"],
        "From Bank": [1, 2, 1],
        "Account": ["A1", "A1", "A2"],
        "To Bank": [2, 1, 2],
        "Account.1": ["B1", "B2", "B1"],
        "Amount Received": [100.0, 200.0, 50.0],
        "Receiving Currency": ["USD", "USD", "USD"],
        "Amount Paid": [100.0, 200.0, 50.0],
        "Payment Currency": ["USD", "USD", "USD"],
        "Payment Format": [1, 2, 1],
        "Is Laundering": [0, 0, 0],
    })


def test_preprocess(sample_raw_df):
    """Preprocess produces transaction_id, account_id, and derived features."""
    from app.pipeline.preprocess import preprocess
    df = preprocess(sample_raw_df)
    assert "transaction_id" in df.columns
    assert "account_id" in df.columns
    assert "amount" in df.columns
    assert "amount_ratio" in df.columns
    assert "time_since_last_tx" in df.columns
    assert "device_change_flag" in df.columns
    assert "location_distance_km" in df.columns
    assert len(df) == len(sample_raw_df)
    assert df["transaction_id"].iloc[0].startswith("tx_")
    assert df["account_id"].iloc[0] in ("A1", "A2")


def test_score_mock():
    """Score adds risk_score column when model is mocked."""
    from unittest.mock import patch, MagicMock
    from app.pipeline.model_runner import FEATURE_COLUMNS, score
    n = 3
    df = pd.DataFrame({c: [0] * n for c in FEATURE_COLUMNS})
    df["transaction_id"] = [f"tx_{i}" for i in range(n)]
    df["account_id"] = ["AC1", "AC1", "AC2"]
    df["amount"] = [10.0, 20.0, 5.0]
    df["timestamp"] = ["2026-03-13T12:00:00Z"] * n

    # Without MODEL_PATH or mock, score() raises
    with patch.dict(os.environ, {}, clear=False):
        if "MODEL_PATH" in os.environ:
            del os.environ["MODEL_PATH"]
        if "MODEL_URL" in os.environ:
            del os.environ["MODEL_URL"]
        with pytest.raises((RuntimeError, ValueError)):
            score(df)

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.9, 0.1], [0.7, 0.3]])
    mock_model.feature_importances_ = np.zeros(len(FEATURE_COLUMNS))
    with patch("app.pipeline.model_runner.MODEL_PATH", "/fake/model.pkl"):
        with patch("app.pipeline.model_runner._get_model", return_value=mock_model):
            out = score(df)
    assert "risk_score" in out.columns
    assert len(out) == n
    assert out["risk_score"].between(0, 1).all()


def test_graphgen():
    """Graph generator returns nodes and edges with expected structure."""
    from app.pipeline.graph_gen import txs_to_graph
    df = pd.DataFrame([
        {"transaction_id": "tx_1", "account_id": "AC1", "Payment Format": "D1", "Account.1": "M1", "From Bank": 1},
        {"transaction_id": "tx_2", "account_id": "AC1", "Payment Format": "D1", "Account.1": "M2", "From Bank": 2},
    ])
    nodes, edges = txs_to_graph(df, account_id="AC1")
    assert len(nodes) >= 2
    assert len(edges) >= 2
    node_ids = {n["id"] for n in nodes}
    assert "account_AC1" in node_ids
    assert "tx_1" in node_ids or any("tx_1" in n["id"] for n in nodes)
    for n in nodes:
        assert "id" in n and "type" in n and "label" in n
    for e in edges:
        assert "source" in e and "target" in e and "type" in e
