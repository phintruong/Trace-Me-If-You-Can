"""Unit tests for pipeline: preprocess, graph_builder."""
import os
import sys
import pytest
import pandas as pd

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


def test_graph_builder_from_raw(sample_raw_df):
    """Build graph from raw df returns nodes, edges, account_to_id."""
    from app.pipeline.graph_builder import build_graph_from_raw, detect_patterns
    nodes, edges, account_to_id, id_to_account = build_graph_from_raw(sample_raw_df)
    assert len(nodes) >= 2
    assert len(edges) == 3
    assert "A1" in account_to_id
    assert "A2" in account_to_id or "B1" in account_to_id
    patterns = detect_patterns(edges, account_to_id)
    assert isinstance(patterns, dict)


def test_graph_for_api():
    """API graph helper returns nodes and edges with id, type, label."""
    from app.pipeline.graph_builder import txs_to_graph_for_api
    df = pd.DataFrame([
        {"transaction_id": "tx_1", "account_id": "AC1", "Payment Format": "D1", "Account.1": "M1", "From Bank": 1},
        {"transaction_id": "tx_2", "account_id": "AC1", "Payment Format": "D1", "Account.1": "M2", "From Bank": 2},
    ])
    nodes, edges = txs_to_graph_for_api(df, account_id="AC1")
    assert len(nodes) >= 2
    assert len(edges) >= 2
    node_ids = {n["id"] for n in nodes}
    assert "account_AC1" in node_ids
    for n in nodes:
        assert "id" in n and "type" in n and "label" in n
    for e in edges:
        assert "source" in e and "target" in e and "type" in e
