"""API contract tests: /alerts, /account, /graph-data, /explain. Mock DB and watsonx."""
import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
for p in (ROOT, BACKEND):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("DB_MODE", "sqlite")
os.environ.setdefault("DB_CONN_STRING", ":memory:")
os.environ.setdefault("RISK_THRESHOLD", "0.7")


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


@pytest.fixture
def mock_db_alerts():
    return [
        {
            "transaction_id": "TX123",
            "account_id": "AC456",
            "timestamp": "2026-03-13T12:22:00Z",
            "amount": 1200.50,
            "risk_score": 0.92,
            "summary": "Large amount, new device, location anomaly",
            "explain_cached": True,
        }
    ]


def test_alerts_returns_200_and_list(client, mock_db_alerts):
    with patch("app.api.alerts.db_client.get_alerts", return_value=mock_db_alerts):
        r = client.get("/alerts?limit=50")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    if data:
        assert "transaction_id" in data[0]
        assert "risk_score" in data[0]
        assert "X-Request-ID" in r.headers


def test_alerts_500_on_failure(client):
    with patch("app.api.alerts.db_client.get_alerts", side_effect=RuntimeError("db error")):
        r = client.get("/alerts")
    assert r.status_code == 500


def test_account_200(client):
    with patch(
        "app.api.account.db_client.get_account_transactions",
        return_value=[
            {"transaction_id": "tx1", "timestamp": "2026-03-13T12:00:00Z", "amount": 100.0, "risk_score": 0.5},
        ],
    ):
        r = client.get("/account/AC1")
    assert r.status_code == 200
    data = r.json()
    assert data["account_id"] == "AC1"
    assert "transactions" in data
    assert "trend" in data


def test_account_404(client):
    with patch("app.api.account.db_client.get_account_transactions", return_value=[]):
        r = client.get("/account/NONE")
    assert r.status_code == 404


def test_graph_data_200(client):
    with patch(
        "app.api.graph.db_client.get_all_predictions_for_graph",
        return_value=[
            {"transaction_id": "tx1", "account_id": "AC1", "timestamp": "", "amount": 10, "risk_score": 0.5, "top_features": []},
        ],
    ):
        r = client.get("/graph-data?account_id=AC1")
    assert r.status_code == 200
    data = r.json()
    assert "nodes" in data
    assert "edges" in data


def test_graph_data_404(client):
    with patch("app.api.graph.db_client.get_all_predictions_for_graph", return_value=[]):
        r = client.get("/graph-data?account_id=AC1")
    assert r.status_code == 404


def test_explain_cached(client):
    with patch("app.api.explain.db_client.get_transaction_row", return_value={"transaction_id": "tx1", "account_id": "AC1", "risk_score": 0.8}):
        with patch("app.api.explain.db_client.get_explain_cache", return_value={"summary_text": "Cached summary.", "model_version": "granite-13b-instruct"}):
            r = client.get("/explain/tx1")
    assert r.status_code == 200
    data = r.json()
    assert data["summary"] == "Cached summary."
    assert "model" in data


def test_explain_404(client):
    with patch("app.api.explain.db_client.get_transaction_row", return_value=None):
        r = client.get("/explain/nonexistent")
    assert r.status_code == 404


def test_explain_mock_watsonx(client):
    with patch("app.api.explain.db_client.get_transaction_row", return_value={
        "transaction_id": "tx1", "account_id": "AC1", "timestamp": "2026-03-13T12:00:00Z",
        "amount": 50.0, "risk_score": 0.9, "top_features": [["amount", 0.4]],
    }):
        with patch("app.api.explain.db_client.get_explain_cache", return_value=None):
            with patch("app.api.explain.watsonx_client.generate_summary", return_value="Deterministic summary from mock."):
                with patch("app.api.explain.db_client.set_explain_cache"):
                    r = client.get("/explain/tx1")
    assert r.status_code == 200
    assert "Deterministic summary from mock" in r.json().get("summary", "")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "db" in data


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "alerts_generated" in data
    assert "watsonx_calls" in data
    assert "watsonx_errors" in data


def test_integration_pipeline_then_alerts():
    """Save predictions to DB, then get_alerts returns at least one (same DB)."""
    import tempfile
    import pandas as pd
    import importlib

    # Use a real file so init_db and save_predictions use the same DB (SQLite :memory: is per-connection)
    fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = fd.name
    fd.close()
    try:
        os.environ["DB_CONN_STRING"] = db_path
        import app.config as cfg
        import app.services.db_client as db
        importlib.reload(cfg)
        importlib.reload(db)
        db.init_db()
        df = pd.DataFrame([{
            "transaction_id": "tx_integ",
            "account_id": "AC1",
            "timestamp": "2026-03-13T12:00:00Z",
            "amount": 999.0,
            "risk_score": 0.95,
            "top_features": [],
        }])
        db.save_predictions(df)
        alerts = db.get_alerts(threshold=0.7, limit=50)
        assert alerts is not None
        assert len(alerts) >= 1
        assert any(a["transaction_id"] == "tx_integ" for a in alerts)
    finally:
        os.environ.pop("DB_CONN_STRING", None)
        try:
            os.unlink(db_path)
        except Exception:
            pass
