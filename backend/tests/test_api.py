"""API contract tests: /health, /accounts/{id}, /flagged, /graph/{id}. Mock DB and pipeline."""
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


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "db" in data


def test_accounts_200(client):
    with patch(
        "app.api.account.db_client.get_account_transactions",
        return_value=[
            {"transaction_id": "tx1", "timestamp": "2026-03-13T12:00:00Z", "amount": 100.0, "risk_score": 0.5},
        ],
    ):
        with patch("app.api.account.db_client.get_account_highest_risk_row", return_value={
            "transaction_id": "tx1", "account_id": "AC1", "risk_score": 0.5, "top_features": [],
        }):
            with patch("app.api.account.db_client.get_explain_cache", return_value={"summary_text": "Cached.", "model_version": "x"}):
                r = client.get("/accounts/AC1")
    assert r.status_code == 200
    data = r.json()
    assert "account_id" in data
    assert "flag" in data
    assert "aiExplanation" in data


def test_accounts_404(client):
    with patch("app.api.account.db_client.get_account_transactions", return_value=[]):
        r = client.get("/accounts/NONE")
    assert r.status_code == 404


def test_flagged_404_without_run(client):
    r = client.get("/flagged")
    assert r.status_code == 404


def test_flagged_200_after_run(client):
    with patch("app.api.pipeline.get_last_run_output", return_value={
        "flagged_accounts": [{"account_id": "A1", "risk_score": 0.9, "detected_patterns": ["hub"]}],
        "graph": {"nodes": [], "edges": []},
        "meta": {},
    }):
        r = client.get("/flagged")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    if data:
        assert data[0]["account_id"] == "A1"


def test_graph_404_without_run(client):
    r = client.get("/graph/1")
    assert r.status_code == 404


def test_graph_200_with_run(client):
    with patch("app.api.pipeline.get_last_run_output", return_value={
        "flagged_accounts": [],
        "graph": {"nodes": [{"id": "1", "label": "1"}], "edges": [{"from": "1", "to": "2", "amount": 10}]},
        "meta": {},
    }):
        r = client.get("/graph/1")
    assert r.status_code == 200
    data = r.json()
    assert "nodes" in data
    assert "edges" in data
