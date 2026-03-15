"""API endpoints for graph insights: clusters, flows, timeline, roles."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.pipeline.run_pipeline import get_last_run_output
from app.pipeline.graph_analysis import get_account_flows, build_timeline

router = APIRouter(tags=["insights"])
logger = logging.getLogger(__name__)


def _require_analysis() -> dict[str, Any]:
    """Return cached analysis or raise 404."""
    out = get_last_run_output()
    if out is None or "analysis" not in out:
        raise HTTPException(
            status_code=404,
            detail="No pipeline run yet. Call POST /pipeline/run first.",
        )
    return out


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------

@router.get("/clusters")
def list_clusters(min_size: int = 2, min_risk: float = 0.0):
    """
    Return all detected communities/clusters.
    Filter by minimum size and minimum aggregate risk score.
    """
    out = _require_analysis()
    clusters = out["analysis"]["clusters"]
    result = []
    for cid_str, cluster in clusters.items():
        cid = int(cid_str) if isinstance(cid_str, str) else cid_str
        if cluster["size"] >= min_size and cluster["risk_score"] >= min_risk:
            result.append({
                "cluster_id": cid,
                "size": cluster["size"],
                "risk_score": cluster["risk_score"],
                "avg_risk": cluster["avg_risk"],
                "max_risk": cluster["max_risk"],
                "accounts": cluster["accounts"],
            })
    result.sort(key=lambda c: -c["risk_score"])
    return {"clusters": result, "total": len(result)}


@router.get("/clusters/{cluster_id}")
def get_cluster(cluster_id: int):
    """
    Return details for a specific cluster: accounts, edges, risk scores, roles.
    """
    out = _require_analysis()
    clusters = out["analysis"]["clusters"]
    # clusters keyed by int
    cluster = clusters.get(cluster_id) or clusters.get(str(cluster_id))
    if not cluster:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    roles = out["analysis"].get("roles", {})
    account_roles = {
        acc: roles.get(acc, {}) for acc in cluster["accounts"]
    }

    # Build nodes with enriched data
    nodes = []
    for acc in cluster["accounts"]:
        role_info = roles.get(acc, {})
        nodes.append({
            "id": acc,
            "role": role_info.get("role", "unknown"),
            "risk_score": role_info.get("risk_score", 0.0),
            "fan_in": role_info.get("fan_in", 0),
            "fan_out": role_info.get("fan_out", 0),
            "cluster_id": cluster_id,
        })

    return {
        "cluster_id": cluster_id,
        "size": cluster["size"],
        "risk_score": cluster["risk_score"],
        "avg_risk": cluster["avg_risk"],
        "max_risk": cluster["max_risk"],
        "nodes": nodes,
        "edges": cluster["edges"],
        "roles": account_roles,
    }


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------

@router.get("/flows/{account_id}")
def get_flows(account_id: str, max_length: int = 6, top_k: int = 10):
    """
    Return suspicious money-flow paths involving account_id.
    Each path includes ordered accounts, transactions, timestamps, roles, and a path score.
    """
    out = _require_analysis()
    analysis = out["analysis"]
    graph_edges = out.get("graph", {}).get("edges", [])
    account_risk_scores = out.get("account_risk_scores", {})
    roles = analysis.get("roles", {})

    # Check account exists
    account_cluster = analysis.get("account_cluster", {})
    if account_id not in account_cluster and account_id not in account_risk_scores:
        # Try to find in graph nodes
        node_ids = {n["id"] for n in out.get("graph", {}).get("nodes", [])}
        if account_id not in node_ids:
            raise HTTPException(status_code=404, detail=f"Account {account_id} not found")

    flows = get_account_flows(
        account_id=account_id,
        graph_edges=graph_edges,
        account_risk_scores=account_risk_scores,
        account_roles=roles,
        max_path_length=max_length,
        top_k=top_k,
    )

    return {
        "account_id": account_id,
        "flows": flows,
        "total": len(flows),
        "account_risk": account_risk_scores.get(account_id, 0.0),
        "account_role": roles.get(account_id, {}).get("role", "unknown"),
    }


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

@router.get("/timeline/{account_id}")
def get_timeline(account_id: str):
    """
    Return ordered timeline of transactions for an account.
    Includes direction, counterparty, amount, counterparty risk, and running balance.
    """
    out = _require_analysis()
    graph_edges = out.get("graph", {}).get("edges", [])
    account_risk_scores = out.get("account_risk_scores", {})
    roles = out["analysis"].get("roles", {})

    timeline = build_timeline(account_id, graph_edges, account_risk_scores)
    if not timeline:
        raise HTTPException(status_code=404, detail=f"No transactions found for account {account_id}")

    role_info = roles.get(account_id, {})

    return {
        "account_id": account_id,
        "role": role_info.get("role", "unknown"),
        "risk_score": account_risk_scores.get(account_id, 0.0),
        "timeline": timeline,
        "total_events": len(timeline),
        "total_in": round(sum(e["amount"] for e in timeline if e["direction"] == "incoming"), 2),
        "total_out": round(sum(e["amount"] for e in timeline if e["direction"] == "outgoing"), 2),
    }


# ---------------------------------------------------------------------------
# Roles (bulk)
# ---------------------------------------------------------------------------

@router.get("/roles")
def list_roles(role: str | None = None, min_risk: float = 0.0):
    """
    Return all account roles. Optionally filter by role type and minimum risk.
    """
    out = _require_analysis()
    roles = out["analysis"].get("roles", {})
    account_cluster = out["analysis"].get("account_cluster", {})

    result = []
    for acc, info in roles.items():
        if role and info.get("role") != role:
            continue
        if info.get("risk_score", 0.0) < min_risk:
            continue
        result.append({
            "account_id": acc,
            "cluster_id": account_cluster.get(acc),
            **info,
        })
    result.sort(key=lambda r: -r.get("risk_score", 0.0))
    return {"roles": result, "total": len(result)}
