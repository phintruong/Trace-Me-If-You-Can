"""POST /pipeline/run; GET /flagged and GET /graph/{id} are registered in main at root."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.pipeline.run_pipeline import run_pipeline, get_last_run_output

router = APIRouter(prefix="/pipeline", tags=["pipeline"])
logger = logging.getLogger(__name__)


@router.post("/run")
def pipeline_run(
    source: str | None = None,
    file_name: str | None = None,
    risk_threshold: float | None = None,
    max_flagged: int = 50,
) -> dict[str, Any]:
    """Run full AML pipeline (load -> preprocess -> GNN -> Railtracks). Caches result for GET /flagged and GET /graph."""
    try:
        result = run_pipeline(
            source=source,
            file_name=file_name,
            risk_threshold=risk_threshold,
            max_flagged=max_flagged,
        )
        return result.api_output
    except Exception as e:
        logger.exception("Pipeline run failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def get_flagged_view():
    """Return flagged accounts from last pipeline run. Run POST /pipeline/run first."""
    out = get_last_run_output()
    if out is None:
        raise HTTPException(status_code=404, detail="No pipeline run yet. Call POST /pipeline/run first.")
    return out.get("flagged_accounts", [])


def get_graph_view(id: str | None = None):
    """Return graph (nodes, edges) from last run. Optional id = account_id to filter subgraph."""
    out = get_last_run_output()
    if out is None:
        raise HTTPException(status_code=404, detail="No pipeline run yet. Call POST /pipeline/run first.")
    graph = out.get("graph", {"nodes": [], "edges": []})
    if id is not None and id.strip():
        aid = id.strip()
        nodes = graph.get("nodes", [])
        edges = [e for e in graph.get("edges", []) if e.get("from") == aid or e.get("to") == aid]
        node_ids = {e["from"] for e in edges} | {e["to"] for e in edges}
        nodes = [n for n in nodes if n.get("id") in node_ids]
        return {"nodes": nodes, "edges": edges}
    return graph
