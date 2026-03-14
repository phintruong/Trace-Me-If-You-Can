"""
Minimal FastAPI app for the AML dashboard.
Run: uvicorn src.api.dashboard:app --reload
"""

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AML Pipeline Dashboard API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory cache of last pipeline run (for demo). Replace with DB/cache as needed.
_cached_output: dict[str, Any] | None = None


@app.post("/api/run-pipeline")
def run_pipeline(
    file_name: str | None = None,
    risk_threshold: float = 0.3,
    max_flagged: int = 50,
) -> dict[str, Any]:
    """Run the full AML pipeline and return structured results. Also caches for GET."""
    global _cached_output
    try:
        from src.pipeline.run_aml_pipeline import run_pipeline as _run
        ctx = _run(file_name=file_name, risk_threshold=risk_threshold, max_flagged=max_flagged)
        _cached_output = ctx.api_output
        return _cached_output or {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results")
def get_results() -> dict[str, Any]:
    """Return last pipeline run results (from cache). Run POST /api/run-pipeline first."""
    if _cached_output is None:
        raise HTTPException(status_code=404, detail="No results yet. Call POST /api/run-pipeline first.")
    return _cached_output


@app.get("/api/flagged")
def get_flagged() -> list[dict[str, Any]]:
    """Return only flagged accounts for the dashboard."""
    if _cached_output is None:
        raise HTTPException(status_code=404, detail="No results yet. Call POST /api/run-pipeline first.")
    return _cached_output.get("flagged_accounts", [])


@app.get("/api/graph")
def get_graph() -> dict[str, Any]:
    """Return graph nodes and edges for frontend visualization."""
    if _cached_output is None:
        raise HTTPException(status_code=404, detail="No results yet. Call POST /api/run-pipeline first.")
    return _cached_output.get("graph", {"nodes": [], "edges": []})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
