"""FastAPI app: POST /pipeline/run, GET /accounts/{id}, GET /flagged, GET /graph/{id}, /health."""

import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.api import account, pipeline, export as export_api, insights
from app.api.pipeline import get_flagged_view, get_graph_view
from app.config import MODEL_PATH, MODEL_URL
from app.services.db_client import init_db

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
        logger.info("DB initialized")
    except Exception as e:
        logger.warning("DB init warning: %s", e)
    yield


app = FastAPI(title="Fraud Alerts API", version="2.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)

app.include_router(account.router, prefix="/accounts", tags=["accounts"])
app.include_router(pipeline.router)
app.include_router(export_api.router)
app.include_router(insights.router)


@app.get("/flagged")
def flagged():
    """Return flagged accounts from last pipeline run. Run POST /pipeline/run first."""
    return get_flagged_view()


@app.get("/graph/{id}")
def graph(id: str | None = None):
    """Return graph (nodes, edges) from last run. id = account_id to filter subgraph."""
    return get_graph_view(id=id)


@app.get("/health")
def health():
    """Check DB connectivity and model readiness."""
    try:
        init_db()
        db_ok = True
    except Exception:
        db_ok = False
    model_ok = bool(MODEL_PATH or MODEL_URL)
    status = "ok" if db_ok and model_ok else "degraded"
    return {"status": status, "db": "ok" if db_ok else "error", "model_configured": model_ok}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
