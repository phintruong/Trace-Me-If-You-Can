"""
Run full AML pipeline: load -> preprocess -> graph -> GNN -> persist -> Railtracks -> output.
Caches result for GET /flagged and GET /graph.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import (
    DATASET_SOURCE,
    OUTPUT_DIR,
    PREDICTIONS_PARQUET,
    RISK_THRESHOLD,
)
from app.pipeline.loader import load_dataset
from app.pipeline.preprocess import preprocess
from app.pipeline.graph_builder import build_graph_from_raw, detect_patterns
from app.pipeline.gnn_runner import run_gnn
from app.pipeline.railtracks_explainer import run_railtracks_explainer
from app.services.db_client import init_db, save_predictions

logger = logging.getLogger(__name__)

# In-memory cache of last pipeline run for GET /flagged and GET /graph
_last_run_output: dict[str, Any] | None = None


@dataclass
class PipelineResult:
    """Result of one pipeline run."""
    scored_df: pd.DataFrame
    account_risk_scores: dict[str, float]
    graph_nodes: list[dict]
    graph_edges: list[dict]
    account_patterns: dict[str, list[str]]
    flagged_accounts: list[dict[str, Any]]
    api_output: dict[str, Any]


def get_last_run_output() -> dict[str, Any] | None:
    """Return cached API output from last POST /pipeline/run."""
    return _last_run_output


def run_pipeline(
    source: str | None = None,
    file_name: str | None = None,
    risk_threshold: float | None = None,
    max_flagged: int = 50,
    model_path: str | Path | None = None,
) -> PipelineResult:
    """
    Load dataset -> preprocess -> build graph -> run GNN -> save to DB/parquet ->
    run Railtracks explainer -> build api_output. Cache result for GET /flagged, GET /graph.
    """
    global _last_run_output
    src = source or DATASET_SOURCE
    threshold = risk_threshold if risk_threshold is not None else RISK_THRESHOLD

    logger.info("Loading dataset from source=%s", src)
    raw_df = load_dataset(source=src, file_name=file_name)
    logger.info("Loaded %d rows", len(raw_df))

    df = preprocess(raw_df)
    logger.info("Preprocessed; columns: %s", list(df.columns))

    graph_nodes, graph_edges, account_to_id, id_to_account = build_graph_from_raw(raw_df)
    account_patterns = detect_patterns(graph_edges, account_to_id)

    scored_df, account_risk_scores = run_gnn(df, model_path=model_path)
    logger.info(
        "GNN scored; risk_score range [%s, %s]",
        scored_df["risk_score"].min(),
        scored_df["risk_score"].max(),
    )

    init_db()
    save_predictions(scored_df)
    logger.info("Saved predictions to DB")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df = scored_df[["transaction_id", "account_id", "timestamp", "amount", "risk_score"]].copy()
    if "top_features" in scored_df.columns:
        out_df["top_features"] = scored_df["top_features"].apply(
            lambda x: json.dumps(x) if x is not None else "[]"
        )
    out_df.to_parquet(PREDICTIONS_PARQUET, index=False)
    logger.info("Wrote %s", PREDICTIONS_PARQUET)

    flagged_accounts = run_railtracks_explainer(
        account_risk_scores=account_risk_scores,
        account_patterns=account_patterns,
        graph_edges=graph_edges,
        risk_threshold=threshold,
        max_flagged=max_flagged,
    )

    api_output = {
        "flagged_accounts": [
            {
                "account_id": fa["account_id"],
                "risk_score": fa["risk_score"],
                "detected_patterns": fa["detected_patterns"],
                "pattern_agent_summary": fa["pattern_agent_summary"],
                "risk_agent_summary": fa["risk_agent_summary"],
                "investigator_explanation": fa["investigator_explanation"],
                "graph_connections": fa["graph_connections"],
            }
            for fa in flagged_accounts
        ],
        "graph": {"nodes": graph_nodes, "edges": graph_edges},
        "meta": {
            "total_flagged": len(flagged_accounts),
            "total_nodes": len(graph_nodes),
            "total_edges": len(graph_edges),
        },
    }
    _last_run_output = api_output

    return PipelineResult(
        scored_df=scored_df,
        account_risk_scores=account_risk_scores,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        account_patterns=account_patterns,
        flagged_accounts=flagged_accounts,
        api_output=api_output,
    )
