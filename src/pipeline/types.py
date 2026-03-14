"""Structured data types passed between pipeline stages."""

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class PipelineContext:
    """Context object passed through the AML pipeline. Each stage reads/writes its outputs."""

    # Stage 1: raw transaction data
    raw_df: Optional[pd.DataFrame] = None
    file_name: Optional[str] = None

    # Stage 2: preprocessed features and targets
    feature_df: Optional[pd.DataFrame] = None
    target_series: Optional[pd.Series] = None
    feature_matrix: Optional[pd.DataFrame] = None  # X for model

    # Stage 3: fraud detection model and predictions
    model: Any = None
    pred_label: Optional[pd.Series] = None
    pred_probability: Optional[pd.Series] = None
    test_indices: Optional[pd.Index] = None  # indices into feature_df for predictions

    # Stage 4: transaction graph (accounts = nodes, transactions = edges)
    graph_nodes: list[dict[str, Any]] = field(default_factory=list)
    graph_edges: list[dict[str, Any]] = field(default_factory=list)
    account_to_id: dict[str, int] = field(default_factory=dict)
    id_to_account: dict[int, str] = field(default_factory=dict)

    # Stage 5: detected patterns per account
    account_patterns: dict[str, list[str]] = field(default_factory=dict)  # account_id -> ["circular", "hub", ...]

    # Stage 6: risk score per account (0-1)
    account_risk_scores: dict[str, float] = field(default_factory=dict)

    # Stage 7: flagged accounts with agent outputs
    flagged_accounts: list[dict[str, Any]] = field(default_factory=list)

    # Stage 8: final API-ready output
    api_output: Optional[dict[str, Any]] = None


@dataclass
class FlaggedAccountResult:
    """One flagged account with full explanation for the dashboard."""

    account_id: str
    risk_score: float
    detected_patterns: list[str]
    pattern_agent_summary: str
    risk_agent_summary: str
    investigator_explanation: str
    graph_connections: list[dict[str, Any]]  # edges involving this account for frontend
