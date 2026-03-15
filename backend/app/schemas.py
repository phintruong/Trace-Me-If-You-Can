"""Pydantic request/response models. Keep responses small and deterministic."""

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class AccountResult(BaseModel):
    """Single account GET response: id, flag, AI explanation."""
    account_id: int
    flag: Literal["NORMAL", "SUSPICIOUS", "LAUNDERING"]
    aiExplanation: str


class AlertItem(BaseModel):
    transaction_id: str
    account_id: str
    timestamp: str
    amount: float
    risk_score: float = Field(ge=0, le=1)
    summary: Optional[str] = None
    explain_cached: bool = False


class TransactionItem(BaseModel):
    transaction_id: str
    timestamp: str
    amount: float
    risk_score: float = Field(ge=0, le=1)


class AccountResponse(BaseModel):
    account_id: str
    transactions: List[TransactionItem]
    trend: str  # e.g. "rising", "stable", "falling"


class GraphNode(BaseModel):
    id: str
    type: str
    label: str


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str
    weight: Optional[int] = 1


class GraphDataResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class ExplainResponse(BaseModel):
    summary: str
    model: str


# --- Insights models ---

class ClusterSummary(BaseModel):
    cluster_id: int
    size: int
    risk_score: float
    avg_risk: float
    max_risk: float
    accounts: List[str]


class ClusterDetail(ClusterSummary):
    nodes: List[dict]
    edges: List[dict]
    roles: dict


class FlowPath(BaseModel):
    accounts: List[str]
    transactions: List[dict]
    path_length: int
    total_value: float
    avg_risk: float
    path_score: float
    direction: Optional[str] = None
    roles: List[str]


class TimelineEvent(BaseModel):
    timestamp: str = ""
    direction: str
    counterparty: str
    amount: float
    counterparty_risk: float
    running_balance: float


class RoleInfo(BaseModel):
    account_id: str
    role: str
    fan_in: int
    fan_out: int
    total_degree: int
    in_value: float
    out_value: float
    risk_score: float = Field(ge=0, le=1)
    cluster_id: Optional[int] = None
