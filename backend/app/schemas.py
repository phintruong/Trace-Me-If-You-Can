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
