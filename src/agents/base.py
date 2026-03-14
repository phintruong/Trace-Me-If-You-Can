"""
Base class and result type for scoring agents.

To add a new scoring agent:
    1. Subclass BaseAgent
    2. Implement name, version, and score()
    3. Register the agent in src/pipeline/runner.py (agents list)

No other files need to change — aggregation, reporting, and output
all operate on the AgentResult interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class AgentResult:
    """Standardized result from any scoring agent.

    Attributes:
        agent_name:   Unique identifier for the agent.
        scores:       Array of shape (n_customers,) with values in [0, 1].
                      Higher = more anomalous / risky.
        explanations: Agent-specific per-customer explanation data.
                      For IF agent: ndarray of shape (n_customers, n_features)
                      containing normalized feature contributions.
                      For other agents: any structure that the report generator
                      knows how to render.
        metadata:     Agent-specific data (trained model, scaler, etc.)
                      Useful for checkpointing and debugging.
    """
    agent_name: str
    scores: np.ndarray
    explanations: np.ndarray
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all scoring agents.

    Each agent takes a feature DataFrame and produces an AgentResult
    containing per-customer scores and explanations.

    The pipeline runner handles:
    - Feature computation (shared across all agents)
    - Checkpointing of agent results
    - Aggregation of multiple agent scores
    - Report generation using agent explanations
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent identifier (e.g., 'isolation_forest')."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Agent version string (e.g., '1.0')."""

    @abstractmethod
    def score(self, features_df: pd.DataFrame) -> AgentResult:
        """Score all customers.

        Args:
            features_df: DataFrame with 'customer_id' column + feature columns.

        Returns:
            AgentResult with scores in [0, 1] and per-customer explanations.
        """
