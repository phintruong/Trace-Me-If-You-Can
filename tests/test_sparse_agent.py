"""Tests for the Sparse Statistical Agent (Agent 3)."""

import numpy as np
import pandas as pd
import pytest

from src.agents.sparse_statistical_agent import SparseStatisticalAgent
from src.config import SPARSE_FEATURES, ZSCORE_CAP


def _make_features_df(n_samples=200, seed=42):
    """Create a synthetic feature DataFrame with heavy-tailed data.

    Most samples are drawn from normal distributions; a few are extreme
    outliers to simulate AML-like patterns.
    """
    rng = np.random.RandomState(seed)
    data = {}
    for feat in SPARSE_FEATURES:
        values = rng.normal(loc=5.0, scale=1.0, size=n_samples)
        data[feat] = values

    # Inject 5 extreme outliers across multiple features
    for i in range(5):
        for feat in SPARSE_FEATURES[:6]:
            data[feat][i] = rng.uniform(50, 200)

    df = pd.DataFrame(data)
    df["customer_id"] = [f"C{i:04d}" for i in range(n_samples)]
    return df


class TestSparseAgent:
    """Verify sparse agent produces meaningful score spread."""

    def test_scores_in_0_1(self):
        df = _make_features_df()
        agent = SparseStatisticalAgent()
        result = agent.score(df)
        assert np.all(result.scores >= 0.0) and np.all(result.scores <= 1.0)

    def test_score_range_not_compressed(self):
        """Scores should use a meaningful portion of [0, 1], not cluster near zero."""
        df = _make_features_df()
        agent = SparseStatisticalAgent()
        result = agent.score(df)
        high_scores = np.sum(result.scores > 0.3)
        assert high_scores >= 3, (
            f"Expected at least 3 accounts with score > 0.3, got {high_scores}. "
            f"Score distribution: min={result.scores.min():.4f}, "
            f"max={result.scores.max():.4f}, mean={result.scores.mean():.4f}"
        )

    def test_heavy_tail_spread(self):
        """With heavy-tailed data, the top 5% of scores should be meaningfully
        separated from the median (not all compressed near zero)."""
        df = _make_features_df(n_samples=500)
        agent = SparseStatisticalAgent()
        result = agent.score(df)
        p50 = np.percentile(result.scores, 50)
        p95 = np.percentile(result.scores, 95)
        gap = p95 - p50
        assert gap > 0.01, (
            f"Top-5% to median gap too small ({gap:.4f}). Agent may be suppressed. "
            f"P50={p50:.4f}, P95={p95:.4f}"
        )

    def test_outliers_score_highest(self):
        """Injected outliers (indices 0-4) should score higher than the median."""
        df = _make_features_df()
        agent = SparseStatisticalAgent()
        result = agent.score(df)
        outlier_scores = result.scores[:5]
        median_score = np.median(result.scores)
        assert np.all(outlier_scores > median_score), (
            f"Outliers should all score above median ({median_score:.4f}), "
            f"but outlier scores are {outlier_scores}"
        )

    def test_explanations_shape(self):
        df = _make_features_df(n_samples=50)
        agent = SparseStatisticalAgent()
        result = agent.score(df)
        assert result.explanations.shape == (50, len(SPARSE_FEATURES))

    def test_version_updated(self):
        agent = SparseStatisticalAgent()
        assert agent.version == "2.0"

    def test_zscore_capped(self):
        """Z-scores in explanations should never exceed the cap."""
        df = _make_features_df()
        agent = SparseStatisticalAgent()
        result = agent.score(df)
        max_abs_z = np.max(np.abs(result.explanations))
        assert max_abs_z <= ZSCORE_CAP + 1e-6, (
            f"Max |z-score| is {max_abs_z:.4f}, exceeds cap of {ZSCORE_CAP}"
        )

    def test_zscore_cap_prevents_extreme_scores(self):
        """With extreme outliers, capped z-scores should keep scores below 1.0."""
        rng = np.random.RandomState(99)
        data = {}
        for feat in SPARSE_FEATURES:
            values = rng.normal(loc=0, scale=1.0, size=100)
            data[feat] = values
        # Inject astronomically extreme values
        for feat in SPARSE_FEATURES:
            data[feat][0] = 1e6
        df = pd.DataFrame(data)
        df["customer_id"] = [f"C{i:04d}" for i in range(100)]

        agent = SparseStatisticalAgent()
        result = agent.score(df)
        # Score should be high but not exactly 1.0 (sigmoid asymptote)
        assert result.scores[0] > 0.9
        assert result.scores[0] < 1.0
