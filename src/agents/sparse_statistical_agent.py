"""
Agent 3: Sparse Statistical anomaly detection via robust Z-scores.

Complements the tree-based Agents 1 & 2 with a purely statistical signal:
- Uses 11 behaviorally-focused features (no raw volume columns)
- Computes robust Z-scores using median and MAD (median absolute deviation)
- Caps absolute Z-scores at ZSCORE_CAP to prevent statistical saturation
- Aggregates via RMS (root-mean-square) of top-K absolute Z-scores per customer
- Fully interpretable: the explanation matrix IS the raw Z-score per feature
"""

import logging

import numpy as np

from src.config import SPARSE_FEATURES, SPARSE_TOP_K, SPARSE_EPS, ZSCORE_CAP
from src.agents.base import BaseAgent, AgentResult

log = logging.getLogger("aml_pipeline")


class SparseStatisticalAgent(BaseAgent):
    """Robust Z-score anomaly detection on sparse behavioral features."""

    @property
    def name(self):
        return "sparse_statistical"

    @property
    def version(self):
        return "2.0"

    def score(self, features_df):
        """Compute robust Z-scores and aggregate into anomaly scores."""
        X = features_df[SPARSE_FEATURES].values
        n_samples, n_features = X.shape

        log.info(f"[{self.name}] Feature matrix: {n_samples:,} samples x {n_features} features")
        log.info(f"[{self.name}] Features: {SPARSE_FEATURES}")

        # --- Compute robust Z-scores ---
        medians = np.median(X, axis=0)                           # (n_features,)
        mad = np.median(np.abs(X - medians), axis=0)             # (n_features,)

        # 0.6745 is the z-score of the 75th percentile of N(0,1),
        # making MAD-based z-scores comparable to standard z-scores
        z_scores = 0.6745 * (X - medians) / (mad + SPARSE_EPS)    # (n_samples, n_features)

        # Cap absolute z-scores to prevent saturation from extreme outliers
        z_scores = np.clip(z_scores, -ZSCORE_CAP, ZSCORE_CAP)

        log.info(f"[{self.name}] Z-score matrix computed (capped at ±{ZSCORE_CAP}). "
                 f"MAD range: [{mad.min():.4f}, {mad.max():.4f}]")

        # --- Aggregate: RMS of top-K absolute Z-scores per customer ---
        abs_z = np.abs(z_scores)

        # Partition to find the top-K values per row efficiently (O(n) vs O(n log n))
        top_k = min(SPARSE_TOP_K, n_features)
        partitioned = np.partition(abs_z, kth=n_features - top_k, axis=1)
        top_k_values = partitioned[:, n_features - top_k:]       # (n_samples, top_k)
        raw_scores = np.sqrt(np.mean(top_k_values ** 2, axis=1)) # RMS of top-K |z|

        log.info(f"[{self.name}] Raw score range (RMS top-{top_k} |z|): "
                 f"[{raw_scores.min():.4f}, {raw_scores.max():.4f}]")

        # --- Normalize to [0, 1] via sigmoid on RMS z-score ---
        # Sigmoid centered at RMS=3 (severe outlier threshold).
        # With z-score cap at 8, max possible RMS ≈ 8 → sigmoid ≈ 0.993
        model_scores = 1.0 / (1.0 + np.exp(-(raw_scores - 3.0)))

        log.info(f"[{self.name}] Scoring complete.")

        return AgentResult(
            agent_name=self.name,
            scores=model_scores,
            explanations=z_scores,
            metadata={
                "medians": medians,
                "mad": mad,
                "features_used": SPARSE_FEATURES,
                "zscore_cap": ZSCORE_CAP,
            },
        )
