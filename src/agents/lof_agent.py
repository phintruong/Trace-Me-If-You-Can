"""
Agent 4: Local Outlier Factor (LOF) density-based anomaly detection.

Complements the ensemble with a fundamentally different paradigm:
- IF agents measure global isolation (how few splits to isolate a point)
- Sparse agent measures univariate extremity (z-scores per feature)
- LOF measures LOCAL density: is this account's neighborhood sparser than
  its neighbors' neighborhoods?

This catches accounts that look "normal" globally but are unusual relative
to their nearest peers — a common pattern in sophisticated laundering where
accounts mimic aggregate statistics but occupy an unusual position in the
joint feature space.

Uses scikit-learn's LocalOutlierFactor with novelty=False (inductive mode).
Features are StandardScaler-normalized so distance metrics aren't dominated
by high-variance columns.
"""

import logging

import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.config import FEATURE_COLUMNS, LOF_N_NEIGHBORS, LOF_CONTAMINATION
from src.agents.base import BaseAgent, AgentResult

log = logging.getLogger("aml_pipeline")


class LOFAgent(BaseAgent):
    """Local Outlier Factor density-based anomaly detection agent."""

    @property
    def name(self):
        return "lof_density"

    @property
    def version(self):
        return "1.0"

    def score(self, features_df):
        """Compute LOF anomaly scores for all customers."""
        X = features_df[FEATURE_COLUMNS].values
        n_samples, n_features = X.shape

        log.info(f"[{self.name}] Feature matrix: {n_samples:,} samples x {n_features} features")

        # Standardize features so Euclidean distance is meaningful
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        log.info(f"[{self.name}] Fitting LOF (n_neighbors={LOF_N_NEIGHBORS}, "
                 f"contamination={LOF_CONTAMINATION})")

        model = LocalOutlierFactor(
            n_neighbors=min(LOF_N_NEIGHBORS, n_samples - 1),
            contamination=LOF_CONTAMINATION,
            metric="euclidean",
            n_jobs=-1,
        )

        # fit_predict returns labels; negative_outlier_factor_ gives raw scores
        model.fit_predict(X_scaled)
        raw_scores = -model.negative_outlier_factor_  # higher = more anomalous

        log.info(f"[{self.name}] Raw LOF score range: [{raw_scores.min():.4f}, {raw_scores.max():.4f}]")

        # Percentile-clip before MinMax to prevent extreme outliers from
        # compressing 99.9% of scores to near-zero.
        p1, p99 = np.percentile(raw_scores, [1, 99])
        clipped = np.clip(raw_scores, p1, p99)

        out_scaler = MinMaxScaler()
        model_scores = out_scaler.fit_transform(clipped.reshape(-1, 1)).flatten()

        log.info(f"[{self.name}] Scoring complete. "
                 f"Mean={model_scores.mean():.4f}, >0.5: {(model_scores > 0.5).sum():,}")

        # Explanations: per-feature contribution to LOF score.
        # LOF doesn't natively provide feature importances, so we compute
        # the squared deviation of each feature from the local neighborhood
        # mean, weighted by the LOF score. This shows WHICH features make
        # this account different from its neighbors.
        log.info(f"[{self.name}] Computing feature contributions...")
        contributions = self._compute_feature_contributions(
            X_scaled, model, model_scores
        )

        return AgentResult(
            agent_name=self.name,
            scores=model_scores,
            explanations=contributions,
            metadata={"model": model, "scaler": scaler, "out_scaler": out_scaler},
        )

    def _compute_feature_contributions(self, X_scaled, model, scores):
        """Compute per-feature contributions to LOF anomaly.

        For each sample, measure how much each feature deviates from the
        mean of its k-nearest neighbors. This shows WHICH features make
        this account different from its local peers.

        Uses a separate NearestNeighbors fit (same params as LOF) to get
        neighbor indices via the stable public API.
        """
        n_samples, n_features = X_scaled.shape

        # Fit a separate kNN to get neighbor indices via public API
        k = min(LOF_N_NEIGHBORS, n_samples - 1)
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        nn.fit(X_scaled)
        indices = nn.kneighbors(X_scaled, return_distance=False)  # (n_samples, k)

        # Vectorized: compute neighborhood means for all samples at once
        # indices shape: (n_samples, k) → gather neighbor rows → mean
        neighbor_features = X_scaled[indices]            # (n_samples, k, n_features)
        neighbor_means = neighbor_features.mean(axis=1)  # (n_samples, n_features)

        # Squared deviation per feature from local neighborhood center
        contributions = (X_scaled - neighbor_means) ** 2

        # Normalize per sample to sum to 1
        row_sums = contributions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        contributions = contributions / row_sums

        return contributions
