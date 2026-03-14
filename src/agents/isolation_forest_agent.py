"""
Agent 1: Isolation Forest anomaly detection.

Wraps scikit-learn's IsolationForest with:
- MinMaxScaler inversion for [0, 1] output scores
- Per-sample feature contribution via tree path-depth analysis
- Full vectorization for both scoring and importance computation

GPU note: scikit-learn's IsolationForest is CPU-only. The algorithm is
inherently sequential per tree (recursive partitioning) and the data fits
in RAM at 60K x 18, so GPU transfer overhead would exceed any parallel gain.
The n_jobs=-1 flag already parallelizes across CPU cores via joblib.
"""

import logging

import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    IF_N_ESTIMATORS, IF_MAX_SAMPLES, IF_RANDOM_STATE, FEATURE_COLUMNS,
)
from src.agents.base import BaseAgent, AgentResult

log = logging.getLogger("aml_pipeline")


class IsolationForestAgent(BaseAgent):
    """Isolation Forest anomaly detection agent."""

    @property
    def name(self):
        return "isolation_forest"

    @property
    def version(self):
        return "1.0"

    def score(self, features_df):
        """Train Isolation Forest and return scored results."""
        X = features_df[FEATURE_COLUMNS].values
        n_samples = X.shape[0]

        log.info(f"[{self.name}] Feature matrix: {n_samples:,} samples x {X.shape[1]} features")
        log.info(f"[{self.name}] Training Isolation Forest (n_estimators={IF_N_ESTIMATORS}, "
                 f"max_samples={min(IF_MAX_SAMPLES, n_samples)})")

        model = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            max_samples=min(IF_MAX_SAMPLES, n_samples),
            contamination="auto",
            random_state=IF_RANDOM_STATE,
            n_jobs=-1,
        )

        log.info(f"[{self.name}] Fitting model...")
        model.fit(X)
        log.info(f"[{self.name}] Model fit complete.")

        log.info(f"[{self.name}] Computing decision function scores...")
        raw_scores = model.decision_function(X)

        # Invert and scale to [0, 1]: higher = more anomalous
        scaler = MinMaxScaler()
        inverted = -raw_scores.reshape(-1, 1)
        model_scores = scaler.fit_transform(inverted).flatten()
        log.info(f"[{self.name}] Scoring complete.")

        log.info(f"[{self.name}] Computing feature contributions...")
        contributions = self._compute_feature_importances(model, X)

        return AgentResult(
            agent_name=self.name,
            scores=model_scores,
            explanations=contributions,
            metadata={"model": model, "scaler": scaler},
        )

    def _compute_feature_importances(self, model, X):
        """Compute per-account feature contributions using path-depth analysis.

        For each sample, compute the average depth at which each feature is used
        across all trees. Features used at shallow depths contribute more to isolation.

        Vectorized: all samples are routed through each tree simultaneously using
        numpy array indexing instead of a Python for-loop over samples.
        """
        n_features = X.shape[1]
        n_samples = X.shape[0]

        contributions = np.zeros((n_samples, n_features))

        for tree in tqdm(
            model.estimators_, desc="  Feature importance", unit="tree", mininterval=1.0
        ):
            tree_model = tree.tree_
            feat_indices = tree_model.feature
            thresholds = tree_model.threshold
            children_left = tree_model.children_left
            children_right = tree_model.children_right

            # Vectorized traversal: route ALL samples simultaneously
            nodes = np.zeros(n_samples, dtype=np.intp)
            depth = 0

            while True:
                current_features = feat_indices[nodes]
                # Identify samples still at internal nodes (feature >= 0)
                active = current_features >= 0
                if not active.any():
                    break

                active_idx = np.where(active)[0]
                active_nodes = nodes[active_idx]
                active_feats = current_features[active_idx]

                # Record contribution: weight by inverse depth
                np.add.at(contributions, (active_idx, active_feats), 1.0 / (depth + 1))

                # Route left or right
                vals = X[active_idx, active_feats]
                thresh = thresholds[active_nodes]
                go_left = vals <= thresh

                new_nodes = np.where(go_left,
                                     children_left[active_nodes],
                                     children_right[active_nodes])
                nodes[active_idx] = new_nodes
                depth += 1

        # Normalize per sample to sum to 1
        row_sums = contributions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        contributions = contributions / row_sums

        return contributions
