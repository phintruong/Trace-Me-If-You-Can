"""
Agent 2: Strict Isolation Forest anomaly detection.

Behaviorally distinct from Agent 1 by using stricter hyperparameters:
- Higher contamination (0.07 vs "auto") → flags more accounts as anomalous
- Larger max_samples (512 vs 256) → each tree sees more data, more stable splits
- More estimators (500 vs 300) → smoother score surface
- Different random_state (123 vs 42) → different tree structure / split boundaries

Same feature set, same scoring interface, same [0, 1] output range.
"""

import logging

import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    IF_STRICT_N_ESTIMATORS, IF_STRICT_MAX_SAMPLES,
    IF_STRICT_CONTAMINATION, IF_STRICT_RANDOM_STATE, FEATURE_COLUMNS,
)
from src.agents.base import BaseAgent, AgentResult

log = logging.getLogger("aml_pipeline")


class IsolationForestStrictAgent(BaseAgent):
    """Strict Isolation Forest anomaly detection agent.

    Uses higher contamination and more trees than Agent 1 to produce
    a stricter anomaly signal. The two agents together give the
    aggregation layer diversity in sensitivity.
    """

    @property
    def name(self):
        return "isolation_forest_strict"

    @property
    def version(self):
        return "1.0"

    def score(self, features_df):
        """Train strict Isolation Forest and return scored results."""
        X = features_df[FEATURE_COLUMNS].values
        n_samples = X.shape[0]

        log.info(f"[{self.name}] Feature matrix: {n_samples:,} samples x {X.shape[1]} features")
        log.info(f"[{self.name}] Training Isolation Forest (n_estimators={IF_STRICT_N_ESTIMATORS}, "
                 f"max_samples={min(IF_STRICT_MAX_SAMPLES, n_samples)}, "
                 f"contamination={IF_STRICT_CONTAMINATION})")

        model = IsolationForest(
            n_estimators=IF_STRICT_N_ESTIMATORS,
            max_samples=min(IF_STRICT_MAX_SAMPLES, n_samples),
            contamination=IF_STRICT_CONTAMINATION,
            random_state=IF_STRICT_RANDOM_STATE,
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

        Identical method to Agent 1 — same algorithm, different model weights.
        The strict model's different tree structure produces different
        contribution profiles even with the same computation.
        """
        n_features = X.shape[1]
        n_samples = X.shape[0]

        contributions = np.zeros((n_samples, n_features))

        for tree in tqdm(
            model.estimators_, desc="  Feature importance (strict)", unit="tree", mininterval=1.0
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
