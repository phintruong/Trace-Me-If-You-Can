"""Tests for risk score aggregation, context scoring, and bucket assignment."""

import numpy as np
import pandas as pd
import pytest

from src.agents.base import AgentResult
from src.aggregation.scorer import aggregate_scores, assign_risk_bucket, compute_context_score
from src.config import (
    SCORE_WEIGHT_AGENT, SCORE_WEIGHT_RULES, SCORE_WEIGHT_CONTEXT,
    ZERO_RULE_WEIGHT_AGENT, ZERO_RULE_WEIGHT_CONTEXT,
    MODEL_OVERRIDE_FLOOR_THRESHOLD, MODEL_OVERRIDE_FLOOR_SCORE,
    RULE_CATEGORIES, STRONG_RULES_FOR_OVERRIDE, GHOST_ACCOUNT_MIN_TX,
)
from src.rules.aml_rules import compute_rule_scores


def _make_agent_result(scores, name="test_agent"):
    """Helper to build an AgentResult with dummy explanations."""
    return AgentResult(
        agent_name=name,
        scores=np.array(scores, dtype=float),
        explanations=np.zeros((len(scores), 1)),
    )


def _make_features_df(n, account_age_days=365, account_type="individual",
                       total_inflow=1000, total_outflow=1000, cash_in_ratio=0.0,
                       io_ratio=1.0, median_in_out_gap_hours=100,
                       balance_retention=0.5, tx_velocity=1.0):
    """Helper to build a minimal features DataFrame for context scoring."""
    return pd.DataFrame({
        "account_age_days": [account_age_days] * n,
        "account_type": [account_type] * n,
        "total_inflow": [total_inflow] * n,
        "total_outflow": [total_outflow] * n,
        "cash_in_ratio": [cash_in_ratio] * n,
        "io_ratio": [io_ratio] * n,
        "median_in_out_gap_hours": [median_in_out_gap_hours] * n,
        "balance_retention": [balance_retention] * n,
        "tx_velocity": [tx_velocity] * n,
    })


class TestWeightedFormula:
    """Verify the weighted scoring formula with conditional weighting."""

    def test_formula_components(self):
        """Final score should equal weighted sum when rules > 0."""
        agent = _make_agent_result([0.6])
        rule_scores = np.array([0.5])
        features_df = _make_features_df(1)
        final, context = aggregate_scores([agent], rule_scores, features_df)
        expected = (SCORE_WEIGHT_AGENT * 0.6
                    + SCORE_WEIGHT_RULES * 0.5
                    + SCORE_WEIGHT_CONTEXT * context[0])
        # Rule floor or model boost may push higher, so check >=
        assert final[0] >= expected - 1e-6, (
            f"Expected >= {expected:.6f}, got {final[0]:.6f}"
        )

    def test_scores_bounded_0_1(self):
        """All final scores must remain in [0, 1]."""
        agent = _make_agent_result([0.0, 0.5, 1.0])
        rule_scores = np.array([0.0, 0.5, 1.0])
        features_df = _make_features_df(3)
        final, _ = aggregate_scores([agent], rule_scores, features_df)
        assert np.all(final >= 0.0) and np.all(final <= 1.0)

    def test_monotonic_in_rule_score(self):
        """Higher rule scores should produce higher final scores (agents/context held constant)."""
        agent = _make_agent_result([0.4] * 5)
        rule_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        features_df = _make_features_df(5)
        final, _ = aggregate_scores([agent], rule_scores, features_df)
        for i in range(len(final) - 1):
            assert final[i] < final[i + 1], (
                f"final[{i}]={final[i]:.4f} >= final[{i+1}]={final[i+1]:.4f}"
            )

    def test_monotonic_in_agent_score(self):
        """Higher agent scores should produce higher final scores."""
        agents = [_make_agent_result([v]) for v in [0.1, 0.3, 0.5, 0.7, 0.9]]
        features_df = _make_features_df(1)
        finals = []
        for a in agents:
            f, _ = aggregate_scores([a], np.array([0.3]), features_df)
            finals.append(f[0])
        for i in range(len(finals) - 1):
            assert finals[i] < finals[i + 1]

    def test_multi_agent_weighted_average(self):
        """Multiple agents should be diversity-weighted before blending."""
        a1 = _make_agent_result([0.8], name="a1")
        a2 = _make_agent_result([0.2], name="a2")
        rule_scores = np.array([0.0])
        features_df = _make_features_df(1)
        final, _ = aggregate_scores([a1, a2], rule_scores, features_df)
        # With 2 agents, weights are normalized subset of AGENT_WEIGHTS
        # final = 0.4 * weighted_avg + 0.4 * 0.0 + 0.2 * context
        assert final[0] > 0.0


class TestContextScore:
    """Verify context score behavior."""

    def test_young_account_high_context(self):
        """Accounts <90 days should get +0.30 context."""
        df = _make_features_df(1, account_age_days=30)
        ctx = compute_context_score(df)
        assert ctx[0] >= 0.30

    def test_semi_young_account_moderate_context(self):
        """Accounts 90-180 days should get +0.15 context."""
        df = _make_features_df(1, account_age_days=120)
        ctx = compute_context_score(df)
        assert ctx[0] >= 0.15
        assert ctx[0] < 0.30  # not as high as young

    def test_old_account_low_context(self):
        """Accounts >180 days with no other flags should get 0 context."""
        df = _make_features_df(1, account_age_days=365)
        ctx = compute_context_score(df)
        assert ctx[0] < 0.15

    def test_high_cash_adds_context(self):
        """High cash_in_ratio should add context score."""
        df = _make_features_df(1, account_age_days=365, cash_in_ratio=0.5)
        ctx = compute_context_score(df)
        assert ctx[0] >= 0.10

    def test_context_bounded_0_1(self):
        """Context scores must be in [0, 1]."""
        df = _make_features_df(1, account_age_days=10, cash_in_ratio=0.9)
        ctx = compute_context_score(df)
        assert 0.0 <= ctx[0] <= 1.0


class TestBucketThresholds:
    """Verify 3-tier risk bucket system."""

    def test_low_below_030(self):
        assert assign_risk_bucket(0.29) == "LOW"

    def test_low_at_zero(self):
        assert assign_risk_bucket(0.0) == "LOW"

    def test_medium_at_030(self):
        assert assign_risk_bucket(0.30) == "MEDIUM"

    def test_medium_below_055(self):
        assert assign_risk_bucket(0.54) == "MEDIUM"

    def test_high_at_058(self):
        """Score at HIGH threshold (0.58) should be HIGH."""
        assert assign_risk_bucket(0.58) == "HIGH"

    def test_high_at_1(self):
        assert assign_risk_bucket(1.0) == "HIGH"

    def test_high_threshold_override(self):
        """Custom high_threshold changes MEDIUM/HIGH boundary."""
        assert assign_risk_bucket(0.50, high_threshold=0.45) == "HIGH"
        assert assign_risk_bucket(0.50, high_threshold=0.55) == "MEDIUM"
        assert assign_risk_bucket(0.50, high_threshold=None) == "MEDIUM"


class TestConditionalWeighting:
    """Verify zero-rule weight redistribution."""

    def test_zero_rules_uses_redistributed_weights(self):
        """When rule_score=0, should use redistributed weights (0.57 agent + 0.43 context)."""
        agent = _make_agent_result([0.6])
        rule_scores = np.array([0.0])
        features_df = _make_features_df(1)
        final, context = aggregate_scores([agent], rule_scores, features_df)
        expected = ZERO_RULE_WEIGHT_AGENT * 0.6 + ZERO_RULE_WEIGHT_CONTEXT * context[0]
        # Model override floor may push higher if agent >= 0.65
        assert final[0] >= expected - 1e-6, (
            f"Expected >= {expected:.6f}, got {final[0]:.6f}"
        )

    def test_nonzero_rules_uses_standard_weights(self):
        """When rule_score>0, should use standard weights (0.40/0.30/0.30)."""
        agent = _make_agent_result([0.6])
        rule_scores = np.array([0.5])
        features_df = _make_features_df(1)
        final, context = aggregate_scores([agent], rule_scores, features_df)
        expected = (SCORE_WEIGHT_AGENT * 0.6
                    + SCORE_WEIGHT_RULES * 0.5
                    + SCORE_WEIGHT_CONTEXT * context[0])
        assert final[0] >= expected - 1e-6, (
            f"Expected >= {expected:.6f}, got {final[0]:.6f}"
        )

    def test_zero_rule_redistribution_produces_higher_scores(self):
        """Zero-rule redistribution should score higher than static weights with rules=0."""
        agent = _make_agent_result([0.7])
        rule_scores = np.array([0.0])
        features_df = _make_features_df(1)
        final, context = aggregate_scores([agent], rule_scores, features_df)
        # Static would give: 0.40*0.7 + 0.30*0.0 + 0.30*ctx = 0.28 + 0.30*ctx
        static_score = SCORE_WEIGHT_AGENT * 0.7 + SCORE_WEIGHT_RULES * 0.0 + SCORE_WEIGHT_CONTEXT * context[0]
        assert final[0] > static_score, (
            f"Redistributed {final[0]:.6f} should exceed static {static_score:.6f}"
        )

    def test_mixed_rules_array(self):
        """Verify correct per-account conditional weighting in a mixed array."""
        agent = _make_agent_result([0.5, 0.5])
        rule_scores = np.array([0.0, 0.4])
        features_df = _make_features_df(2)
        final, context = aggregate_scores([agent], rule_scores, features_df)
        # Account 0 (no rules): redistributed weights
        # Account 1 (has rules): standard weights
        assert final[0] != final[1], "Conditional weighting should produce different scores"


class TestModelOverrideFloor:
    """Verify model override floor behavior."""

    def test_high_agent_guarantees_medium(self):
        """Agent score >= 0.65 should guarantee final >= 0.30 (MEDIUM)."""
        agent = _make_agent_result([0.70])
        rule_scores = np.array([0.0])
        # Old account, no cash — context near 0
        features_df = _make_features_df(1, account_age_days=365)
        final, _ = aggregate_scores([agent], rule_scores, features_df)
        assert final[0] >= MODEL_OVERRIDE_FLOOR_SCORE, (
            f"Expected >= {MODEL_OVERRIDE_FLOOR_SCORE}, got {final[0]:.6f}"
        )

    def test_low_agent_no_floor(self):
        """Agent score < 0.65 should not get the floor boost."""
        agent = _make_agent_result([0.20])
        rule_scores = np.array([0.0])
        # Old account, no cash — context near 0
        features_df = _make_features_df(1, account_age_days=365)
        final, _ = aggregate_scores([agent], rule_scores, features_df)
        # 0.57*0.20 = 0.114 + ~0 context = well below 0.30
        assert final[0] < MODEL_OVERRIDE_FLOOR_SCORE, (
            f"Expected < {MODEL_OVERRIDE_FLOOR_SCORE}, got {final[0]:.6f}"
        )

    def test_floor_does_not_reduce_high_scores(self):
        """The floor should never reduce scores that are already above it."""
        agent = _make_agent_result([0.80])
        rule_scores = np.array([0.5])
        features_df = _make_features_df(1)
        final, context = aggregate_scores([agent], rule_scores, features_df)
        base = (SCORE_WEIGHT_AGENT * 0.80
                + SCORE_WEIGHT_RULES * 0.5
                + SCORE_WEIGHT_CONTEXT * context[0])
        assert final[0] >= base - 1e-6


# ---------------------------------------------------------------------------
# Helper for rule-level tests — builds a full features DataFrame
# ---------------------------------------------------------------------------
def _make_rule_features_df(n=100, **overrides):
    """Build a feature DataFrame for rule testing with all required columns."""
    rng = np.random.RandomState(42)
    base = {
        "total_inflow": rng.uniform(500, 50000, n),
        "total_outflow": rng.uniform(500, 50000, n),
        "io_ratio": rng.uniform(0.5, 2.0, n),
        "tx_velocity": rng.uniform(0.1, 5.0, n),
        "avg_tx_size": rng.uniform(50, 5000, n),
        "std_tx_size": rng.uniform(10, 1000, n),
        "cv_tx_size": rng.uniform(0.1, 2.0, n),
        "median_in_out_gap_hours": rng.uniform(10, 500, n),
        "pct_round": rng.uniform(0, 0.3, n),
        "repeat_amt_ratio": rng.uniform(0, 0.2, n),
        "burst_score": rng.uniform(1.0, 3.0, n),
        "max_tx_fraction": rng.uniform(0.05, 0.5, n),
        "offhours_ratio": rng.uniform(0, 0.3, n),
        "account_age_days": rng.uniform(30, 1000, n),
        "person_age_years": rng.uniform(20, 70, n),
        "flow_per_account_day": rng.uniform(1, 100, n),
        "flow_acceleration": rng.uniform(0.5, 2.0, n),
        "cash_in_total": rng.uniform(0, 5000, n),
        "cash_out_total": rng.uniform(0, 5000, n),
        "cash_in_ratio": rng.uniform(0, 0.3, n),
        "cash_out_ratio": rng.uniform(0, 0.3, n),
        "cash_cycle_ratio": rng.uniform(0, 0.2, n),
        "channel_switch_ratio": rng.uniform(0, 0.3, n),
        "retail_ratio": rng.uniform(0.1, 0.8, n),
        "distinct_provinces": rng.randint(1, 4, n),
        "atm_in_total": rng.uniform(0, 3000, n),
        "card_out_total": rng.uniform(0, 3000, n),
        "channel_loop_count": rng.randint(0, 50, n),
        "balance_retention": rng.uniform(0.1, 0.8, n),
        "merchant_concentration": rng.uniform(0, 0.5, n),
        "sustained_passthrough_count": rng.randint(0, 2, n),
        "wire_tx_count": rng.randint(0, 5, n),
        "wire_total_volume": rng.uniform(0, 5000, n),
        "wire_velocity": rng.uniform(0, 0.1, n),
        "wire_volume_ratio": rng.uniform(0, 0.2, n),
        "wire_max_weekly_count": rng.randint(0, 2, n),
        "job_category": np.full(n, "SALARIED"),
        "account_type": np.full(n, "individual"),
        # New features
        "inactive_days": np.zeros(n),
        "exchange_outflow_ratio": rng.uniform(0, 0.2, n),
        "raw_tx_count": rng.randint(5, 50, n),
        "declared_income": rng.uniform(30000, 80000, n),
        "balance_cycles": rng.randint(0, 3, n),
        "outflow_count": rng.randint(2, 20, n),
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestNewRules:
    """Verify new rules R39, R41-R44 fire correctly."""

    def test_r39_fires_on_dormant_reactivation(self):
        """R39 should fire when inactive>180d, high flow, high velocity."""
        df = _make_rule_features_df(100)
        # Set account 0 as dormant reactivation
        df.loc[0, "inactive_days"] = 200
        df.loc[0, "total_inflow"] = 100000  # very high flow
        df.loc[0, "total_outflow"] = 100000
        df.loc[0, "tx_velocity"] = 10.0  # very high velocity
        scores, triggers = compute_rule_scores(df)
        r39_fired = any(t["rule"] == "R39_DORMANT_REACTIVATION" for t in triggers[0])
        assert r39_fired, "R39 should fire on dormant account with high reactivation"

    def test_r39_skips_active_account(self):
        """R39 should not fire on recently active accounts."""
        df = _make_rule_features_df(100)
        df.loc[0, "inactive_days"] = 10  # recently active
        df.loc[0, "total_inflow"] = 100000
        df.loc[0, "total_outflow"] = 100000
        df.loc[0, "tx_velocity"] = 10.0
        _, triggers = compute_rule_scores(df)
        r39_fired = any(t["rule"] == "R39_DORMANT_REACTIVATION" for t in triggers[0])
        assert not r39_fired, "R39 should not fire on active accounts"

    def test_r41_micro_structuring(self):
        """R41 should fire on high tx count + low avg size + significant flow."""
        df = _make_rule_features_df(100)
        df.loc[0, "raw_tx_count"] = 500  # very high count
        df.loc[0, "avg_tx_size"] = 10.0  # very small average
        df.loc[0, "total_inflow"] = 80000
        df.loc[0, "total_outflow"] = 80000
        _, triggers = compute_rule_scores(df)
        r41_fired = any(t["rule"] == "R41_MICRO_STRUCTURING" for t in triggers[0])
        assert r41_fired, "R41 should fire on micro-structuring pattern"

    def test_r42_requires_passthrough(self):
        """R42 should not fire without passthrough even with high exchange ratio."""
        df = _make_rule_features_df(100)
        df.loc[0, "exchange_outflow_ratio"] = 0.60
        df.loc[0, "sustained_passthrough_count"] = 0  # no passthrough
        _, triggers = compute_rule_scores(df)
        r42_fired = any(t["rule"] == "R42_EXCHANGE_FUNNEL" for t in triggers[0])
        assert not r42_fired, "R42 should require passthrough count >= 2"

    def test_r42_fires_with_both_conditions(self):
        """R42 should fire with high exchange ratio + passthrough."""
        df = _make_rule_features_df(100)
        df.loc[0, "exchange_outflow_ratio"] = 0.60
        df.loc[0, "sustained_passthrough_count"] = 3
        _, triggers = compute_rule_scores(df)
        r42_fired = any(t["rule"] == "R42_EXCHANGE_FUNNEL" for t in triggers[0])
        assert r42_fired, "R42 should fire with exchange ratio + passthrough"

    def test_r43_skips_zero_income(self):
        """R43 should not fire when declared_income is 0 (no data/business)."""
        df = _make_rule_features_df(100)
        df.loc[0, "declared_income"] = 0
        df.loc[0, "total_inflow"] = 200000
        df.loc[0, "total_outflow"] = 200000
        _, triggers = compute_rule_scores(df)
        r43_fired = any(t["rule"] == "R43_INCOME_MISMATCH" for t in triggers[0])
        assert not r43_fired, "R43 must not fire on zero-income accounts"

    def test_r43_fires_on_low_income_high_flow(self):
        """R43 should fire when low income but very high flow."""
        df = _make_rule_features_df(100)
        df.loc[0, "declared_income"] = 5000  # very low income
        df.loc[0, "total_inflow"] = 500000  # extremely high flow
        df.loc[0, "total_outflow"] = 500000
        _, triggers = compute_rule_scores(df)
        r43_fired = any(t["rule"] == "R43_INCOME_MISMATCH" for t in triggers[0])
        assert r43_fired, "R43 should fire on low income + high flow"

    def test_r44_requires_both_conditions(self):
        """R44 should only fire with high cycles AND low retention."""
        df = _make_rule_features_df(100)
        # High cycles but high retention — should NOT fire
        df.loc[0, "balance_cycles"] = 10
        df.loc[0, "balance_retention"] = 0.50
        _, triggers = compute_rule_scores(df)
        r44_fired = any(t["rule"] == "R44_BALANCE_CYCLING" for t in triggers[0])
        assert not r44_fired, "R44 should not fire with high retention"

        # High cycles AND low retention — should fire
        df.loc[0, "balance_retention"] = 0.10
        _, triggers = compute_rule_scores(df)
        r44_fired = any(t["rule"] == "R44_BALANCE_CYCLING" for t in triggers[0])
        assert r44_fired, "R44 should fire with high cycles + low retention"

    def test_network_category_exists(self):
        """RULE_CATEGORIES should include NETWORK with R42."""
        assert "NETWORK" in RULE_CATEGORIES
        assert "R42_EXCHANGE_FUNNEL" in RULE_CATEGORIES["NETWORK"]

    def test_new_strong_rules_in_override_set(self):
        """R39, R42, R43 should be in STRONG_RULES_FOR_OVERRIDE."""
        for rule in ["R39_DORMANT_REACTIVATION", "R42_EXCHANGE_FUNNEL", "R43_INCOME_MISMATCH"]:
            assert rule in STRONG_RULES_FOR_OVERRIDE, f"{rule} missing from STRONG_RULES_FOR_OVERRIDE"


class TestGhostAccountDamping:
    """Verify ghost account damping reduces scores for low-tx accounts."""

    def test_single_tx_heavily_damped(self):
        """Account with 1 tx and low rules should be heavily damped."""
        agent = _make_agent_result([0.5])
        rule_scores = np.array([0.1])  # low rules — no corroboration exemption
        features_df = _make_features_df(1)
        features_df["raw_tx_count"] = 1
        final, _ = aggregate_scores([agent], rule_scores, features_df)
        # With damping: multiplied by 1/5 = 0.2
        assert final[0] < 0.10, f"Ghost account (1 tx, low rules) should be heavily damped, got {final[0]:.4f}"

    def test_ghost_damping_bypassed_when_rules_corroborate(self):
        """Ghost damping should be bypassed when rule_score >= 0.40."""
        agent = _make_agent_result([0.5])
        rule_scores = np.array([0.5])  # strong rules corroborate
        features_df = _make_features_df(1)
        features_df["raw_tx_count"] = 1  # would normally damp to 0.2x
        final, _ = aggregate_scores([agent], rule_scores, features_df)
        # No damping applied — should be at least the rule floor
        assert final[0] >= 0.25, f"Rule-corroborated ghost account should not be damped, got {final[0]:.4f}"

    def test_five_tx_no_damping(self):
        """Account with 5+ transactions should not be damped."""
        agent = _make_agent_result([0.5])
        rule_scores = np.array([0.5])
        features_df = _make_features_df(1)
        features_df["raw_tx_count"] = 10
        final_with, _ = aggregate_scores([agent], rule_scores, features_df)

        features_df2 = _make_features_df(1)
        features_df2["raw_tx_count"] = 50
        final_without, _ = aggregate_scores([agent], rule_scores, features_df2)

        assert abs(final_with[0] - final_without[0]) < 0.01, "5+ tx accounts should score the same"
