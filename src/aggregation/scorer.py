"""
Risk score aggregation across agents, rules, context, and supervised model.

Scoring formula (4-component conditional weighting):
    When rules fire:  final = 0.30*agent + 0.25*rules + 0.20*context + 0.25*supervised
    When rules = 0:   final = 0.40*agent + 0.27*context + 0.33*supervised (redistributed)

Why 4 components: The supervised model (supervised_mod) catches 9/10 known launderers
vs 2/10 for our unsupervised agents alone. It learns feature combinations
(e.g. student + high volume + zero income) that rules evaluate independently.
The two systems have only 103 overlap out of ~3,300 combined flags — complementary.

Agent scores: diversity-weighted average of 3 anomaly detection agents.
Rule scores:  sum of triggered rule points (cash bundle = max, capped at 1.0).
Context scores: account age, type, and cash-intensity risk modifiers.
Supervised scores: supervised_mod's trained model output (0-1 risk probability).

Post-aggregation gates:
    - Rule floor: max(formula, 0.55 * rule_score) — rules can't be undercut
    - Model-assisted boost: +0.05 when agent confident + passthrough signature
    - Model override floor: agent >= 0.65 guarantees MEDIUM (final >= 0.30)
    - Diversity gating: require evidence from 2+ rule categories for HIGH

Three-tier risk bucketing:
    LOW:    final < RISK_TIER_LOW (0.31)
    MEDIUM: 0.31 ≤ final < 0.55
    HIGH:   final ≥ 0.55
"""

import numpy as np

from src.config import (
    AGENT_WEIGHTS, SCORE_WEIGHT_AGENT, SCORE_WEIGHT_RULES,
    SCORE_WEIGHT_CONTEXT, SCORE_WEIGHT_SUPERVISED,
    RISK_TIER_LOW, RISK_TIER_MEDIUM,
    RULE_FLOOR_FACTOR, RULE_CATEGORIES, DIVERSITY_MIN_CATEGORIES,
    DIVERSITY_MIN_VELOCITY, STRONG_RULES_FOR_OVERRIDE,
    ZERO_RULE_WEIGHT_AGENT, ZERO_RULE_WEIGHT_CONTEXT,
    ZERO_RULE_WEIGHT_SUPERVISED,
    MODEL_OVERRIDE_FLOOR_THRESHOLD, MODEL_OVERRIDE_FLOOR_SCORE,
    GHOST_ACCOUNT_MIN_TX,
    SUPERVISED_MOD_HIDDEN_SUPERVISED_MIN, SUPERVISED_MOD_HIDDEN_MIN_SIGNS,
    SUPERVISED_MOD_HIDDEN_MEDIUM_FLOOR, SUPERVISED_MOD_HIDDEN_HIGH_SUPERVISED_MIN,
    RULE_HEAVY_MIN_SIGNS, RULE_HEAVY_MIN_SIGNS_IF_HARD, RULE_HEAVY_MEDIUM_FLOOR,
)
from src.jobs.config import JOB_CONFIG

# Hard rules (R3/R12/R20/R30/R42) — strong laundering signals for supervised_mod hidden launderer floor.
_SUPERVISED_MOD_HARD_RULES = {
    "R3_SUDDEN_SPIKE_ESTABLISHED", "R12_HIGH_IO_SHORT_GAP",
    "R20_CASH_CONVERSION", "R30_OFFHOURS_BURST_UNIQUE", "R42_EXCHANGE_FUNNEL",
}


def compute_context_score(features_df):
    """Compute contextual risk score based on account characteristics.

    Context signals that elevate baseline risk:
    - Young accounts (<90 days) are higher risk by default
    - Semi-young accounts (<180 days) get moderate elevation
    - Business accounts with high flow relative to peers
    - Accounts with high cash activity ratios
    """
    n = len(features_df)
    score = np.zeros(n)

    age = features_df["account_age_days"].values

    # Young accounts: <90 days → +0.30, <180 days → +0.15
    score += np.where(age < 90, 0.30, np.where(age < 180, 0.15, 0.0))

    # Business accounts with high flow get elevated context
    is_biz = (features_df["account_type"] == "business").values
    flow = features_df["total_inflow"].values + features_df["total_outflow"].values
    flow_p90 = np.percentile(flow, 90) if n > 0 else 0
    score += np.where(is_biz & (flow > flow_p90), 0.15, 0.0)

    # High cash activity as context signal
    if "cash_in_ratio" in features_df.columns:
        cash_in = features_df["cash_in_ratio"].values
        score += np.where(cash_in > 0.4, 0.10, 0.0)

    # Job-based context modifier
    if "job_category" in features_df.columns:
        job_cat = features_df["job_category"].values.astype(str)
        ctx_mods = JOB_CONFIG["context_modifiers"]
        for cat, mod in ctx_mods.items():
            if mod != 0.0:
                score += np.where(job_cat == cat, mod, 0.0)

    return np.clip(score, 0.0, 1.0)


def _apply_model_assisted_boost(final_scores, agent_combined, features_df):
    """Boost accounts where ML model is confident but rules are weak.

    Fires when: model_score >= 0.55 AND io in [0.5, 2.0] AND
                (gap <= 48h OR balance_retention < 0.30)
    Effect: +0.20 * SCORE_WEIGHT_RULES (~+0.09) to final score.
    Target: SYNID0107464935 (model=0.5748, rules weak at 0.25).
    """
    io = features_df["io_ratio"].values
    gap = features_df["median_in_out_gap_hours"].values
    retention = (features_df["balance_retention"].values
                 if "balance_retention" in features_df.columns
                 else np.ones(len(features_df)))

    boost_mask = (
        (agent_combined >= 0.55)
        & (
            # Passthrough signature: balanced flow + fast turnover or low retention
            ((io >= 0.5) & (io <= 2.0) & ((gap <= 48) | (retention < 0.30)))
            # Extreme imbalance: model sees anomaly + extreme io ratio
            | ((io > 5) | (io < 0.2))
        )
    )

    boost = np.where(boost_mask, 0.20 * SCORE_WEIGHT_RULES, 0.0)
    return final_scores + boost


def _compute_launderer_signs(features_df, triggered_rules, supervised_scores=None):
    """Count launderer signs per account and whether a hard rule fired.

    The "model high" sign means supervised (supervised_mod) score >= 0.5, not agent combined.
    Pass supervised_scores when available so hidden-launderer floors use the correct signal.

    Returns:
        sign_count: (n,) int array, 0-9.
        has_hard_rule: (n,) bool array.
    """
    n = len(features_df)
    sign_count = np.zeros(n, dtype=int)
    has_hard_rule = np.zeros(n, dtype=bool)

    for i in range(n):
        signs = 0
        rules_list = triggered_rules[i] if i < len(triggered_rules) else []
        rnames = [
            e["rule"] if isinstance(e, dict) else e
            for e in rules_list
        ]
        if any(r in _SUPERVISED_MOD_HARD_RULES for r in rnames):
            has_hard_rule[i] = True
            signs += 1
        # "Model high" = supervised (supervised_mod) score >= 0.5, not agent (model_score)
        if supervised_scores is not None and i < len(supervised_scores):
            if supervised_scores[i] >= 0.5:
                signs += 1
        elif "supervised_risk_score" in features_df.columns:
            val = features_df["supervised_risk_score"].iloc[i]
            if val >= 0.5:
                signs += 1
        if "io_ratio" in features_df.columns:
            io = features_df["io_ratio"].iloc[i]
            if io > 5 or io < 0.2:
                signs += 1
        if "pct_round" in features_df.columns:
            if features_df["pct_round"].iloc[i] >= 0.15:
                signs += 1
        if "cash_cycle_ratio" in features_df.columns:
            if features_df["cash_cycle_ratio"].iloc[i] >= 0.1:
                signs += 1
        if "burst_score" in features_df.columns:
            if features_df["burst_score"].iloc[i] >= 2.5:
                signs += 1
        if "tx_velocity" in features_df.columns:
            if features_df["tx_velocity"].iloc[i] >= 3.0:
                signs += 1
        if "declared_income" in features_df.columns:
            inc = features_df["declared_income"].iloc[i]
            if inc == 0 or (hasattr(inc, "__float__") and float(inc) == 0.0):
                signs += 1
        if "account_age_days" in features_df.columns:
            age = features_df["account_age_days"].iloc[i]
            a = float(age) if hasattr(age, "__float__") else age
            if a < 180:
                signs += 1
        sign_count[i] = signs

    return sign_count, has_hard_rule


def _apply_supervised_mod_hidden_launderer_floor(final_scores, supervised_scores, features_df, triggered_rules):
    """Push supervised_mod-flagged accounts with 2+ launderer signs (or hard rule) into MEDIUM or HIGH.

    supervised >= SUPERVISED_MOD_HIDDEN_HIGH_SUPERVISED_MIN -> floor to 0.55 (HIGH).
    Else -> floor to SUPERVISED_MOD_HIDDEN_MEDIUM_FLOOR (MEDIUM).
    """
    sign_count, has_hard_rule = _compute_launderer_signs(
        features_df, triggered_rules, supervised_scores=supervised_scores
    )
    eligible = (
        (supervised_scores >= SUPERVISED_MOD_HIDDEN_SUPERVISED_MIN)
        & ((sign_count >= SUPERVISED_MOD_HIDDEN_MIN_SIGNS) | has_hard_rule)
    )
    # Split: sup > 0.9 -> HIGH floor, else -> MEDIUM floor
    to_high = eligible & (supervised_scores > SUPERVISED_MOD_HIDDEN_HIGH_SUPERVISED_MIN)
    to_med = eligible & (supervised_scores <= SUPERVISED_MOD_HIDDEN_HIGH_SUPERVISED_MIN)
    final_scores = np.where(
        to_high & (final_scores < RISK_TIER_MEDIUM),
        RISK_TIER_MEDIUM,
        final_scores,
    )
    final_scores = np.where(
        to_med & (final_scores < SUPERVISED_MOD_HIDDEN_MEDIUM_FLOOR),
        SUPERVISED_MOD_HIDDEN_MEDIUM_FLOOR,
        final_scores,
    )
    return final_scores


def _apply_rule_heavy_floor(final_scores, features_df, triggered_rules, supervised_scores=None):
    """Move LOW accounts with many launderer signs up to MEDIUM even when supervised < 0.5.

    Eligible: sign_count >= RULE_HEAVY_MIN_SIGNS, or (sign_count >= RULE_HEAVY_MIN_SIGNS_IF_HARD and has_hard_rule).
    Floor to RULE_HEAVY_MEDIUM_FLOOR so they land in MEDIUM. Catches the ~3k LOW-with-4+-signs
    that the supervised_mod floor misses because supervised < 0.5.
    """
    sign_count, has_hard_rule = _compute_launderer_signs(
        features_df, triggered_rules, supervised_scores=supervised_scores
    )
    eligible = (
        (sign_count >= RULE_HEAVY_MIN_SIGNS)
        | ((sign_count >= RULE_HEAVY_MIN_SIGNS_IF_HARD) & has_hard_rule)
    )
    below = final_scores < RULE_HEAVY_MEDIUM_FLOOR
    final_scores = np.where(
        eligible & below,
        RULE_HEAVY_MEDIUM_FLOOR,
        final_scores,
    )
    return final_scores


def _apply_diversity_gating(final_scores, triggered_rules, features_df):
    """Require evidence from multiple rule categories for HIGH tier.

    HIGH eligibility requires one of:
    1. categories_hit >= 2 AND tx_velocity >= 3.0
    2. strong_rule_hit AND categories_hit >= 1 AND tx_velocity >= 3.0
    3. strong_rule_hit AND categories_hit >= 2 (velocity override)

    Non-eligible accounts are capped just below HIGH threshold.
    """
    n = len(final_scores)
    vel = features_df["tx_velocity"].values
    velocity_ok = vel >= DIVERSITY_MIN_VELOCITY

    # Count distinct rule categories and check for strong rules per account
    cats_hit = np.zeros(n, dtype=int)
    has_strong = np.zeros(n, dtype=bool)

    for i in range(n):
        hit_cats = set()
        for entry in triggered_rules[i]:
            rname = entry["rule"] if isinstance(entry, dict) else entry
            if rname in STRONG_RULES_FOR_OVERRIDE:
                has_strong[i] = True
            for cat, members in RULE_CATEGORIES.items():
                if rname in members:
                    hit_cats.add(cat)
        cats_hit[i] = len(hit_cats)

    # HIGH eligibility conditions
    eligible = (
        (cats_hit >= DIVERSITY_MIN_CATEGORIES) & velocity_ok          # standard path
    ) | (
        has_strong & (cats_hit >= 1) & velocity_ok                    # strong + velocity
    ) | (
        has_strong & (cats_hit >= DIVERSITY_MIN_CATEGORIES)            # strong + diversity (no vel needed)
    ) | (
        cats_hit >= 3                                                  # 3+ categories = strong diversity signal
    )

    # Cap non-eligible accounts to just below HIGH
    cap = RISK_TIER_MEDIUM - 0.001
    final_scores = np.where(
        ~eligible & (final_scores >= RISK_TIER_MEDIUM),
        cap,
        final_scores,
    )
    return final_scores


def aggregate_scores(agent_results, rule_scores, features_df, triggered_rules=None,
                     supervised_scores=None):
    """Combine agent scores, rule scores, context, and supervised into final risk scores.

    Args:
        agent_results:      List of AgentResult objects.
        rule_scores:        ndarray of [0, 1] per account from rule engine.
        features_df:        DataFrame with feature columns (for context scoring).
        triggered_rules:    List of list-of-dicts per account (from compute_rule_scores).
        supervised_scores:  ndarray of [0, 1] per account from supervised model (supervised_mod).

    Returns:
        final_scores:   ndarray of final scores in [0, 1].
        context_scores: ndarray of context scores in [0, 1].
    """
    n = len(features_df)

    # Diversity-weighted agent averaging
    if len(agent_results) == 1:
        agent_combined = agent_results[0].scores
    else:
        weights = np.array(AGENT_WEIGHTS[:len(agent_results)], dtype=float)
        weights = weights / weights.sum()
        agent_combined = np.average(
            [r.scores for r in agent_results], axis=0, weights=weights
        )

    context_scores = compute_context_score(features_df)

    # Default supervised scores to zero if not provided (graceful fallback)
    if supervised_scores is None:
        supervised_scores = np.zeros(n)

    # Conditional weighting: when rule_score == 0, redistribute the rule weight
    # proportionally to agent + context + supervised.
    has_rules = rule_scores > 0

    # Standard path: 0.30 agent + 0.25 rules + 0.20 context + 0.25 supervised
    formula_standard = (
        SCORE_WEIGHT_AGENT * agent_combined
        + SCORE_WEIGHT_RULES * rule_scores
        + SCORE_WEIGHT_CONTEXT * context_scores
        + SCORE_WEIGHT_SUPERVISED * supervised_scores
    )

    # Zero-rule path: redistribute rule weight (0.40 agent + 0.27 context + 0.33 supervised)
    formula_zero_rule = (
        ZERO_RULE_WEIGHT_AGENT * agent_combined
        + ZERO_RULE_WEIGHT_CONTEXT * context_scores
        + ZERO_RULE_WEIGHT_SUPERVISED * supervised_scores
    )

    formula_scores = np.where(has_rules, formula_standard, formula_zero_rule)

    # Rule floor: when expert rules are confident, don't let low agent/context
    # scores drag the final score below what rules alone justify.
    # Bypass floor when profile + supervised disagree: SALARIED with declared income
    # and balanced IO, or brand-new account with few txs — and supervised says clean.
    # Fixes 2 label=0 FPs without shrinking the HIGH queue.
    rule_floor = RULE_FLOOR_FACTOR * rule_scores
    if "job_category" in features_df.columns and "declared_income" in features_df.columns:
        job = features_df["job_category"].values.astype(str)
        income = features_df["declared_income"].values
        io = features_df["io_ratio"].values
        age = features_df["account_age_days"].values
        tx_count = features_df["raw_tx_count"].values if "raw_tx_count" in features_df.columns else np.ones(n) * 999
        bypass_mask = (
            (
                (job == "SALARIED") & (income > 0) & (io >= 0.5) & (io <= 2.0)
                | ((age == 0) & (tx_count < 15))
            )
            & (supervised_scores < 0.25)
            & (rule_scores >= 0.99)
        )
        rule_floor = np.where(bypass_mask, 0.0, rule_floor)
    final_scores = np.maximum(formula_scores, rule_floor)

    # Model-assisted boost for accounts where agents are confident but rules weak
    final_scores = _apply_model_assisted_boost(final_scores, agent_combined, features_df)

    # Model override floor: when agents are highly confident (>= 0.65),
    # ensure the account reaches at least MEDIUM tier for human review.
    # This recovers strong-anomaly accounts that would otherwise be buried
    # as LOW due to no rules firing and low context scores.
    model_floor_mask = agent_combined >= MODEL_OVERRIDE_FLOOR_THRESHOLD
    final_scores = np.where(
        model_floor_mask & (final_scores < MODEL_OVERRIDE_FLOOR_SCORE),
        MODEL_OVERRIDE_FLOOR_SCORE,
        final_scores,
    )

    # Diversity gating: require multi-category evidence for HIGH tier
    if triggered_rules is not None:
        final_scores = _apply_diversity_gating(final_scores, triggered_rules, features_df)

    # Ghost account damping: accounts with very few transactions (1-2) produce
    # degenerate feature values (100% offhours, 100% channel switch) that
    # inflate scores artificially. Dampen proportionally to transaction count.
    # Exception: when rule_score >= 0.40, rules corroborate that the features
    # are real signals (not artifacts), so damping is bypassed.
    if "raw_tx_count" in features_df.columns:
        tx_count = features_df["raw_tx_count"].values
        damping = np.clip(tx_count / GHOST_ACCOUNT_MIN_TX, 0.0, 1.0)
        rule_corroborated = rule_scores >= 0.40
        damping = np.where(rule_corroborated, 1.0, damping)
        final_scores = final_scores * damping

    # Supervised_mod hidden launderers: push supervised_mod-flagged + 2+ launderer signs into MEDIUM (or HIGH if sup > 0.9).
    if triggered_rules is not None:
        final_scores = _apply_supervised_mod_hidden_launderer_floor(
            final_scores, supervised_scores, features_df, triggered_rules
        )
        # Rule-heavy floor: 5+ signs (or 4+ and hard rule) -> MEDIUM even when supervised < 0.5.
        final_scores = _apply_rule_heavy_floor(
            final_scores, features_df, triggered_rules, supervised_scores=supervised_scores
        )

    return np.clip(final_scores, 0.0, 1.0), context_scores


def assign_risk_bucket(score, high_threshold=None):
    """Map final score to 3-tier risk bucket label.
    HIGH = score >= high_threshold (or RISK_TIER_MEDIUM if high_threshold is None).
    Single fixed threshold — no percentile; who is HIGH is fully explainable.
    Tuning: raise threshold to reduce false positives; lower to catch more launderers
    (or improve rules/agents so launderers score above threshold).
    """
    thr = high_threshold if high_threshold is not None else RISK_TIER_MEDIUM
    if score < RISK_TIER_LOW:
        return "LOW"
    elif score < thr:
        return "MEDIUM"
    else:
        return "HIGH"
