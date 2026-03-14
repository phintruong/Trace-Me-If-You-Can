"""
Whitebox explainability report generation.

Generates fully transparent risk reports for flagged accounts.
Every number in the final score is traced back to the exact agent,
feature, or rule that produced it — no black boxes.

Pure string formatting — no data loading or computation.
"""

import numpy as np

from src.config import (
    FEATURE_COLUMNS, SPARSE_FEATURES,
    SCORE_WEIGHT_AGENT, SCORE_WEIGHT_RULES, SCORE_WEIGHT_CONTEXT,
    SCORE_WEIGHT_SUPERVISED, RISK_TIER_LOW, RISK_TIER_MEDIUM,
)


def generate_account_report(
    customer_id, features_row, agent_results, sample_index,
    rule_score, context_score, final_score, triggered_rules, feature_percentiles,
):
    """Generate a fully transparent risk report for one account.

    Args:
        customer_id:        Customer identifier string.
        features_row:       pandas Series with all feature values for this customer.
        agent_results:      List of AgentResult objects (all agents).
        sample_index:       Integer index of this customer in the score arrays.
        rule_score:         Float rule score for this customer.
        context_score:      Float context score for this customer.
        final_score:        Float final aggregated score.
        triggered_rules:    List of rule-trigger dicts for this customer.
        feature_percentiles: Dict mapping feature name → percentile rank (0–100).
    """
    bucket = _assign_bucket(final_score)
    n_agents = len(agent_results)
    agent_scores = [r.scores[sample_index] for r in agent_results]
    agent_mean = float(np.mean(agent_scores))

    lines = []

    # ── Header ──────────────────────────────────────────────────────────
    lines.append("=" * 76)
    lines.append(f"  AML RISK REPORT — {customer_id}")
    lines.append("=" * 76)
    lines.append("")

    # ── Section 1: Final Decision ───────────────────────────────────────
    lines.append("FINAL DECISION")
    lines.append(f"  Final Score:   {final_score:.4f}")
    lines.append(f"  Risk Tier:     {bucket}")
    lines.append("")

    # ── Section 2: Score Decomposition ──────────────────────────────────
    supervised_score = features_row.get("supervised_score", 0.0)
    lines.append("SCORE DECOMPOSITION")
    lines.append(f"  formula:  final = {SCORE_WEIGHT_AGENT}*agent + {SCORE_WEIGHT_RULES}*rules + {SCORE_WEIGHT_CONTEXT}*context + {SCORE_WEIGHT_SUPERVISED}*supervised")
    lines.append("")
    lines.append(f"  agent_combined (weighted avg of {n_agents} agents):  {agent_mean:.4f}")
    for r in agent_results:
        score = r.scores[sample_index]
        lines.append(f"    {r.agent_name:30s}  score = {score:.4f}")
    lines.append(f"  rule_score:                              {rule_score:.4f}")
    lines.append(f"  context_score:                           {context_score:.4f}")
    lines.append(f"  supervised_score (supervised_mod):                {supervised_score:.4f}")
    lines.append("")
    computed = (SCORE_WEIGHT_AGENT * agent_mean
                + SCORE_WEIGHT_RULES * rule_score
                + SCORE_WEIGHT_CONTEXT * context_score
                + SCORE_WEIGHT_SUPERVISED * supervised_score)
    lines.append(f"  calculation:  {SCORE_WEIGHT_AGENT}*{agent_mean:.4f} + "
                 f"{SCORE_WEIGHT_RULES}*{rule_score:.4f} + "
                 f"{SCORE_WEIGHT_CONTEXT}*{context_score:.4f} + "
                 f"{SCORE_WEIGHT_SUPERVISED}*{supervised_score:.4f} = {computed:.4f}")
    lines.append("")

    # ── Section 3: Account Info ─────────────────────────────────────────
    acct_type = features_row.get("account_type", "unknown")
    lines.append("ACCOUNT INFO")
    lines.append(f"  Account Type:   {acct_type.upper()}")
    lines.append(f"  Account Age:    {features_row['account_age_days']:.0f} days")
    person_age = features_row["person_age_years"]
    if acct_type == "individual" and person_age > 0:
        lines.append(f"  Person Age:     {person_age:.0f} years")
    elif acct_type == "business":
        lines.append(f"  Person Age:     N/A (business account)")
    else:
        lines.append(f"  Person Age:     N/A")
    lines.append(f"  Scored against: {acct_type} peer group")
    lines.append("")

    # Cash behavior summary
    cash_in = features_row.get("cash_in_ratio", 0)
    cash_out = features_row.get("cash_out_ratio", 0)
    cash_cycle = features_row.get("cash_cycle_ratio", 0)
    lines.append("CASH BEHAVIOR")
    lines.append(f"  Cash In Ratio:    {cash_in:.4f}")
    lines.append(f"  Cash Out Ratio:   {cash_out:.4f}")
    lines.append(f"  Cash Cycle Ratio: {cash_cycle:.4f}")
    lines.append("")

    # ── Section 4: Per-Agent Explanations ───────────────────────────────
    for r in agent_results:
        _render_agent_explanation(lines, r, sample_index, features_row, feature_percentiles)

    # ── Section 5: Triggered Rules (full detail) ────────────────────────
    lines.append("RULE ENGINE DETAIL")
    lines.append(f"  Total rule score: {rule_score:.4f}")
    if triggered_rules:
        for rule_info in triggered_rules:
            pt = rule_info["points"]
            sign = "+" if pt >= 0 else ""
            lines.append(
                f"    [{rule_info['rule']}]  {sign}{pt:.2f}  "
                f"— {rule_info['detail']}"
            )
    else:
        lines.append("    No rules triggered.")
    lines.append("")

    # ── Section 6: Context Score Detail ─────────────────────────────────
    lines.append("CONTEXT SCORE DETAIL")
    lines.append(f"  Total context score: {context_score:.4f}")
    age_days = features_row["account_age_days"]
    if age_days < 90:
        lines.append(f"    Young account ({age_days:.0f} days < 90): +0.30")
    elif age_days < 180:
        lines.append(f"    Semi-young account ({age_days:.0f} days < 180): +0.15")
    if acct_type == "business":
        lines.append(f"    Business account (high-flow context evaluated)")
    if cash_in > 0.4:
        lines.append(f"    High cash activity (cash_in_ratio={cash_in:.2f} > 0.4): +0.10")
    lines.append("")

    # ── Section 7: Peer Comparison ──────────────────────────────────────
    primary = agent_results[0]
    n_explain = min(len(FEATURE_COLUMNS), primary.explanations.shape[1])
    contrib_pairs = list(zip(FEATURE_COLUMNS[:n_explain], primary.explanations[sample_index][:n_explain]))
    contrib_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    lines.append("PEER COMPARISON (top 5 features by Agent 1 contribution)")
    for feat_name, _ in contrib_pairs[:5]:
        feat_val = features_row.get(feat_name, 0)
        pct = feature_percentiles.get(feat_name, 0)
        direction = "higher" if pct > 50 else "lower"
        lines.append(
            f"  {feat_name:30s}  value={feat_val:>12.2f}  "
            f"{direction} than {max(pct, 100-pct):.1f}% of accounts"
        )
    lines.append("")
    lines.append("-" * 76)

    return "\n".join(lines)


def _render_agent_explanation(lines, agent_result, sample_index, features_row, feature_percentiles):
    """Render the explanation block for a single agent."""
    name = agent_result.agent_name
    score = agent_result.scores[sample_index]
    explanation = agent_result.explanations[sample_index]

    if name.startswith("isolation_forest"):
        label = "Isolation Forest" if name == "isolation_forest" else "Isolation Forest (Strict)"
        lines.append(f"AGENT: {label}  (name={name}, score={score:.4f})")
        lines.append("  Method: Tree path-depth feature contributions")
        lines.append("  Interpretation: higher contribution = feature isolated this account at shallower depth")
        lines.append("")

        n_explain = min(len(FEATURE_COLUMNS), len(explanation))
        contrib_pairs = list(zip(FEATURE_COLUMNS[:n_explain], explanation[:n_explain]))
        contrib_pairs.sort(key=lambda x: x[1], reverse=True)
        lines.append("  Top 5 driving features:")
        for feat_name, contrib in contrib_pairs[:5]:
            feat_val = features_row.get(feat_name, 0)
            pct = feature_percentiles.get(feat_name, 0)
            lines.append(
                f"    {feat_name:30s}  value={feat_val:>12.2f}  "
                f"contribution={contrib:.4f}  (percentile {pct:.1f})"
            )
        lines.append("")

    elif name == "sparse_statistical":
        lines.append(f"AGENT: Sparse Statistical Z-Score  (name={name}, score={score:.4f})")
        lines.append("  Method: Robust Z-scores using median/MAD (capped at +/-8)")
        lines.append("  Interpretation: |z| > 3 = extreme outlier, |z| > 2 = notable, |z| < 1 = normal")
        lines.append("")

        n_explain = min(len(SPARSE_FEATURES), len(explanation))
        z_pairs = list(zip(SPARSE_FEATURES[:n_explain], explanation[:n_explain]))
        z_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        lines.append(f"  All {n_explain} features ranked by |z-score|:")
        for feat_name, z_val in z_pairs:
            feat_val = features_row.get(feat_name, 0)
            severity = _z_severity(z_val)
            direction = "above" if z_val > 0 else "below"
            lines.append(
                f"    {feat_name:30s}  value={feat_val:>12.2f}  "
                f"z={z_val:>+8.3f}  ({direction} median)  [{severity}]"
            )
        lines.append("")

    elif name == "lof_density":
        lines.append(f"AGENT: Local Outlier Factor  (name={name}, score={score:.4f})")
        lines.append("  Method: Density-based local outlier detection")
        lines.append("  Interpretation: higher contribution = feature deviates most from local neighbors")
        lines.append("")

        n_explain = min(len(FEATURE_COLUMNS), len(explanation))
        contrib_pairs = list(zip(FEATURE_COLUMNS[:n_explain], explanation[:n_explain]))
        contrib_pairs.sort(key=lambda x: x[1], reverse=True)
        lines.append("  Top 5 features driving local anomaly:")
        for feat_name, contrib in contrib_pairs[:5]:
            feat_val = features_row.get(feat_name, 0)
            pct = feature_percentiles.get(feat_name, 0)
            lines.append(
                f"    {feat_name:30s}  value={feat_val:>12.2f}  "
                f"contribution={contrib:.4f}  (percentile {pct:.1f})"
            )
        lines.append("")

    else:
        lines.append(f"AGENT: {name}  (score={score:.4f})")
        lines.append("  (explanation format not recognized — showing raw top values)")
        top_indices = np.argsort(np.abs(explanation))[::-1][:5]
        for idx in top_indices:
            lines.append(f"    feature[{idx}] = {explanation[idx]:.4f}")
        lines.append("")


def _z_severity(z):
    """Map absolute z-score to human-readable severity label."""
    az = abs(z)
    if az >= 4.0:
        return "EXTREME"
    elif az >= 3.0:
        return "SEVERE"
    elif az >= 2.0:
        return "ELEVATED"
    elif az >= 1.0:
        return "MILD"
    else:
        return "NORMAL"


def _assign_bucket(score):
    """Map score to risk bucket (used internally by report generator)."""
    if score < RISK_TIER_LOW:
        return "LOW"
    elif score < RISK_TIER_MEDIUM:
        return "MEDIUM"
    else:
        return "HIGH"
