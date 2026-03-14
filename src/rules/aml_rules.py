"""
Rule-based AML risk scoring layer — DISCRIMINATING MODE.

33 vectorized rules (19 original + R25-R31 targeted + R32-R38 new + R39-R44 lifecycle/network).
Raw sum capped at RULE_SCORE_CAP. Cash bundle R13/R14/R15 = max-of-three.
Job-aware scaling: cash rules softened for CASH_HEAVY, velocity/flow rules
strengthened for SALARIED/STUDENT.

DISCRIMINATION STRATEGY:
- Each rule has tight, conjunction-heavy conditions (low fire rate)
- Core strong rules: R3, R12, R20, R25, R38, R39, R42, R43
- Progressive bonus: 2 strong → +0.10, 3+ strong → +0.25
- Demographic rules are soft boosters (low weight, require behavioral conjunctions)
- Wire rules target high-risk cross-border layering patterns
- Multi-signal convergence catches accounts with many weak-but-concurrent flags
- Lifecycle coverage: Entry (R39) → Movement (R41) → Layering (R44) → Exit (R42) → Profile (R43)

  R1:  Rapid Pass-Through        (+0.30) — gap<48h, io 0.7-1.5, flow>p75
  R3:  Sudden Spike Established  (+0.25) [STRONG] — age>180d, accel>3x
  R4:  Structuring Pattern       (DISABLED)
  R5:  Burst Activity            (DISABLED)
  R7:  Large Tx Concentration    (DISABLED)
  R8:  Off-Hours Activity        (+0.10) — offhours>0.6, flow>p75
  R9:  Extreme Flow Imbalance    (+0.15) — io>10|<0.1, flow>p75
  R10: New Account + High Flow   (+0.05) — age<90, flow>p95
  R11: High Flow + High Velocity (DISABLED)
  R12: High IO + Short Gap       (+0.25) [STRONG] — io>2.5, gap<24h
  R13/R14/R15: Cash bundle max   (+0.25/0.25/0.00)
  R17: Channel Switching         (+0.25) — chan_switch>0.75
  R18: No Retail Usage           (+0.20) — retail<0.03, flow>p95
  R20: Cash Conversion           (+0.30) [STRONG] — atm_in>10k, card>1.5x
  R21: Repeated Channel Loop     (+0.05/0.10/0.20 by 100/250/500)
  R22: Sustained Passthrough     (+0.20) — age>60, passthrough>=3, io 0.3-3
  R23: Balance Drain             (+0.15) — retention<0.05, flow>p75
  R24: Counterparty Reuse        (+0.25) — merch_conc>0.80
  R25: Job Inconsistency         (+0.20-0.25) [STRONG]
  R26: Student High Wealth       (DISABLED)
  R27: Geographic Layering       (+0.15) — 5+ provinces, gap<48h, support signal
  R28: New Account Layering      (+0.25) — age<120, chan_switch>0.40, flow>p90
  R29: Passthrough No Retail     (+0.30) — passthrough>=2, retail<0.05
  R30: Offhours Burst Unique     (+0.30) — offhours>0.50, burst>3, repeat<0.08
  R31: New Account Deficit       (+0.25) — age<120, io<0.35, flow>p70
  R32: Young + New Account       (+0.10) — age<25, acct<180d, flow>p75 [SOFT]
  R33: Short Tenure + Wire       (+0.10) — acct<120d, wire_vel>p85 [SOFT]
  R34: Student + High Flow       (+0.10) — STUDENT, flow>p85, wire>0 [SOFT]
  R35: High Wire Count           (+0.15) — wire_count>p95
  R36: High Wire Velocity        (+0.15) — wire_vel>p90, acct<180d
  R37: Wire Clustering           (+0.15) — weekly_max>3, wire_vol>p90
  R38: Multi-Signal Convergence  (+0.20) [STRONG] — 4+ independent flags (12 total)
  R39: Dormant Reactivation      (+0.20) [STRONG] — inactive>180d, flow>p85, vel>p80
  R41: Micro-Structuring         (+0.15) — tx_count>p90, avg_size<p30, flow>p70
  R42: Exchange Funnel            (+0.25) [STRONG] — exchange_out>0.40, passthrough>=2
  R43: Income Mismatch           (+0.20) [STRONG] — income<p40, flow>p90
  R44: Balance Cycling           (+0.15) — cycles>5, retention<0.20
"""

import logging

import numpy as np
from tqdm import tqdm

from src.config import (
    RULE_SCORE_CAP,
    R39_INACTIVE_DAYS_MIN, R39_RECENT_FLOW_PERCENTILE, R39_VELOCITY_PERCENTILE,
    R41_TX_COUNT_PERCENTILE, R41_AVG_SIZE_PERCENTILE, R41_TOTAL_FLOW_PERCENTILE,
    R42_EXCHANGE_OUTFLOW_RATIO_MIN, R42_PASSTHROUGH_MIN,
    R43_INCOME_PERCENTILE, R43_TOTAL_FLOW_PERCENTILE,
    R44_CYCLE_MIN, R44_RETENTION_MAX,
    R38_CONVERGENCE_MIN_FLAGS,
)
from src.jobs.config import JOB_CONFIG

log = logging.getLogger("aml_pipeline")


def _build_job_multiplier(job_cat, rule_group, default=1.0):
    """Build a per-account multiplier array from JOB_CONFIG rule_scaling.

    Args:
        job_cat:    numpy array of job category strings.
        rule_group: key in JOB_CONFIG["rule_scaling"] (e.g. "cash_rules").
        default:    multiplier for categories not listed in the config.

    Returns:
        numpy array of float multipliers, same length as job_cat.
    """
    mapping = JOB_CONFIG["rule_scaling"].get(rule_group, {})
    mult = np.full(len(job_cat), default, dtype=float)
    for cat, val in mapping.items():
        mult[job_cat == cat] = val
    return mult


def compute_rule_scores(features_df):
    """Apply AML rule-based scoring layer — fully vectorized.

    Returns:
        rule_scores:     ndarray of [0, 1] per account
        triggered_rules: list of list-of-dicts per account with rule details
    """
    n = len(features_df)

    # Extract columns as numpy arrays for speed
    gap = features_df["median_in_out_gap_hours"].values
    io = features_df["io_ratio"].values
    acct_age = features_df["account_age_days"].values
    vel = features_df["tx_velocity"].values
    accel = features_df["flow_acceleration"].values
    pct_rnd = features_df["pct_round"].values
    rep_rat = features_df["repeat_amt_ratio"].values
    avg_sz = features_df["avg_tx_size"].values
    burst = features_df["burst_score"].values
    p_age = features_df["person_age_years"].values
    max_frac = features_df["max_tx_fraction"].values
    offhours = features_df["offhours_ratio"].values
    total_flow = features_df["total_inflow"].values + features_df["total_outflow"].values
    flow_per_day = features_df["flow_per_account_day"].values

    # Cash features
    cash_in_ratio = features_df["cash_in_ratio"].values if "cash_in_ratio" in features_df.columns else np.zeros(n)
    cash_out_ratio = features_df["cash_out_ratio"].values if "cash_out_ratio" in features_df.columns else np.zeros(n)

    # Job category (safe fallback)
    job_cat = features_df["job_category"].values.astype(str) if "job_category" in features_df.columns else np.full(n, "UNKNOWN")

    # Wire features (safe fallback)
    wire_tx_count = features_df["wire_tx_count"].values if "wire_tx_count" in features_df.columns else np.zeros(n)
    wire_velocity = features_df["wire_velocity"].values if "wire_velocity" in features_df.columns else np.zeros(n)
    wire_total_volume = features_df["wire_total_volume"].values if "wire_total_volume" in features_df.columns else np.zeros(n)
    wire_max_weekly = features_df["wire_max_weekly_count"].values if "wire_max_weekly_count" in features_df.columns else np.zeros(n)

    # New features for R39-R44 (safe fallback to zeros)
    inactive_days = features_df["inactive_days"].values if "inactive_days" in features_df.columns else np.zeros(n)
    exchange_out_ratio = features_df["exchange_outflow_ratio"].values if "exchange_outflow_ratio" in features_df.columns else np.zeros(n)
    raw_tx_count = features_df["raw_tx_count"].values if "raw_tx_count" in features_df.columns else np.zeros(n)
    declared_income = features_df["declared_income"].values if "declared_income" in features_df.columns else np.zeros(n)
    balance_cycles_arr = features_df["balance_cycles"].values if "balance_cycles" in features_df.columns else np.zeros(n)
    outflow_tx_count = features_df["outflow_count"].values if "outflow_count" in features_df.columns else np.zeros(n)

    # Compute percentile thresholds for data-driven rules
    avg_sz_p95 = np.percentile(avg_sz, 95)
    flow_per_day_p90 = np.percentile(flow_per_day, 90)
    flow_per_day_p95 = np.percentile(flow_per_day, 95)
    velocity_p90 = np.percentile(vel, 90)

    # ── Job-aware multipliers ─────────────────────────────────────────────
    cash_mult = _build_job_multiplier(job_cat, "cash_rules")
    vel_mult = _build_job_multiplier(job_cat, "velocity_passthrough_rules")
    flow_mult = _build_job_multiplier(job_cat, "large_flow_rules")
    biz_mult = _build_job_multiplier(job_cat, "business_behavior_rules")

    # ── Original Rules (R1-R9) ───────────────────────────────────────────

    # R1: Rapid Pass-Through (velocity-scaled) — require significant flow
    total_flow_p75 = np.percentile(total_flow, 75) if n > 0 else 0
    r1_mask = (gap > 0) & (gap < 48) & (io >= 0.7) & (io <= 1.5) & (total_flow > total_flow_p75)
    r1_scores = np.where(r1_mask, 0.30, 0.0) * vel_mult

    # R3: Sudden Spike on Established Account
    r3_mask = (acct_age > 180) & (accel > 3.0)
    r3_scores = np.where(r3_mask, 0.25, 0.0)

    # R4: DISABLED — zero launderer lift, low discrimination value
    r4_threshold = np.zeros(n, dtype=bool)
    r4_general = np.zeros(n, dtype=bool)
    r4_scores = np.zeros(n)

    # R5: DISABLED — zero launderer lift
    r5_mask = np.zeros(n, dtype=bool)
    r5_scores = np.zeros(n)

    # R7: DISABLED — zero launderer lift
    r7_mask = np.zeros(n, dtype=bool)
    r7_scores = np.zeros(n)

    # R8: Off-Hours Activity — tightened: higher offhours + top-quartile flow
    r8_mask = (offhours > 0.6) & (total_flow > total_flow_p75)
    r8_scores = np.where(r8_mask, 0.10, 0.0) * biz_mult

    # R9: Extreme Flow Imbalance (flow-scaled) — uses p75 computed above
    r9_mask = ((io > 10) | (io < 0.1)) & (total_flow > total_flow_p75)
    r9_scores = np.where(r9_mask, 0.15, 0.0) * flow_mult

    # R6: Age modifier — disabled
    age_mod = np.ones(n)

    # ── New Cash & Behavioral Rules (R10-R16) ────────────────────────────

    # R10: New Account + High Flow — young account, require >p95 daily flow, reduced weight
    r10_mask = (acct_age < 90) & (flow_per_day > flow_per_day_p95)
    r10_scores = np.where(r10_mask, 0.05, 0.0)

    # R11: DISABLED — zero launderer lift
    r11_mask = np.zeros(n, dtype=bool)
    r11_scores = np.zeros(n)

    # R12: High IO Ratio + Short Gap (velocity-scaled) — 2.5 but requires top-half flow
    total_flow_p50 = np.percentile(total_flow, 50) if n > 0 else 0
    r12_mask = (io > 2.5) & (gap > 0) & (gap < 24) & (total_flow > total_flow_p50)
    r12_scores = np.where(r12_mask, 0.25, 0.0) * vel_mult

    # R13: High Cash Inflow (cash-scaled)
    r13_mask = cash_in_ratio > 0.6
    r13_scores = np.where(r13_mask, 0.25, 0.0) * cash_mult

    # R14: High Cash Outflow (cash-scaled)
    r14_mask = cash_out_ratio > 0.6
    r14_scores = np.where(r14_mask, 0.25, 0.0) * cash_mult

    # R15: DISABLED — zero launderer lift (cash cycle never fires for known launderers)
    r15_mask = np.zeros(n, dtype=bool)
    r15_scores = np.zeros(n)

    # ── Quiet Laundering Rules (R17-R20) ────────────────────────────────

    # Quiet laundering features (safe fallback to zeros if not yet computed)
    chan_switch = features_df["channel_switch_ratio"].values if "channel_switch_ratio" in features_df.columns else np.zeros(n)
    retail_rat = features_df["retail_ratio"].values if "retail_ratio" in features_df.columns else np.zeros(n)
    atm_in = features_df["atm_in_total"].values if "atm_in_total" in features_df.columns else np.zeros(n)
    card_out = features_df["card_out_total"].values if "card_out_total" in features_df.columns else np.zeros(n)

    total_flow_p90 = np.percentile(total_flow, 90)  # kept for R20 detail output

    # R17: Channel Switching — tightened threshold, reduced points
    r17_mask = chan_switch > 0.75
    r17_scores = np.where(r17_mask, 0.25, 0.0)

    # R18: No Retail Usage — tighter: <3% retail, top-5% flow
    total_flow_p95 = np.percentile(total_flow, 95) if n > 0 else 0
    r18_mask = (retail_rat < 0.03) & (total_flow > total_flow_p95)
    r18_scores = np.where(r18_mask, 0.20, 0.0) * biz_mult

    # R20: Cash Conversion — ATM deposits converted to card/electronic spending at >1.5x rate
    r20_mask = (atm_in > 10000) & (card_out > 1.5 * atm_in)
    r20_scores = np.where(r20_mask, 0.30, 0.0)

    # ── Stealth Laundering Rules (R21-R24) ─────────────────────────────

    # Stealth features (safe fallback to zeros)
    loop_count = features_df["channel_loop_count"].values if "channel_loop_count" in features_df.columns else np.zeros(n)
    passthrough_ct = features_df["sustained_passthrough_count"].values if "sustained_passthrough_count" in features_df.columns else np.zeros(n)
    bal_retention = features_df["balance_retention"].values if "balance_retention" in features_df.columns else np.ones(n)
    merch_conc = features_df["merchant_concentration"].values if "merchant_concentration" in features_df.columns else np.zeros(n)

    # R21: Repeated Channel Loop — tightened thresholds + supporting condition
    # Requires loop pattern PLUS a behavioral signal (passthrough io, fast gap, or high velocity)
    r21_support = (
        ((io >= 0.5) & (io <= 2.0))   # balanced flow (passthrough signature)
        | ((gap > 0) & (gap < 48))      # fast turnaround
        | (vel > velocity_p90)           # high velocity
    )
    r21_scores = np.where(
        (loop_count >= 500) & r21_support, 0.20,
        np.where(
            (loop_count >= 250) & r21_support, 0.10,
            np.where(
                (loop_count >= 100) & r21_support, 0.05, 0.0
            )
        )
    )
    r21_triggered = (loop_count >= 100) & r21_support

    # R22: Sustained Passthrough — lowered to 3 but requires low retail (no real spending)
    r22_mask = (acct_age > 60) & (passthrough_ct >= 3) & (io >= 0.3) & (io <= 3.0) & (retail_rat < 0.10)
    r22_scores = np.where(r22_mask, 0.20, 0.0) * biz_mult

    # R23: Balance Drain — tighter: <5% retention, top-quartile flow
    r23_mask = (bal_retention < 0.05) & (total_flow > total_flow_p75)
    r23_scores = np.where(r23_mask, 0.15, 0.0)

    # R24: Counterparty Reuse — tightened to >80% concentration
    r24_mask = merch_conc > 0.80
    r24_scores = np.where(r24_mask, 0.25, 0.0)

    # ── Job-Aware Rules (R25-R26) ─────────────────────────────────────

    inc = JOB_CONFIG["inconsistency"]
    sw = JOB_CONFIG["student_wealth"]

    is_student = job_cat == "STUDENT"
    is_salaried = job_cat == "SALARIED"
    is_cash_heavy = job_cat == "CASH_HEAVY"

    # Percentile thresholds for job inconsistency
    flow_p_inc = np.percentile(total_flow, inc["student_high_flow_p"]) if n > 0 else 0
    vel_p_inc = np.percentile(vel, inc["student_high_velocity_p"]) if n > 0 else 0
    # Use balance_retention inversely — low retention = high flow-through
    bal_p_inc = np.percentile(total_flow, inc["student_high_balance_p"]) if n > 0 else 0

    # R25: Job–Behavior Inconsistency [strong]
    # Sub-condition A: STUDENT with high flow OR high velocity OR high total_flow
    r25_student = is_student & (
        (total_flow > flow_p_inc) | (vel > vel_p_inc) | (total_flow > bal_p_inc)
    )
    # Sub-condition B: SALARIED with heavy cash inflow
    r25_salaried = is_salaried & (cash_in_ratio > inc["salaried_cash_in_ratio"])
    # Sub-condition C: CASH_HEAVY with low retail + rapid pass-through
    r25_cash_heavy = is_cash_heavy & (
        (retail_rat < inc["cash_heavy_low_retail"])
        & (gap > 0) & (gap < inc["cash_heavy_high_passthrough_gap"])
    )

    r25_mask = r25_student | r25_salaried | r25_cash_heavy
    r25_scores = np.where(
        r25_student, inc["student_penalty"],
        np.where(r25_salaried, inc["salaried_penalty"],
                 np.where(r25_cash_heavy, inc["cash_heavy_penalty"], 0.0))
    )

    # R26: Student High Wealth [strong]
    flow_p_sw = np.percentile(total_flow, sw["flow_percentile"]) if n > 0 else 0
    vel_p_sw = np.percentile(vel, sw["velocity_percentile"]) if n > 0 else 0
    bal_p_sw = np.percentile(total_flow, sw["balance_percentile"]) if n > 0 else 0

    sw_flow_hit = is_student & (total_flow > flow_p_sw)
    sw_vel_hit = is_student & (vel > vel_p_sw)
    sw_bal_hit = is_student & (total_flow > bal_p_sw)
    sw_hit_count = sw_flow_hit.astype(int) + sw_vel_hit.astype(int) + sw_bal_hit.astype(int)

    # R26: DISABLED — zero launderer lift
    r26_mask = np.zeros(n, dtype=bool)
    r26_scores = np.zeros(n)

    # ── Multi-Signal Conjunction Rules (R27-R28) ──────────────────────
    prov = features_df["distinct_provinces"].values if "distinct_provinces" in features_df.columns else np.ones(n)

    # R27: Multi-Province + Short Gap + Moderate IO — geographic layering pattern
    # Tightened: 5+ provinces, requires supporting behavioral signal
    r27_support = (
        (chan_switch > 0.40)            # channel switching behavior
        | (passthrough_ct >= 2)          # sustained passthrough
        | (offhours > 0.30)              # off-hours activity
        | (io > 2.5) | (io < 0.4)       # extreme flow imbalance
    )
    r27_mask = (
        (prov >= 5) & (gap > 0) & (gap < 48)
        & (io >= 0.3) & (io <= 3.0)
        & (vel > np.percentile(vel, 70))
        & r27_support
    )
    r27_scores = np.where(r27_mask, 0.15, 0.0)

    # R28: New Account + Channel Diversity + Moderate Flow — fresh account layering
    # New accounts with diverse channel usage suggest immediate layering setup
    r28_mask = (acct_age < 120) & (chan_switch > 0.40) & (flow_per_day > flow_per_day_p90) & (prov >= 2)
    r28_scores = np.where(r28_mask, 0.25, 0.0)

    # ── Stealth Launderer Rules (R29-R31) ──────────────────────────────
    # Targeted at launderers missed by all other rules/agents.
    # Designed from deep investigation of SYNID0107832828, 0200755995, 0200187014.

    # R29: Low-Volume Passthrough + No Retail — catches "invisible" pass-through
    # launderers who keep volumes modest but retain nothing and do no real spending.
    # Target: SYNID0107832828 (student, EMT-only, io=0.51, passthrough=3, retail=0.003)
    r29_mask = (
        (passthrough_ct >= 2)       # sustained pass-through behavior
        & (retail_rat < 0.05)       # virtually no retail spending
        & (io >= 0.2) & (io <= 2.0) # balanced in/out (pass-through signature)
        & (bal_retention < 0.40)    # not retaining funds
    )
    r29_scores = np.where(r29_mask, 0.30, 0.0) * biz_mult

    # R30: Off-Hours Burst + No Repeat Amounts — catches sophisticated launderers
    # who transact almost exclusively outside business hours with unique amounts
    # to avoid structuring detection.
    # Target: SYNID0200755995 (business, burst=3.9@p99, offhours=0.96@p98, repeat=0.04@p2)
    r30_mask = (
        (offhours > 0.50)           # majority off-hours
        & (burst > 3.0)             # high burst activity
        & (rep_rat < 0.08)          # almost no repeated amounts (anti-structuring)
        & (retail_rat < 0.05)       # no retail
    )
    r30_scores = np.where(r30_mask, 0.30, 0.0)

    # R31: New Account Deficit Spending — catches shell accounts that spend
    # far more than they receive, indicating external funding or layered inflows.
    # Target: SYNID0200187014 (48 days old, io=0.27, spending 3.7x inflow)
    r31_mask = (
        (acct_age < 120)            # new account
        & (io < 0.35)               # outflow >> inflow
        & (flow_per_day > np.percentile(flow_per_day, 70))  # non-trivial activity
    )
    r31_scores = np.where(r31_mask, 0.25, 0.0)

    # ── Demographic Soft Boosters (R32-R34) ─────────────────────────────
    # Low-weight rules using age/tenure as soft risk signals (not hard blocks).
    # Non-discriminatory: require conjunction with behavioral signals.

    flow_per_day_p75 = np.percentile(flow_per_day, 75) if n > 0 else 0
    total_flow_p85 = np.percentile(total_flow, 85) if n > 0 else 0

    # R32: Young + New Account — young person, new account, above-average daily flow
    r32_mask = (
        (p_age < 25) & (~np.isnan(p_age))
        & (acct_age < 180)
        & (flow_per_day > flow_per_day_p75)
    )
    r32_scores = np.where(r32_mask, 0.10, 0.0)

    # R33: Short Tenure + High Wire — new account with elevated wire velocity
    wire_active = wire_velocity[wire_velocity > 0]
    wire_vel_p85 = np.percentile(wire_active, 85) if len(wire_active) > 10 else 999
    r33_mask = (acct_age < 120) & (wire_velocity > wire_vel_p85)
    r33_scores = np.where(r33_mask, 0.10, 0.0)

    # R34: Student + High Flow + Wire — student with high flow and wire activity
    r34_mask = is_student & (total_flow > total_flow_p85) & (wire_tx_count > 0)
    r34_scores = np.where(r34_mask, 0.10, 0.0)

    # ── Wire Rules (R35-R37) ─────────────────────────────────────────────
    # Wire transactions are high-risk for cross-border layering.

    wire_count_p95 = np.percentile(wire_active, 95) if len(wire_active) > 10 else 999
    wire_vel_p90 = np.percentile(wire_active, 90) if len(wire_active) > 10 else 999
    wire_vol_active = wire_total_volume[wire_total_volume > 0]
    wire_vol_p90 = np.percentile(wire_vol_active, 90) if len(wire_vol_active) > 10 else 999

    # R35: High Wire Count — wire tx count above p95 among wire-active accounts
    r35_mask = wire_tx_count > wire_count_p95
    r35_scores = np.where(r35_mask, 0.15, 0.0)

    # R36: High Wire Velocity + New Account — fast wire activity on young account
    r36_mask = (wire_velocity > wire_vel_p90) & (acct_age < 180)
    r36_scores = np.where(r36_mask, 0.15, 0.0)

    # R37: Wire Clustering — dense weekly clustering with high volume
    r37_mask = (wire_max_weekly > 3) & (wire_total_volume > wire_vol_p90)
    r37_scores = np.where(r37_mask, 0.15, 0.0)

    # ── New Lifecycle & Network Rules (R39, R41-R44) ─────────────────────

    # New percentile thresholds for R39-R44
    total_flow_p85 = np.percentile(total_flow, R39_RECENT_FLOW_PERCENTILE) if n > 0 else 0
    velocity_p80 = np.percentile(vel, R39_VELOCITY_PERCENTILE) if n > 0 else 0
    raw_tx_count_p90 = np.percentile(raw_tx_count, R41_TX_COUNT_PERCENTILE) if n > 0 else 0
    avg_sz_p30 = np.percentile(avg_sz, R41_AVG_SIZE_PERCENTILE) if n > 0 else 0
    total_flow_p70 = np.percentile(total_flow, R41_TOTAL_FLOW_PERCENTILE) if n > 0 else 0
    # Income percentile: only over accounts with declared income > 0
    # (businesses and missing-data accounts excluded from R43 by design)
    income_positive = declared_income[declared_income > 0]
    income_p40 = np.percentile(income_positive, R43_INCOME_PERCENTILE) if len(income_positive) > 10 else 0
    total_flow_p90_r43 = np.percentile(total_flow, R43_TOTAL_FLOW_PERCENTILE) if n > 0 else 0

    # R39: Dormant Account Reactivation [STRONG +0.20]
    # Regulatory rationale (FATF Rec 10): Dormant accounts suddenly resuming
    # high-volume activity suggest the account was set up for eventual use in
    # a layering scheme. The combination of extended inactivity + immediate
    # high-flow resumption is a strong smurfing/layering indicator.
    r39_mask = (
        (inactive_days > R39_INACTIVE_DAYS_MIN)
        & (total_flow > total_flow_p85)
        & (vel > velocity_p80)
    )
    r39_scores = np.where(r39_mask, 0.20, 0.0)

    # R41: Micro-Structuring [Medium +0.15]
    # Regulatory rationale (PCMLTFA s.9): Breaking large amounts into many
    # small transactions to avoid reporting thresholds. Signature: very high
    # transaction count, unusually small average size, but total flow is
    # substantial — the sum reveals intent despite individual smallness.
    r41_mask = (
        (raw_tx_count > raw_tx_count_p90)
        & (avg_sz < avg_sz_p30)
        & (total_flow > total_flow_p70)
    )
    r41_scores = np.where(r41_mask, 0.15, 0.0)

    # R42: Exchange Funnel [STRONG +0.25]
    # Regulatory rationale (FINTRAC ML Typologies): Accounts acting as a
    # funnel into cross-border channels (wire + Western Union) with sustained
    # passthrough behavior execute the classic placement→layering progression.
    # The passthrough requirement ensures funds don't originate legitimately.
    r42_mask = (
        (exchange_out_ratio > R42_EXCHANGE_OUTFLOW_RATIO_MIN)
        & (passthrough_ct >= R42_PASSTHROUGH_MIN)
    )
    r42_scores = np.where(r42_mask, 0.25, 0.0)

    # R43: Income Mismatch [STRONG +0.20]
    # Regulatory rationale (FINTRAC KYCC Guidance): When declared income is
    # in the bottom 40% but total flow is in the top 10%, the gap cannot be
    # explained by normal economic activity. Non-discriminatory: threshold is
    # population-relative (percentile), not an absolute income cutoff.
    # Only fires for accounts with non-zero declared income.
    income_available = declared_income > 0
    r43_mask = (
        income_available
        & (declared_income < income_p40)
        & (total_flow > total_flow_p90_r43)
    )
    r43_scores = np.where(r43_mask, 0.20, 0.0)

    # R44: Balance Cycling [Medium +0.15]
    # Regulatory rationale (FATF Typology): Repeated fill-then-drain cycles
    # indicate an account is being used as a transient vessel. Combined with
    # low retention, confirms funds are passed through rather than saved.
    r44_mask = (
        (balance_cycles_arr > R44_CYCLE_MIN)
        & (bal_retention < R44_RETENTION_MAX)
    )
    r44_scores = np.where(r44_mask, 0.15, 0.0)

    # ── Multi-Signal Convergence (R38) ───────────────────────────────────
    # Fires when 4+ independent red flags are present simultaneously.
    # Catches accounts where no single signal is extreme but the combination is.
    # Tightened: wire>=3 (heavy users only), passthrough requires pt_count>=2
    # (dropped gap+io shortcut that fired on 30% of normal accounts),
    # low_retail<0.03, channel_switch>0.50.
    flag_wire = (wire_tx_count >= 3).astype(int)
    flag_offhours = (offhours > 0.30).astype(int)
    flag_passthrough = (passthrough_ct >= 2).astype(int)
    flag_channel_switch = (chan_switch > 0.50).astype(int)
    flag_flow_accel = (accel > 2.0).astype(int)
    flag_new_account = (acct_age < 120).astype(int)
    flag_young_age = ((p_age < 25) & (~np.isnan(p_age))).astype(int)
    flag_high_velocity = (vel > velocity_p90).astype(int)
    flag_low_retail = (retail_rat < 0.03).astype(int)

    # New convergence flags from R39-R44 (3 additional independent signals)
    flag_dormant = (inactive_days > R39_INACTIVE_DAYS_MIN).astype(int)
    flag_exchange_funnel = (exchange_out_ratio > R42_EXCHANGE_OUTFLOW_RATIO_MIN).astype(int)
    flag_balance_cycling = (balance_cycles_arr > R44_CYCLE_MIN).astype(int)

    # For BUSINESS accounts, exclude low_retail and offhours from convergence
    # count — these are normal business behavior, not independent red flags.
    is_business = (job_cat == "BUSINESS").astype(int)
    convergence_count = (
        flag_wire + flag_offhours * (1 - is_business) + flag_passthrough + flag_channel_switch
        + flag_flow_accel + flag_new_account + flag_young_age
        + flag_high_velocity + flag_low_retail * (1 - is_business)
        + flag_dormant + flag_exchange_funnel + flag_balance_cycling
    )
    r38_mask = convergence_count >= R38_CONVERGENCE_MIN_FLAGS
    r38_scores = np.where(r38_mask, 0.20, 0.0)

    # ── Aggregate ────────────────────────────────────────────────────────
    # Cash bundle: take max of R13/R14/R15 instead of summing (prevents stacking)
    cash_bundle = np.maximum(np.maximum(r13_scores, r14_scores), r15_scores)

    raw = (r1_scores + r3_scores + r4_scores + r5_scores
           + r7_scores + r8_scores + r9_scores
           + r10_scores + r11_scores + r12_scores
           + cash_bundle
           + r17_scores + r18_scores + r20_scores
           + r21_scores + r22_scores + r23_scores + r24_scores
           + r25_scores + r26_scores
           + r27_scores + r28_scores
           + r29_scores + r30_scores + r31_scores
           + r32_scores + r33_scores + r34_scores
           + r35_scores + r36_scores + r37_scores
           + r38_scores
           + r39_scores + r41_scores + r42_scores
           + r43_scores + r44_scores) * age_mod
    rule_scores = np.minimum(raw, RULE_SCORE_CAP)

    # CORE strong rules — R3, R12, R20, R25 (original) + R39, R42, R43 (new)
    strong_masks = [r3_mask, r12_mask, r20_mask, r25_mask,
                    r39_mask, r42_mask, r43_mask]
    strong_count = sum(m.astype(np.int64) for m in strong_masks)
    strong_count = np.clip(strong_count, 0, len(strong_masks))

    # Progressive bonus: 2 strong → +0.10, 3+ strong → +0.25
    rule_scores = rule_scores + np.where(
        strong_count >= 3, 0.25,
        np.where(strong_count >= 2, 0.10, 0.0)
    )
    rule_scores = np.clip(rule_scores, 0.0, 1.0)

    # Build triggered_rules list (per-account detail strings)
    log.info("  Building rule trigger details...")
    triggered_rules = [[] for _ in range(n)]
    for i in tqdm(range(n), desc="  Rule details", unit="acct", mininterval=1.0, disable=(n < 1000)):
        triggers = []
        if r1_mask[i]:
            triggers.append({
                "rule": "R1_RAPID_PASSTHROUGH",
                "points": float(r1_scores[i]),
                "detail": f"median_gap={gap[i]:.1f}h, io_ratio={io[i]:.2f}, job={job_cat[i]}, vel_mult={vel_mult[i]:.2f}",
            })
        if r3_mask[i]:
            triggers.append({
                "rule": "R3_SUDDEN_SPIKE_ESTABLISHED",
                "points": 0.25,
                "detail": f"account_age={acct_age[i]:.0f}d, acceleration={accel[i]:.2f}x",
            })
        if r4_threshold[i]:
            triggers.append({
                "rule": "R4_STRUCTURING_THRESHOLD",
                "points": 0.35,
                "detail": f"pct_round={pct_rnd[i]:.2f}, repeat_ratio={rep_rat[i]:.2f}, avg_size=${avg_sz[i]:.0f}",
            })
        elif r4_general[i]:
            triggers.append({
                "rule": "R4_STRUCTURING_GENERAL",
                "points": 0.20,
                "detail": f"pct_round={pct_rnd[i]:.2f}, repeat_ratio={rep_rat[i]:.2f}",
            })
        if r5_mask[i]:
            triggers.append({
                "rule": "R5_BURST_ACTIVITY",
                "points": 0.15,
                "detail": f"burst_score={burst[i]:.2f}",
            })
        if r7_mask[i]:
            triggers.append({
                "rule": "R7_LARGE_TX_CONCENTRATION",
                "points": float(r7_scores[i]),
                "detail": f"max_tx_fraction={max_frac[i]:.2f}, avg_tx_size=${avg_sz[i]:.0f} (p95=${avg_sz_p95:.0f}), job={job_cat[i]}, flow_mult={flow_mult[i]:.2f}",
            })
        if r8_mask[i]:
            triggers.append({
                "rule": "R8_OFFHOURS_ACTIVITY",
                "points": float(r8_scores[i]),
                "detail": f"offhours_ratio={offhours[i]:.2f}, total_flow=${total_flow[i]:.0f}, job={job_cat[i]}, biz_mult={biz_mult[i]:.2f}",
            })
        if r9_mask[i]:
            triggers.append({
                "rule": "R9_EXTREME_FLOW_IMBALANCE",
                "points": float(r9_scores[i]),
                "detail": f"io_ratio={io[i]:.2f}, total_flow=${total_flow[i]:.0f}, job={job_cat[i]}, flow_mult={flow_mult[i]:.2f}",
            })
        if r10_mask[i]:
            triggers.append({
                "rule": "R10_NEW_ACCOUNT_HIGH_FLOW",
                "points": 0.05,
                "detail": f"account_age={acct_age[i]:.0f}d, flow_per_day={flow_per_day[i]:.2f} (p95={flow_per_day_p95:.2f})",
            })
        if r11_mask[i]:
            triggers.append({
                "rule": "R11_HIGH_FLOW_HIGH_VELOCITY",
                "points": float(r11_scores[i]),
                "detail": f"flow_per_day={flow_per_day[i]:.2f} (p90={flow_per_day_p90:.2f}), velocity={vel[i]:.2f} (p90={velocity_p90:.2f}), job={job_cat[i]}, vel_mult={vel_mult[i]:.2f}",
            })
        if r12_mask[i]:
            triggers.append({
                "rule": "R12_HIGH_IO_SHORT_GAP",
                "points": float(r12_scores[i]),
                "detail": f"io_ratio={io[i]:.2f}, median_gap={gap[i]:.1f}h, job={job_cat[i]}, vel_mult={vel_mult[i]:.2f}",
            })
        # Cash bundle: only the highest-scoring cash rule contributes to the score
        cash_active = cash_bundle[i]
        if cash_active > 0:
            if r15_mask[i] and r15_scores[i] >= r13_scores[i] and r15_scores[i] >= r14_scores[i]:
                triggers.append({
                    "rule": "R15_CASH_CYCLE",
                    "points": float(r15_scores[i]),
                    "detail": f"cash_in_ratio={cash_in_ratio[i]:.2f}, cash_out_ratio={cash_out_ratio[i]:.2f}, job={job_cat[i]}, cash_mult={cash_mult[i]:.2f} [cash bundle winner]",
                })
            elif r13_mask[i] and r13_scores[i] >= r14_scores[i]:
                triggers.append({
                    "rule": "R13_HIGH_CASH_INFLOW",
                    "points": float(r13_scores[i]),
                    "detail": f"cash_in_ratio={cash_in_ratio[i]:.2f}, job={job_cat[i]}, cash_mult={cash_mult[i]:.2f} [cash bundle winner]",
                })
            elif r14_mask[i]:
                triggers.append({
                    "rule": "R14_HIGH_CASH_OUTFLOW",
                    "points": float(r14_scores[i]),
                    "detail": f"cash_out_ratio={cash_out_ratio[i]:.2f}, job={job_cat[i]}, cash_mult={cash_mult[i]:.2f} [cash bundle winner]",
                })
        if r17_mask[i]:
            triggers.append({
                "rule": "R17_CHANNEL_SWITCHING",
                "points": 0.25,
                "detail": f"channel_switch_ratio={chan_switch[i]:.2f}",
            })
        if r18_mask[i]:
            triggers.append({
                "rule": "R18_NO_RETAIL_USAGE",
                "points": float(r18_scores[i]),
                "detail": f"retail_ratio={retail_rat[i]:.2f}, total_flow=${total_flow[i]:.0f} (p95=${total_flow_p95:.0f}), job={job_cat[i]}, biz_mult={biz_mult[i]:.2f}",
            })
        if r20_mask[i]:
            triggers.append({
                "rule": "R20_CASH_CONVERSION",
                "points": 0.30,
                "detail": f"atm_in=${atm_in[i]:.0f}, card_out=${card_out[i]:.0f}, ratio={card_out[i]/(atm_in[i]+1):.2f}x",
            })
        if r21_triggered[i]:
            pts = 0.20 if loop_count[i] >= 500 else (0.10 if loop_count[i] >= 250 else 0.05)
            triggers.append({
                "rule": "R21_REPEATED_CHANNEL_LOOP",
                "points": pts,
                "detail": f"channel_loop_count={loop_count[i]:.0f}, io={io[i]:.2f}, gap={gap[i]:.1f}h, vel={vel[i]:.2f}",
            })
        if r22_mask[i]:
            triggers.append({
                "rule": "R22_SUSTAINED_PASSTHROUGH",
                "points": float(r22_scores[i]),
                "detail": f"account_age={acct_age[i]:.0f}d, median_gap={gap[i]:.1f}h, passthrough_count={passthrough_ct[i]:.0f}, io={io[i]:.2f}, job={job_cat[i]}, biz_mult={biz_mult[i]:.2f}",
            })
        if r23_mask[i]:
            triggers.append({
                "rule": "R23_BALANCE_DRAIN",
                "points": 0.15,
                "detail": f"balance_retention={bal_retention[i]:.2f}, total_flow=${total_flow[i]:.0f}",
            })
        if r24_mask[i]:
            triggers.append({
                "rule": "R24_COUNTERPARTY_REUSE",
                "points": 0.25,
                "detail": f"merchant_concentration={merch_conc[i]:.2f}",
            })
        # R25: Job–Behavior Inconsistency
        if r25_mask[i]:
            sub = "student" if r25_student[i] else ("salaried" if r25_salaried[i] else "cash_heavy")
            triggers.append({
                "rule": "R25_JOB_INCONSISTENCY",
                "points": float(r25_scores[i]),
                "detail": f"job={job_cat[i]}, sub_type={sub}, cash_in_ratio={cash_in_ratio[i]:.2f}, velocity={vel[i]:.2f}, total_flow=${total_flow[i]:.0f}",
            })
        # R26: Student High Wealth
        if r26_mask[i]:
            triggers.append({
                "rule": "R26_STUDENT_HIGH_WEALTH",
                "points": float(r26_scores[i]),
                "detail": f"job=STUDENT, hits={int(sw_hit_count[i])}, total_flow=${total_flow[i]:.0f}, velocity={vel[i]:.2f}",
            })
        # R27: Multi-Province Geographic Layering
        if r27_mask[i]:
            triggers.append({
                "rule": "R27_GEOGRAPHIC_LAYERING",
                "points": 0.15,
                "detail": f"provinces={prov[i]:.0f}, median_gap={gap[i]:.1f}h, io_ratio={io[i]:.2f}, velocity={vel[i]:.2f}, chan_sw={chan_switch[i]:.2f}",
            })
        # R28: New Account Channel Layering
        if r28_mask[i]:
            triggers.append({
                "rule": "R28_NEW_ACCOUNT_LAYERING",
                "points": 0.25,
                "detail": f"account_age={acct_age[i]:.0f}d, chan_switch={chan_switch[i]:.2f}, flow_per_day={flow_per_day[i]:.2f}, provinces={prov[i]:.0f}",
            })
        if r29_mask[i]:
            triggers.append({
                "rule": "R29_PASSTHROUGH_NO_RETAIL",
                "points": float(r29_scores[i]),
                "detail": f"passthrough={passthrough_ct[i]:.0f}, retail={retail_rat[i]:.2f}, io={io[i]:.2f}, retention={bal_retention[i]:.2f}, job={job_cat[i]}, biz_mult={biz_mult[i]:.2f}",
            })
        if r30_mask[i]:
            triggers.append({
                "rule": "R30_OFFHOURS_BURST_UNIQUE",
                "points": 0.30,
                "detail": f"offhours={offhours[i]:.2f}, burst={burst[i]:.2f}, repeat_ratio={rep_rat[i]:.2f}, retail={retail_rat[i]:.2f}",
            })
        if r31_mask[i]:
            triggers.append({
                "rule": "R31_NEW_ACCOUNT_DEFICIT",
                "points": 0.25,
                "detail": f"account_age={acct_age[i]:.0f}d, io_ratio={io[i]:.2f}, flow_per_day={flow_per_day[i]:.2f}",
            })
        if r32_mask[i]:
            triggers.append({
                "rule": "R32_YOUNG_NEW_ACCOUNT",
                "points": 0.10,
                "detail": f"person_age={p_age[i]:.0f}, account_age={acct_age[i]:.0f}d, flow_per_day={flow_per_day[i]:.2f}",
            })
        if r33_mask[i]:
            triggers.append({
                "rule": "R33_SHORT_TENURE_WIRE",
                "points": 0.10,
                "detail": f"account_age={acct_age[i]:.0f}d, wire_velocity={wire_velocity[i]:.4f}",
            })
        if r34_mask[i]:
            triggers.append({
                "rule": "R34_STUDENT_HIGH_FLOW",
                "points": 0.10,
                "detail": f"job=STUDENT, total_flow=${total_flow[i]:.0f}, wire_tx_count={wire_tx_count[i]:.0f}",
            })
        if r35_mask[i]:
            triggers.append({
                "rule": "R35_HIGH_WIRE_COUNT",
                "points": 0.15,
                "detail": f"wire_tx_count={wire_tx_count[i]:.0f} (p95={wire_count_p95:.0f})",
            })
        if r36_mask[i]:
            triggers.append({
                "rule": "R36_HIGH_WIRE_VELOCITY",
                "points": 0.15,
                "detail": f"wire_velocity={wire_velocity[i]:.4f}, account_age={acct_age[i]:.0f}d",
            })
        if r37_mask[i]:
            triggers.append({
                "rule": "R37_WIRE_CLUSTERING",
                "points": 0.15,
                "detail": f"wire_max_weekly={wire_max_weekly[i]:.0f}, wire_volume=${wire_total_volume[i]:.0f}",
            })
        if r38_mask[i]:
            triggers.append({
                "rule": "R38_MULTI_SIGNAL_CONVERGENCE",
                "points": 0.20,
                "detail": (
                    f"signals={int(convergence_count[i])}"
                    f", wire={bool(flag_wire[i])}, offhours={bool(flag_offhours[i])}"
                    f", passthrough={bool(flag_passthrough[i])}, chan_sw={bool(flag_channel_switch[i])}"
                    f", accel={bool(flag_flow_accel[i])}, new_acct={bool(flag_new_account[i])}"
                    f", young={bool(flag_young_age[i])}, hi_vel={bool(flag_high_velocity[i])}"
                    f", low_retail={bool(flag_low_retail[i])}"
                    f", dormant={bool(flag_dormant[i])}, exch_funnel={bool(flag_exchange_funnel[i])}"
                    f", bal_cycling={bool(flag_balance_cycling[i])}"
                ),
            })
        if r39_mask[i]:
            triggers.append({
                "rule": "R39_DORMANT_REACTIVATION",
                "points": 0.20,
                "detail": f"inactive_days={inactive_days[i]:.0f}d, total_flow=${total_flow[i]:.0f} (p85=${total_flow_p85:.0f}), velocity={vel[i]:.4f} (p80={velocity_p80:.4f})",
            })
        if r41_mask[i]:
            triggers.append({
                "rule": "R41_MICRO_STRUCTURING",
                "points": 0.15,
                "detail": f"raw_tx_count={raw_tx_count[i]:.0f} (p90={raw_tx_count_p90:.0f}), avg_tx_size=${avg_sz[i]:.0f} (p30=${avg_sz_p30:.0f}), total_flow=${total_flow[i]:.0f}",
            })
        if r42_mask[i]:
            triggers.append({
                "rule": "R42_EXCHANGE_FUNNEL",
                "points": 0.25,
                "detail": f"exchange_outflow_ratio={exchange_out_ratio[i]:.2f}, passthrough_count={passthrough_ct[i]:.0f}",
            })
        if r43_mask[i]:
            triggers.append({
                "rule": "R43_INCOME_MISMATCH",
                "points": 0.20,
                "detail": f"declared_income=${declared_income[i]:.0f} (p40=${income_p40:.0f}), total_flow=${total_flow[i]:.0f} (p90=${total_flow_p90_r43:.0f})",
            })
        if r44_mask[i]:
            triggers.append({
                "rule": "R44_BALANCE_CYCLING",
                "points": 0.15,
                "detail": f"balance_cycles={balance_cycles_arr[i]:.0f}, balance_retention={bal_retention[i]:.2f}",
            })
        if age_mod[i] != 1.0 and raw[i] / age_mod[i] > 0:
            triggers.append({
                "rule": "R6_AGE_MODIFIER",
                "points": age_mod[i],
                "detail": f"person_age={p_age[i]:.0f}, modifier={age_mod[i]}x",
            })
        if strong_count[i] >= 3:
            triggers.append({"rule": "BONUS_STRONG_RULES", "points": 0.25, "detail": f"strong_rules={int(strong_count[i])}"})
        elif strong_count[i] >= 2:
            triggers.append({"rule": "BONUS_STRONG_RULES", "points": 0.10, "detail": f"strong_rules={int(strong_count[i])}"})

        triggered_rules[i] = triggers

    return rule_scores, triggered_rules
