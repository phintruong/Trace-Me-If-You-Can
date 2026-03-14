"""
Central configuration for the AML risk scoring pipeline.

All constants, paths, and parameters live here — single source of truth.
"""

import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_RAW = PROJECT_ROOT / "data_o"
DATA_PROCESSED = PROJECT_ROOT / "data_f"
OUTPUT_DIR = DATA_PROCESSED / "risk_scores"
CHECKPOINT_DIR = DATA_PROCESSED / "checkpoints"
LOG_DIR = DATA_PROCESSED / "logs"

# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------
REFERENCE_DATE = pd.Timestamp("2025-01-31")
OBSERVATION_WINDOW_DAYS = 90  # rolling window for feature computation

# ---------------------------------------------------------------------------
# Transaction file mapping  (filename → type label)
# ---------------------------------------------------------------------------
TRANSACTION_FILES = {
    "card.csv": "card",
    "eft.csv": "eft",
    "emt.csv": "emt",
    "abm.csv": "abm",
    "wire.csv": "wire",
    "cheque.csv": "cheque",
    "westernunion.csv": "westernunion",
}

# ---------------------------------------------------------------------------
# Isolation Forest parameters (Agent 1 — default)
# ---------------------------------------------------------------------------
IF_N_ESTIMATORS = 300
IF_MAX_SAMPLES = 256
IF_RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Isolation Forest parameters (Agent 2 — strict)
# ---------------------------------------------------------------------------
IF_STRICT_N_ESTIMATORS = 500
IF_STRICT_MAX_SAMPLES = 512
IF_STRICT_CONTAMINATION = 0.07
IF_STRICT_RANDOM_STATE = 123

# ---------------------------------------------------------------------------
# Sparse Statistical agent parameters (Agent 3)
# ---------------------------------------------------------------------------
SPARSE_FEATURES = [
    "tx_velocity", "io_ratio", "burst_score", "repeat_amt_ratio",
    "flow_acceleration", "max_tx_fraction", "pct_round", "flow_per_account_day",
    "cash_in_ratio", "cash_out_ratio", "cash_cycle_ratio",
]
SPARSE_TOP_K = 5  # number of top |z|-scores to aggregate per customer
SPARSE_EPS = 1e-6  # epsilon for MAD division (separate from feature EPS)
ZSCORE_CAP = 8.0   # cap absolute z-scores to prevent saturation

# ---------------------------------------------------------------------------
# LOF agent parameters (Agent 4 — density-based)
# ---------------------------------------------------------------------------
LOF_N_NEIGHBORS = 20            # k for local density estimation
LOF_CONTAMINATION = "auto"      # let sklearn decide the threshold

# ---------------------------------------------------------------------------
# Aggregation — weighted formula: agent + rules + context
# ---------------------------------------------------------------------------
# 4-component formula: agent + rules + context + supervised (supervised_mod)
# Tuned via grid search: supervised=0.60 catches 9/10 known launderers as HIGH
# with only 386 total HIGH (vs 2/10 HIGH without supervised).
# Agent/rules/context share remaining 0.40 equally.
SCORE_WEIGHT_AGENT = 0.13        # weight for ensemble agent score
SCORE_WEIGHT_RULES = 0.13        # weight for rule engine score
SCORE_WEIGHT_CONTEXT = 0.14      # weight for contextual risk score
SCORE_WEIGHT_SUPERVISED = 0.60   # weight for supervised model score

# Conditional weighting: when rule_score == 0, redistribute rule weight
# proportionally to agent + context + supervised.
# 0.13 / (0.13 + 0.14 + 0.60) = 0.149, 0.14 / 0.87 = 0.161, 0.60 / 0.87 = 0.690
ZERO_RULE_WEIGHT_AGENT = 0.15
ZERO_RULE_WEIGHT_CONTEXT = 0.16
ZERO_RULE_WEIGHT_SUPERVISED = 0.69

# Model override floor: if agent_combined >= threshold, guarantee
# final_score >= floor score (MEDIUM tier) regardless of other components.
MODEL_OVERRIDE_FLOOR_THRESHOLD = 0.65
MODEL_OVERRIDE_FLOOR_SCORE = 0.30

# Supervised_mod hidden launderers: push supervised_mod-flagged + 2+ launderer signs into MEDIUM (or HIGH if sup > 0.9).
SUPERVISED_MOD_HIDDEN_SUPERVISED_MIN = 0.5    # supervised_mod "flagged" threshold
SUPERVISED_MOD_HIDDEN_MIN_SIGNS = 2           # minimum launderer signs (or hard rule) to apply floor
SUPERVISED_MOD_HIDDEN_MEDIUM_FLOOR = 0.35     # floor so they land in MEDIUM
SUPERVISED_MOD_HIDDEN_HIGH_SUPERVISED_MIN = 0.9  # supervised > this -> floor to HIGH (0.55)

# Rule-heavy floor: move LOW accounts with many launderer signs up to MEDIUM even when supervised < 0.5.
RULE_HEAVY_MIN_SIGNS = 5             # 5+ signs -> floor to MEDIUM
RULE_HEAVY_MIN_SIGNS_IF_HARD = 4     # 4+ signs and hard rule -> floor to MEDIUM
RULE_HEAVY_MEDIUM_FLOOR = 0.31      # floor so they land in MEDIUM [bar, 0.55)


# Raw rule score cap — prevents saturation when many rules fire.
# Without a cap, rule scores trivially hit 1.0 (theoretical max ~5.5 with all
# rules + job scaling), destroying dynamic range. Cap at 1.0 so rules
# contribute at most 0.30 * 1.0 = 0.30 to the final score.
RULE_SCORE_CAP = 1.0

# Rule floor factor: final = max(weighted_formula, RULE_FLOOR * rule_score).
# When expert rules are confident, low agent scores shouldn't drag the final
# score below what rules alone would justify.
RULE_FLOOR_FACTOR = 0.55  # floor guarantees near-MEDIUM when rules are strong

# ---------------------------------------------------------------------------
# R39-R44 New rule thresholds (configurable)
# ---------------------------------------------------------------------------

# R39: Dormant Reactivation — long-inactive account suddenly resumes high activity
R39_INACTIVE_DAYS_MIN = 180       # account must have been inactive >= 6 months
R39_RECENT_FLOW_PERCENTILE = 85   # total_flow must be above this percentile
R39_VELOCITY_PERCENTILE = 80      # tx_velocity must be above this percentile

# R41: Micro-Structuring — many small transactions summing to large flows
R41_TX_COUNT_PERCENTILE = 90      # raw_tx_count must exceed this percentile
R41_AVG_SIZE_PERCENTILE = 30      # avg_tx_size must be below this percentile
R41_TOTAL_FLOW_PERCENTILE = 70    # total_flow must exceed this percentile

# R42: Exchange Funnel — funds exiting via cross-border channels with passthrough
R42_EXCHANGE_OUTFLOW_RATIO_MIN = 0.40   # >= 40% of outflow via wire+WU
R42_PASSTHROUGH_MIN = 2                 # sustained_passthrough_count >= this

# R43: Income Mismatch — declared income inconsistent with transaction volume
R43_INCOME_PERCENTILE = 40        # declared_income below this percentile = "low"
R43_TOTAL_FLOW_PERCENTILE = 90    # total_flow must exceed this percentile

# R44: Balance Cycling — repeated fill-then-drain layering pattern
R44_CYCLE_MIN = 5                 # minimum balance_cycles to trigger
R44_RETENTION_MAX = 0.20          # balance_retention must be below this

# Balance cycle feature thresholds (used in engine.py)
BALANCE_CYCLE_FILL_THRESHOLD = 0.60   # fraction of peak pseudo-balance = "filled"
BALANCE_CYCLE_DRAIN_THRESHOLD = 0.10  # fraction of peak pseudo-balance = "drained"

# R38 convergence minimum flags (configurable)
R38_CONVERGENCE_MIN_FLAGS = 4

# Ghost account damping — minimum transaction count for full scoring
GHOST_ACCOUNT_MIN_TX = 5

# Per-agent weights for diversity-weighted averaging.
# Order must match the agents list in runner.py: [IF, sparse_statistical, lof_density]
# IF_strict removed — 98.7% correlated with IF (redundant, wasted diversity).
# Sparse elevated: best launderer separation (0.61 mean vs 0.30 for IF).
AGENT_WEIGHTS = [0.20, 0.55, 0.25]  # [IF, sparse_statistical, lof_density]

# Risk tier thresholds (3-tier system)
RISK_TIER_LOW = 0.31            # below this → LOW; MEDIUM = [0.31, 0.55)
RISK_TIER_MEDIUM = 0.55         # HIGH = final_score >= this
# 0.55 after supervised integration — captures 9/10 known launderers as HIGH.

# ---------------------------------------------------------------------------
# Rule diversity gating — prevent correlated rule stacking from producing HIGH
# ---------------------------------------------------------------------------
RULE_CATEGORIES = {
    "STRUCTURE":   {"R21_REPEATED_CHANNEL_LOOP", "R27_GEOGRAPHIC_LAYERING",
                    "R17_CHANNEL_SWITCHING", "R24_COUNTERPARTY_REUSE",
                    "R41_MICRO_STRUCTURING"},
    "TIMING":      {"R1_RAPID_PASSTHROUGH", "R12_HIGH_IO_SHORT_GAP",
                    "R8_OFFHOURS_ACTIVITY", "R3_SUDDEN_SPIKE_ESTABLISHED",
                    "R39_DORMANT_REACTIVATION"},
    "CASH":        {"R20_CASH_CONVERSION", "R13_HIGH_CASH_INFLOW",
                    "R14_HIGH_CASH_OUTFLOW", "R15_CASH_CYCLE"},
    "ACCOUNT_KYC": {"R10_NEW_ACCOUNT_HIGH_FLOW", "R28_NEW_ACCOUNT_LAYERING",
                    "R31_NEW_ACCOUNT_DEFICIT", "R25_JOB_INCONSISTENCY",
                    "R26_STUDENT_HIGH_WEALTH", "R43_INCOME_MISMATCH"},
    "BALANCE":     {"R23_BALANCE_DRAIN", "R22_SUSTAINED_PASSTHROUGH",
                    "R29_PASSTHROUGH_NO_RETAIL", "R44_BALANCE_CYCLING"},
    "FLOW":        {"R9_EXTREME_FLOW_IMBALANCE", "R18_NO_RETAIL_USAGE",
                    "R30_OFFHOURS_BURST_UNIQUE"},
    "WIRE":        {"R35_HIGH_WIRE_COUNT", "R36_HIGH_WIRE_VELOCITY",
                    "R37_WIRE_CLUSTERING"},
    "DEMOGRAPHIC": {"R32_YOUNG_NEW_ACCOUNT", "R33_SHORT_TENURE_WIRE",
                    "R34_STUDENT_HIGH_FLOW"},
    "COMPOSITE":   {"R38_MULTI_SIGNAL_CONVERGENCE"},
    "NETWORK":     {"R42_EXCHANGE_FUNNEL"},
}
DIVERSITY_MIN_CATEGORIES = 2        # distinct categories needed for HIGH eligibility
DIVERSITY_MIN_VELOCITY = 3.0        # tx_velocity floor for HIGH eligibility
STRONG_RULES_FOR_OVERRIDE = {       # bypass category count (still need >=1 category)
    "R3_SUDDEN_SPIKE_ESTABLISHED", "R12_HIGH_IO_SHORT_GAP",
    "R15_CASH_CYCLE", "R20_CASH_CONVERSION", "R25_JOB_INCONSISTENCY",
    "R29_PASSTHROUGH_NO_RETAIL", "R30_OFFHOURS_BURST_UNIQUE",
    "R38_MULTI_SIGNAL_CONVERGENCE",
    "R39_DORMANT_REACTIVATION", "R42_EXCHANGE_FUNNEL",
    "R43_INCOME_MISMATCH",
}

# ---------------------------------------------------------------------------
# Log-transform columns (heavy-tailed features, applied before modeling)
# ---------------------------------------------------------------------------
LOG_TRANSFORM_COLUMNS = [
    "total_inflow", "total_outflow", "avg_tx_size", "std_tx_size",
    "flow_per_account_day", "cash_in_total", "cash_out_total",
    "wire_total_volume",
]

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
CHUNK_SIZE = 100_000            # rows per chunk for large CSV reads
EPS = 1.0                       # epsilon for division safety

# ---------------------------------------------------------------------------
# Gemini LLM (human-readable reports)
# ---------------------------------------------------------------------------
GEMINI_API_KEYS = [
    v for k in ("GEMINI_KEY1", "GEMINI_KEY2", "GEMINI_KEY3", "GEMINI_KEY4", "GEMINI_KEY5")
    if (v := os.environ.get(k, "")) and v != f"your-key-{k[-1]}-here"
]
GEMINI_MODEL = "gemini-2.5-flash"
ENABLE_LLM_REPORTS = True

# ---------------------------------------------------------------------------
# Feature columns used by the model (22 original + 5 cash + 5 quiet laundering)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "total_inflow", "total_outflow", "io_ratio",
    "tx_velocity", "avg_tx_size", "std_tx_size", "cv_tx_size",
    "median_in_out_gap_hours", "pct_round", "repeat_amt_ratio",
    "burst_score", "max_tx_fraction", "offhours_ratio",
    "account_age_days", "person_age_years",
    "flow_per_account_day", "flow_acceleration",
    "cash_in_total", "cash_out_total",
    "cash_in_ratio", "cash_out_ratio", "cash_cycle_ratio",
    "channel_switch_ratio", "retail_ratio", "distinct_provinces",
    "atm_in_total", "card_out_total",
    "channel_loop_count", "balance_retention",
    "merchant_concentration", "sustained_passthrough_count",
    "wire_tx_count", "wire_total_volume", "wire_velocity",
    "wire_volume_ratio", "wire_max_weekly_count",
    # New features for R39-R44 + ghost damping
    "inactive_days", "exchange_outflow_ratio", "raw_tx_count",
    "declared_income", "balance_cycles", "outflow_count",
]
