"""
Configurable thresholds for job-based AML risk modifiers.

All job-related tunables live here — single source of truth.
Adjust values to calibrate false-positive / recall tradeoff.
"""

JOB_CONFIG = {
    # ── Context score modifiers (added in compute_context_score) ──────────
    # Positive = increase suspicion, negative = reduce baseline risk.
    "context_modifiers": {
        "CASH_HEAVY": -0.08,   # cash usage is expected for this occupation
        "SALARIED":    0.05,   # anomalies in cash/velocity are more meaningful
        "BUSINESS":    0.00,   # neutral — already handled by account_type logic
        "STUDENT":     0.12,   # students shouldn't have significant financial activity — any is suspicious
        "UNKNOWN":     0.15,   # can't verify behavioral norms → elevate
    },

    # ── Rule scaling multipliers ──────────────────────────────────────────
    # Multiplied against raw rule points. 1.0 = no change.
    "rule_scaling": {
        # R13/R14/R15 — cash inflow/outflow/cycle
        "cash_rules": {
            "CASH_HEAVY": 0.50,
            "SALARIED":   1.00,
            "STUDENT":    1.00,
            "BUSINESS":   1.00,
            "UNKNOWN":    1.00,
        },
        # R1 (rapid pass-through), R11 (high flow + velocity), R12 (high IO + short gap)
        "velocity_passthrough_rules": {
            "CASH_HEAVY": 1.00,
            "SALARIED":   1.25,
            "STUDENT":    1.30,
            "BUSINESS":   1.00,
            "UNKNOWN":    1.00,
        },
        # R7 (large tx concentration), R9 (extreme flow imbalance)
        "large_flow_rules": {
            "CASH_HEAVY": 1.00,
            "SALARIED":   1.00,
            "STUDENT":    1.40,
            "BUSINESS":   1.00,
            "UNKNOWN":    1.00,
        },
        # R8 (off-hours), R18 (no retail), R22 (sustained passthrough),
        # R29 (passthrough no retail) — normal behavior for business accounts
        "business_behavior_rules": {
            "CASH_HEAVY": 1.00,
            "SALARIED":   1.00,
            "STUDENT":    1.00,
            "BUSINESS":   0.40,
            "UNKNOWN":    1.00,
        },
    },

    # ── R25: Job–Behavior Inconsistency ───────────────────────────────────
    "inconsistency": {
        # STUDENT with high flow/velocity/balance (percentile thresholds)
        "student_high_flow_p":     90,
        "student_high_velocity_p": 90,
        "student_high_balance_p":  90,
        "student_penalty":         0.25,
        # SALARIED with heavy cash inflow
        "salaried_cash_in_ratio":  0.50,
        "salaried_penalty":        0.20,
        # CASH_HEAVY with low retail + rapid pass-through
        "cash_heavy_low_retail":   0.05,
        "cash_heavy_high_passthrough_gap": 48,  # hours
        "cash_heavy_penalty":      0.20,
    },

    # ── R26: Student High Wealth ──────────────────────────────────────────
    "student_wealth": {
        "flow_percentile":     95,
        "velocity_percentile": 95,
        "balance_percentile":  95,
        "tier1_points":        0.25,   # 1 threshold exceeded
        "tier2_points":        0.35,   # 2+ thresholds exceeded
    },
}
