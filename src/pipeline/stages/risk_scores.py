"""Stage 6: Generate risk scores per account (combine model score + pattern severity)."""

from collections import defaultdict

import numpy as np

from src.pipeline.types import PipelineContext


def stage_risk_scores(ctx: PipelineContext) -> PipelineContext:
    """
    Aggregate prediction probability per account (from test-set transactions) and boost by patterns.
    Sets ctx.account_risk_scores.
    """
    if ctx.pred_probability is None or ctx.test_indices is None or ctx.raw_df is None:
        raise ValueError("Stage 3 (fraud_detection) and raw_df required before risk_scores.")
    raw_test = ctx.raw_df.loc[ctx.test_indices]
    probs = np.asarray(ctx.pred_probability).ravel()
    account_scores: dict[str, list[float]] = defaultdict(list)
    for i in range(min(len(ctx.test_indices), len(probs))):
        row = raw_test.iloc[i]
        for col in ("Account", "Account.1"):
            if col in row.index:
                acc = str(row[col])
                account_scores[acc].append(probs[i])
    account_risk_scores = {}
    for acc, probs in account_scores.items():
        base = max(probs) if probs else 0.0
        patterns = ctx.account_patterns.get(acc, [])
        boost = 0.0
        if "circular" in patterns:
            boost += 0.15
        if "hub" in patterns:
            boost += 0.1
        if "rapid_movement" in patterns:
            boost += 0.1
        account_risk_scores[acc] = min(1.0, base + boost)
    ctx.account_risk_scores = account_risk_scores
    return ctx
