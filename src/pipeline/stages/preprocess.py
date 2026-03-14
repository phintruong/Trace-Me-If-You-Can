"""Stage 2: Preprocess transactions (features + target)."""

from src.pipeline.types import PipelineContext
from src.features.engine import build_model_matrix


def stage_preprocess(ctx: PipelineContext) -> PipelineContext:
    """Build feature matrix and target from raw_df. Sets ctx.feature_matrix, ctx.target_series, ctx.feature_df."""
    if ctx.raw_df is None:
        raise ValueError("Stage 1 (load_data) must run before preprocess.")
    X, y = build_model_matrix(ctx.raw_df)
    ctx.feature_matrix = X
    ctx.target_series = y
    ctx.feature_df = X.copy()
    return ctx
