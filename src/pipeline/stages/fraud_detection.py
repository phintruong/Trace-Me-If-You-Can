"""Stage 3: Run fraud detection model (Random Forest). Predictions on full feature matrix or train/test split."""

from src.pipeline.types import PipelineContext
from src.model.train import train_random_forest


def stage_fraud_detection(ctx: PipelineContext) -> PipelineContext:
    """
    Train RF and predict on test set. Sets ctx.model, ctx.pred_label, ctx.pred_probability, ctx.test_indices.
    Predictions align with ctx.feature_df by index (test subset).
    """
    if ctx.feature_matrix is None or ctx.target_series is None:
        raise ValueError("Stage 2 (preprocess) must run before fraud_detection.")
    X = ctx.feature_matrix
    y = ctx.target_series
    model, X_train, X_test, y_train, y_test = train_random_forest(X, y)
    ctx.model = model
    proba = model.predict_proba(X_test)[:, 1]
    ctx.pred_label = (proba >= 0.5).astype(int)
    ctx.pred_probability = proba
    ctx.test_indices = X_test.index
    return ctx
