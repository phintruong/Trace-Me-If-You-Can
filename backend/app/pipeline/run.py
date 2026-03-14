"""Run inference pipeline: load -> preprocess -> score -> persist (DB + parquet)."""

import logging
from pathlib import Path

import pandas as pd

from app.config import DATASET_SOURCE, OUTPUT_DIR, PREDICTIONS_PARQUET
from app.pipeline.loader import load_dataset
from app.pipeline.preprocess import preprocess
from app.pipeline.model_runner import score
from app.services.db_client import save_predictions

logger = logging.getLogger(__name__)


def run_pipeline(source: str = None) -> pd.DataFrame:
    """
    Load dataset, preprocess, score, save to DB and predictions.parquet.
    Returns the scored DataFrame.
    """
    src = source or DATASET_SOURCE
    logger.info("Loading dataset from source=%s", src)
    df = load_dataset(src)
    logger.info("Loaded %d rows", len(df))

    df = preprocess(df)
    logger.info("Preprocessed; columns: %s", list(df.columns))

    scored = score(df)
    logger.info("Scored; risk_score range [%s, %s]", scored["risk_score"].min(), scored["risk_score"].max())

    # Persist to DB
    save_predictions(scored)
    logger.info("Saved predictions to DB")

    # Write parquet: keep API-needed columns; top_features as string for parquet compatibility
    out_df = scored[["transaction_id", "account_id", "timestamp", "amount", "risk_score"]].copy()
    if "top_features" in scored.columns:
        out_df["top_features"] = scored["top_features"].apply(
            lambda x: str(x) if x is not None else "[]"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_PARQUET
    out_df.to_parquet(out_path, index=False)
    logger.info("Wrote %s", out_path)

    return scored
