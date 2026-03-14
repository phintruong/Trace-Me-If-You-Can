"""Simple IBM AML pipeline: load data, build features, train RF, print metrics, save model."""

import argparse
from datetime import datetime

import joblib

from src.config import OUTPUT_DIR
from src.data.ibm_loader import get_dataset_path, load_transactions
from src.features.engine import build_model_matrix
from src.model.train import evaluate_model, train_random_forest
from src.utils.logging import setup_logging


def run_pipeline(file_name=None):
    """Execute baseline AML model training for one IBM transaction CSV."""
    logger = setup_logging()

    csv_path = get_dataset_path(file_name=file_name)
    logger.info(f"Loading dataset: {csv_path}")
    df = load_transactions(file_name=file_name)
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

    X, y = build_model_matrix(df)
    logger.info(f"Model matrix ready: X={X.shape}, y={y.shape}")

    model, X_train, X_test, y_train, y_test = train_random_forest(X, y)
    metrics = evaluate_model(model, X_test, y_test)

    print("=" * 60)
    print("IBM AML RANDOM FOREST EVALUATION")
    print("=" * 60)
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print("\nClassification Report:")
    print(metrics["classification_report"])
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"\nROC-AUC: {metrics['roc_auc']:.6f}")
    print(f"PR-AUC (Average Precision): {metrics['pr_auc']:.6f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_df = X_test.copy()
    pred_df["actual_label"] = y_test.values
    pred_df["pred_label"] = model.predict(X_test)
    pred_df["pred_probability"] = model.predict_proba(X_test)[:, 1]
    out_path = OUTPUT_DIR / f"rf_eval_predictions_{ts}.csv"
    pred_df.to_csv(out_path, index=False)
    logger.info(f"Saved test predictions: {out_path}")

    model_path = OUTPUT_DIR / "model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved model: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Run IBM AML RF baseline pipeline")
    parser.add_argument(
        "--file",
        default=None,
        help="Dataset CSV filename under kagglehub_cache/.../versions/8 (default from config)",
    )
    args = parser.parse_args()

    run_pipeline(file_name=args.file)


if __name__ == "__main__":
    main()
