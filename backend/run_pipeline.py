"""
Run the fraud inference pipeline (load -> preprocess -> graph -> GNN -> DB + parquet -> Railtracks).
Usage (from repo root with PYTHONPATH including project root and backend):
  python -m backend.run_pipeline
  python backend/run_pipeline.py --source ibm
  python backend/run_pipeline.py --source /path/to/local.csv
"""
import argparse
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import logging
from app.pipeline.run_pipeline import run_pipeline

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run fraud inference pipeline")
    parser.add_argument("--source", default=os.getenv("DATASET_SOURCE", "ibm"), help="ibm or path to CSV")
    parser.add_argument("--file-name", default=None, help="Dataset file name when source=ibm")
    parser.add_argument("--risk-threshold", type=float, default=None, help="Risk threshold for flagged accounts")
    parser.add_argument("--max-flagged", type=int, default=50, help="Max flagged accounts")
    args = parser.parse_args()
    run_pipeline(
        source=args.source,
        file_name=args.file_name,
        risk_threshold=args.risk_threshold,
        max_flagged=args.max_flagged,
    )
    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
