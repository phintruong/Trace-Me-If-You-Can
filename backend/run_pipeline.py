"""
Run the fraud inference pipeline (load -> preprocess -> score -> DB + parquet).
Usage (from repo root with PYTHONPATH=. or from backend/):
  python -m backend.run_pipeline
  python backend/run_pipeline.py --source ibm
  python backend/run_pipeline.py --source /path/to/local.csv
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure project root on path for src.* and app.*
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import logging
from app.pipeline.run import run_pipeline

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run fraud inference pipeline")
    parser.add_argument("--source", default=os.getenv("DATASET_SOURCE", "ibm"), help="ibm or path to CSV")
    args = parser.parse_args()
    run_pipeline(source=args.source)


if __name__ == "__main__":
    main()
