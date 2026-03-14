"""Load IBM AML transactions from local Kaggle cache."""

from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, DATASET_DIR, DEFAULT_DATASET_FILE


def get_dataset_path(file_name=None):
    """Resolve transaction CSV path: try Data/ then kagglehub cache."""
    name = file_name or DEFAULT_DATASET_FILE
    for base in (DATA_DIR, DATASET_DIR):
        csv_path = base / name
        if csv_path.exists():
            return csv_path
    raise FileNotFoundError(
        f"Dataset file not found: {name}. Looked in {DATA_DIR} and {DATASET_DIR}."
    )


def load_transactions(file_name=None):
    """Read selected transaction CSV into a pandas DataFrame."""
    path = get_dataset_path(file_name=file_name)
    return pd.read_csv(path)
