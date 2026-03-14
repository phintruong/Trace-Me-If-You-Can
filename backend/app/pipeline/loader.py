"""Load AML dataset from datasets/ibm_aml or local CSV path."""

from pathlib import Path

import pandas as pd

from app.config import DATASET_SOURCE, DATASETS_DIR, DEFAULT_DATASET_FILE


def get_dataset_path(source: str | None = None, file_name: str | None = None) -> Path:
    """Resolve CSV path: datasets/ibm_aml or explicit file path."""
    src = (source or DATASET_SOURCE).strip()
    if src.lower() == "ibm":
        name = file_name or DEFAULT_DATASET_FILE
        path = DATASETS_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}. Place IBM AML CSV in {DATASETS_DIR} or set DATASET_SOURCE to a CSV path."
            )
        return path
    path = Path(src)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path


def load_dataset(source: str | None = None, file_name: str | None = None) -> pd.DataFrame:
    """Load transaction CSV. source='ibm' uses datasets/ibm_aml, else path to CSV."""
    path = get_dataset_path(source=source, file_name=file_name)
    return pd.read_csv(path)
