"""Dataset loader: IBM (via src.data.ibm_loader) or local CSV path."""

from pathlib import Path

import pandas as pd


def load_dataset(source: str) -> pd.DataFrame:
    """
    Load transaction dataset.
    - source == "ibm": use configured IBM/Kaggle dataset via src.data.ibm_loader.
    - else: treat source as path to local CSV.
    """
    if source.strip().lower() == "ibm":
        try:
            from src.data.ibm_loader import load_transactions
            return load_transactions()
        except ImportError:
            # Fallback: try project-relative path
            from app.config import PROJECT_ROOT
            path = PROJECT_ROOT / "kagglehub_cache" / "datasets" / "ealtman2019" / "ibm-transactions-for-anti-money-laundering-aml" / "versions" / "8" / "HI-Small_Trans.csv"
            if path.exists():
                return pd.read_csv(path)
            raise FileNotFoundError(
                "IBM dataset not found. Set PYTHONPATH to project root or provide a CSV path as source."
            )
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path)
