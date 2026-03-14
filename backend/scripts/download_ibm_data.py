"""
Download the IBM AML transaction dataset into datasets/ibm_aml so the backend pipeline can find it.

Run once:  python backend/scripts/download_ibm_data.py

Requires:  pip install kagglehub
Kaggle:    https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
You may need to accept the dataset terms on Kaggle and have credentials set up.
"""

import os
import shutil
from pathlib import Path

# backend/scripts/ -> backend/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TARGET_DIR = PROJECT_ROOT / "datasets" / "ibm_aml"
CACHE = PROJECT_ROOT / "kagglehub_cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ["KAGGLEHUB_CACHE"] = str(CACHE.resolve())

import kagglehub

DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
DEFAULT_FILE = "HI-Small_Trans.csv"


def main():
    print("Downloading IBM AML dataset (this may take a moment)...")
    path_str = kagglehub.dataset_download(DATASET)
    path = Path(path_str)
    print(f"Downloaded to: {path}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for f in path.iterdir():
        if f.is_file():
            shutil.copy2(f, TARGET_DIR / f.name)
            print(f"Copied {f.name} to {TARGET_DIR}")
    if (path / DEFAULT_FILE).exists():
        print("You can now run:  python backend/run_pipeline.py --source ibm")
    else:
        print("Run pipeline with:  python backend/run_pipeline.py --source", TARGET_DIR / list(TARGET_DIR.iterdir())[0].name if list(TARGET_DIR.iterdir()) else "")


if __name__ == "__main__":
    main()
