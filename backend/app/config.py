"""Load configuration from environment. No secrets in repo."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Database
DB_CONN_STRING = os.getenv("DB_CONN_STRING", "")
DB_MODE = os.getenv("DB_MODE", "sqlite").lower()
if DB_MODE not in ("sqlite", "db2"):
    DB_MODE = "sqlite"

# Default SQLite path when DB_MODE=sqlite and no DB_CONN_STRING
_BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = _BACKEND_ROOT.parent.parent  # GenAI-Genesis
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "outputs" / "fraud_backend.db"

# Model
MODEL_URL = os.getenv("MODEL_URL", "")
MODEL_PATH = os.getenv("MODEL_PATH", "")

# Risk
RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.7"))
RISK_THRESHOLD = max(0.0, min(1.0, RISK_THRESHOLD))

# Watsonx
WATSONX_URL = os.getenv("WATSONX_URL", "")
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "granite-13b-instruct")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Cache
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))

# Pipeline output
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
PREDICTIONS_PARQUET = OUTPUT_DIR / "predictions.parquet"

# Dataset source for pipeline (ibm = use ibm_loader, else path to CSV)
DATASET_SOURCE = os.getenv("DATASET_SOURCE", "ibm")
