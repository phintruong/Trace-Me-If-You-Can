"""Load configuration from environment. Backend-centric paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Paths: backend/ is parent of app/, project root is parent of backend/
_APP_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _APP_DIR.parent
PROJECT_ROOT = _BACKEND_DIR.parent

# Load .env from project root so backend finds it when run from any cwd
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

# Data
DATASETS_DIR = PROJECT_ROOT / "datasets" / "ibm_aml"
DEFAULT_DATASET_FILE = "HI-Small_Trans.csv"
DATASET_SOURCE = os.getenv("DATASET_SOURCE", "ibm")  # "ibm" or path to CSV

# Backend-owned dirs
BACKEND_DIR = _BACKEND_DIR
MODEL_DIR = _BACKEND_DIR / "model"
OUTPUT_DIR = _BACKEND_DIR / "outputs"
PREDICTIONS_PARQUET = OUTPUT_DIR / "predictions.parquet"

# Model
MODEL_URL = os.getenv("MODEL_URL", "")
MODEL_PATH = os.getenv("MODEL_PATH", "") or (str(MODEL_DIR / "run_1_GraphSAGE_A+B_(Synergy).pkl") if (MODEL_DIR / "run_1_GraphSAGE_A+B_(Synergy).pkl").exists() else "")

# DB
DB_CONN_STRING = os.getenv("DB_CONN_STRING", "")
DB_MODE = os.getenv("DB_MODE", "sqlite").lower()
if DB_MODE not in ("sqlite", "db2"):
    DB_MODE = "sqlite"
DEFAULT_SQLITE_PATH = OUTPUT_DIR / "fraud_backend.db"

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
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))

# Pipeline feature columns (must match training)
IBM_REQUIRED_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format",
    "Is Laundering",
]
MODEL_FEATURE_COLUMNS = [
    "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format",
    "Hour", "DayOfWeek", "Day", "Month",
]
