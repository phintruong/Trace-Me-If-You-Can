"""Configuration for the IBM AML transaction-dataset pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
DATASET_DIR = PROJECT_ROOT / "kagglehub_cache" / "datasets" / "ealtman2019" / "ibm-transactions-for-anti-money-laundering-aml" / "versions" / "8"
DEFAULT_DATASET_FILE = "HI-Small_Trans.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"

IBM_REQUIRED_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]

MODEL_FEATURE_COLUMNS = [
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Hour",
    "DayOfWeek",
    "Day",
    "Month",
]

RANDOM_STATE = 42
MAX_ROWS = 300_000
TEST_SIZE = 0.2
RF_N_ESTIMATORS = 200
