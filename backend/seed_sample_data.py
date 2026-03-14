"""
Seed the predictions table with a small sample so you can test Watsonx and the
account API without running the full pipeline (no model required).

Usage:
  From repo root (GenAI-Genesis):
    set PYTHONPATH=%CD%\backend;%CD%
    python backend/seed_sample_data.py

  From backend directory:
    set PYTHONPATH=..
    python seed_sample_data.py

  Then start the API and call:
    curl "http://localhost:8080/account/1"
    curl "http://localhost:8080/account/2"
    curl "http://localhost:8080/account/3"

  Account 1 -> NORMAL, Account 2 -> SUSPICIOUS, Account 3 -> LAUNDERING.
  First request per account triggers Watsonx for aiExplanation (cached after).
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load .env from backend so credentials work when run from repo root
from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

import pandas as pd
from app.services.db_client import init_db, save_predictions

# Sample rows: account_id, transaction_id, timestamp, amount, risk_score, top_features
# Account 1: NORMAL (max risk < 0.7)
# Account 2: SUSPICIOUS (max risk in [0.7, 0.9))
# Account 3: LAUNDERING (max risk >= 0.9)
SAMPLE_ROWS = [
    {"transaction_id": "seed_tx_1a", "account_id": "1", "timestamp": "2024-01-15T10:00:00Z", "amount": 50.0, "risk_score": 0.3, "top_features": [("Amount Paid", 0.1), ("Hour", 0.05)]},
    {"transaction_id": "seed_tx_1b", "account_id": "1", "timestamp": "2024-01-16T14:30:00Z", "amount": 120.0, "risk_score": 0.5, "top_features": [("Amount Paid", 0.2), ("DayOfWeek", 0.08)]},
    {"transaction_id": "seed_tx_2a", "account_id": "2", "timestamp": "2024-01-14T09:00:00Z", "amount": 5000.0, "risk_score": 0.75, "top_features": [("Amount Paid", 0.35), ("Hour", 0.2), ("From Bank", 0.15)]},
    {"transaction_id": "seed_tx_2b", "account_id": "2", "timestamp": "2024-01-15T22:00:00Z", "amount": 800.0, "risk_score": 0.68, "top_features": [("Amount Paid", 0.25), ("Payment Format", 0.12)]},
    {"transaction_id": "seed_tx_3a", "account_id": "3", "timestamp": "2024-01-13T03:00:00Z", "amount": 25000.0, "risk_score": 0.95, "top_features": [("Amount Paid", 0.5), ("Hour", 0.3), ("From Bank", 0.25), ("To Bank", 0.2)]},
]


def main():
    init_db()
    df = pd.DataFrame(SAMPLE_ROWS)
    save_predictions(df)
    print(f"Seeded {len(df)} sample prediction rows (accounts 1, 2, 3).")
    print("Start the API and try: GET /account/1, /account/2, /account/3")


if __name__ == "__main__":
    main()
