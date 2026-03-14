"""Pytest fixtures and path setup."""
import os
import sys
from pathlib import Path

# Add project root and backend to path
BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
for p in (str(ROOT), str(BACKEND)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Prefer SQLite in-memory for tests
os.environ.setdefault("DB_MODE", "sqlite")
os.environ.setdefault("DB_CONN_STRING", ":memory:")
os.environ.setdefault("RISK_THRESHOLD", "0.7")
