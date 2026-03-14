"""
AML Risk Scoring Pipeline — entry point.

Usage:
    python run_pipeline.py                    # Score all customers
    python run_pipeline.py SYNID0100957188    # Score single customer
    python run_pipeline.py --resume           # Resume from last checkpoint
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.runner import main

if __name__ == "__main__":
    main()
