"""Checkpoint save/load for resumable pipeline execution."""

import logging
import joblib

from src.config import CHECKPOINT_DIR

log = logging.getLogger("aml_pipeline")


def save_checkpoint(name, data):
    """Save a checkpoint to disk."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{name}.pkl"
    joblib.dump(data, path, compress=3)
    log.debug(f"Checkpoint saved: {path} ({path.stat().st_size / 1024:.0f} KB)")


def load_checkpoint(name):
    """Load a checkpoint from disk. Returns None if not found."""
    path = CHECKPOINT_DIR / f"{name}.pkl"
    if path.exists():
        log.info(f"Resuming from checkpoint: {path}")
        return joblib.load(path)
    return None


def has_checkpoint(name):
    """Check if a checkpoint exists."""
    return (CHECKPOINT_DIR / f"{name}.pkl").exists()


def clear_checkpoints():
    """Remove all checkpoints."""
    if CHECKPOINT_DIR.exists():
        for f in CHECKPOINT_DIR.glob("*.pkl"):
            f.unlink()
        log.info("All checkpoints cleared.")
