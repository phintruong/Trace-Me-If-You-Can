"""Graceful shutdown handling for the AML pipeline."""

import sys
import signal
import logging

log = logging.getLogger("aml_pipeline")

_shutdown_requested = False
_current_stage = "initializing"
_completed_stages = []


def _handle_signal(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    if _shutdown_requested:
        log.warning("Second interrupt received — forcing exit.")
        sys.exit(1)
    _shutdown_requested = True
    log.warning("Shutdown requested (Ctrl+C). Finishing current operation...")
    log.warning(f"  Current stage: {_current_stage}")
    log.warning(f"  Completed stages: {_completed_stages}")
    log.warning("Press Ctrl+C again to force quit.")


def install_signal_handlers():
    """Install Ctrl+C and SIGBREAK handlers."""
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_signal)


def check_shutdown(stage_name=""):
    """Check if shutdown was requested; if so, save state and exit cleanly."""
    if _shutdown_requested:
        log.warning(f"Graceful shutdown at stage: {stage_name or _current_stage}")
        log.warning(f"Completed stages: {_completed_stages}")
        log.warning("Checkpoints saved. Re-run with --resume to continue.")
        sys.exit(0)


def set_stage(name):
    """Update the current pipeline stage for shutdown reporting."""
    global _current_stage
    _current_stage = name
    log.info(f"{'='*50}")
    log.info(f"STAGE: {name}")
    log.info(f"{'='*50}")


def complete_stage(name):
    """Mark a stage as complete."""
    _completed_stages.append(name)
    log.info(f"Stage complete: {name}")
