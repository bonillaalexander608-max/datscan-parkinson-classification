"""Portable project paths shared by the notebooks."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
DATA_ROOT = Path(
    os.getenv("DATSCAN_DATA_ROOT", PROJECT_ROOT / "data")
).expanduser().resolve()
RESULTS_DIR = PROJECT_ROOT / "results"


def ensure_results_dir() -> Path:
    """Create and return the directory for small reproducible outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR
