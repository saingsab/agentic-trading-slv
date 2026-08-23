"""Paths, environment-derived settings, and constants shared across slv.

No secrets live here. API keys are read from environment variables at
import time and are None if unset — callers that need them (fetch/fred.py,
in a later phase) are responsible for failing loudly if they're missing.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root: slv/config.py -> slv/ -> repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
DB_PATH = DATA_DIR / "slv.db"
BRIEFS_DIR = REPO_ROOT / "briefs"

# Secrets from env vars only. Never hardcode or commit.
FRED_API_KEY = os.environ.get("FRED_API_KEY")

# XAG/USD proxy. SI=F is silver futures, not spot — see CLAUDE.md domain notes.
INSTRUMENT_SYMBOL = "SI=F"
