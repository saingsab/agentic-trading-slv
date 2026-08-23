"""Load the hand-seeded event calendar (data/calendar_seed.csv) into events.

The seed CSV only carries date and name, filled in a year ahead from
published FOMC/CPI/NFP schedules — this is the free substitute for a paid
economic calendar API (see PLAN.md). actual/consensus/prior/surprise_z start
NULL and are expected to be filled in later, by hand or a future fetcher,
once each release has happened.

Idempotent, and deliberately INSERT OR IGNORE rather than REPLACE: unlike
the other fetchers, existing event rows may carry manually-entered
actual/consensus/prior values that a re-run must not wipe back to NULL.
Re-seeding only adds rows for dates not already present.
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from slv.config import DATA_DIR

SEED_PATH = DATA_DIR / "calendar_seed.csv"


def load_calendar(conn: sqlite3.Connection, seed_path: Path = SEED_PATH) -> int:
    if not seed_path.exists():
        raise RuntimeError(f"calendar seed not found: {seed_path}")

    with seed_path.open(newline="") as f:
        rows = [(row["date"], row["name"]) for row in csv.DictReader(f)]

    if not rows:
        raise RuntimeError(f"calendar seed is empty: {seed_path}")

    conn.executemany(
        "INSERT OR IGNORE INTO events (date, name, actual, consensus, prior, surprise_z) "
        "VALUES (?, ?, NULL, NULL, NULL, NULL)",
        rows,
    )
    conn.commit()
    return len(rows)
