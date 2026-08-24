"""SQLite schema, connection, and migrations for slv.db.

Schema is from PLAN.md. Two categories of table, enforced by convention here
and by code in later phases (journal.py for theses immutability):

- Raw tables (prices, macro, cot, events) are append-only.
- Derived tables (indicators) can be dropped and rebuilt from raw data.
- theses rows are immutable after creation; corrections create new rows.

`CREATE TABLE IF NOT EXISTS` makes connect() idempotent: calling it against
an existing slv.db is a no-op on the schema, never a destructive migration.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from slv.config import DB_PATH

# entry_zone is stored as text (e.g. "72.50-73.20") rather than two columns
# because it's always displayed and entered as a single range; journal.py
# parses it when checking the in-condition rubric item in Phase 3.
SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    date TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL,
    symbol TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (date, symbol, source)
);

CREATE TABLE IF NOT EXISTS macro (
    date TEXT NOT NULL,
    series_id TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (date, series_id)
);

CREATE TABLE IF NOT EXISTS cot (
    report_date TEXT PRIMARY KEY,
    mm_net REAL,
    mm_long REAL,
    mm_short REAL,
    open_interest REAL
);

CREATE TABLE IF NOT EXISTS events (
    date TEXT NOT NULL,
    name TEXT NOT NULL,
    actual REAL,
    consensus REAL,
    prior REAL,
    surprise_z REAL,
    PRIMARY KEY (date, name)
);

CREATE TABLE IF NOT EXISTS indicators (
    date TEXT PRIMARY KEY,
    atr14 REAL,
    ema20 REAL,
    ema50 REAL,
    sma200 REAL,
    rsi14 REAL,
    rvol20 REAL,
    gsr REAL,
    dist_ema20_atr REAL,
    range_pctile_60d REAL
);

CREATE TABLE IF NOT EXISTS theses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_zone TEXT NOT NULL,
    invalidation REAL NOT NULL,
    target REAL NOT NULL,
    size_pct_equity REAL NOT NULL,
    leverage REAL NOT NULL,
    rationale TEXT,
    claim_types TEXT,
    provenance TEXT,
    status TEXT NOT NULL DEFAULT 'open'
);

-- entry_price is the actual fill (vs. theses.entry_zone, the planned
-- range), captured here because journal.py's `thesis close` is the first
-- point you actually know it. r_multiple/process_score/process_detail_json
-- start NULL and are filled in later by `slv journal score` -- a separate,
-- deliberately batched step (see PLAN.md's weekly cadence), not computed
-- automatically at close.
CREATE TABLE IF NOT EXISTS thesis_outcomes (
    thesis_id INTEGER PRIMARY KEY REFERENCES theses(id),
    closed_at TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    r_multiple REAL,
    process_score REAL,
    process_detail_json TEXT
);

CREATE TABLE IF NOT EXISTS briefs (
    date TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    facts_json TEXT NOT NULL
);

-- holdout_* columns are the audit trail for the sealed holdout (PLAN.md:
-- "unlocking it is a manual CLI action by you, logged, and rare").
-- holdout_metrics_json stays NULL on every run that didn't unlock it --
-- the overwhelming majority, by design.
CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_hash TEXT NOT NULL,
    params_json TEXT NOT NULL,
    train_metrics_json TEXT,
    created_at TEXT NOT NULL,
    holdout_unlocked INTEGER NOT NULL DEFAULT 0,
    holdout_unlock_reason TEXT,
    holdout_metrics_json TEXT
);
"""

# Kept in sync with SCHEMA by hand; test_db.py checks they match.
TABLES = (
    "prices",
    "macro",
    "cot",
    "events",
    "indicators",
    "theses",
    "thesis_outcomes",
    "briefs",
    "backtest_runs",
)


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive, idempotent patches for tables created under an older
    schema version. A brand-new DB never touches this -- CREATE TABLE
    already has every current column, so each check below is a no-op.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(thesis_outcomes)")}
    if columns and "entry_price" not in columns:
        conn.execute("ALTER TABLE thesis_outcomes ADD COLUMN entry_price REAL")

    bt_columns = {row[1] for row in conn.execute("PRAGMA table_info(backtest_runs)")}
    if bt_columns and "holdout_unlocked" not in bt_columns:
        conn.execute("ALTER TABLE backtest_runs ADD COLUMN holdout_unlocked INTEGER NOT NULL DEFAULT 0")
        conn.execute("ALTER TABLE backtest_runs ADD COLUMN holdout_unlock_reason TEXT")
        conn.execute("ALTER TABLE backtest_runs ADD COLUMN holdout_metrics_json TEXT")


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a connection to slv.db, creating the file and schema if missing.

    Safe to call repeatedly: existing tables are left untouched beyond the
    additive migrations in _migrate().
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    _migrate(conn)
    conn.commit()
    return conn


if __name__ == "__main__":
    # `python -m slv.db` creates data/slv.db with the current schema.
    connect().close()
    print(f"slv.db ready at {DB_PATH}")
