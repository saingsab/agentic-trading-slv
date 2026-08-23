"""Fetch macro series from the FRED API.

Idempotent: rows are keyed on (date, series_id) and written with
INSERT OR REPLACE.
"""
from __future__ import annotations

import sqlite3

import requests

from slv.config import FRED_API_KEY

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

# 10y TIPS real yield and 10y breakeven inflation — see PLAN.md data sources.
SERIES_IDS = ("DFII10", "T10YIE")


def fetch_series(conn: sqlite3.Connection, series_id: str) -> int:
    if not FRED_API_KEY:
        raise RuntimeError(
            "FRED_API_KEY is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and export it."
        )

    resp = requests.get(
        FRED_URL,
        params={"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    observations = resp.json()["observations"]

    # FRED marks missing observations with "." rather than omitting them.
    rows = [
        (obs["date"], series_id, float(obs["value"]))
        for obs in observations
        if obs["value"] != "."
    ]
    if not rows:
        raise RuntimeError(f"FRED returned no usable observations for {series_id!r}")

    conn.executemany(
        "INSERT OR REPLACE INTO macro (date, series_id, value) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def fetch_all(conn: sqlite3.Connection) -> int:
    return sum(fetch_series(conn, series_id) for series_id in SERIES_IDS)
