"""Fetch weekly Legacy Futures-Only COT data for silver.

CLAUDE.md flags that CFTC's bulk historical COT CSV changes column names
between years. Rather than parse that file, this hits CFTC's Public
Reporting API (Socrata-backed, publicreporting.cftc.gov) which serves the
same Legacy report under a stable, documented schema. We still check for
the fields we need before using them, in case CFTC changes that schema too.

Idempotent: rows are keyed on report_date and written with INSERT OR REPLACE.
"""
from __future__ import annotations

import sqlite3

import requests

COT_API_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
MARKET_NAME = "SILVER - COMMODITY EXCHANGE INC."

# The Legacy report's category is "Non-Commercial" (large speculators), not
# literally "Managed Money" (that's a Disaggregated-report term) — but it's
# the closest Legacy-report analog, which is what PLAN.md's mm_* columns
# are naming.
REQUIRED_FIELDS = (
    "report_date_as_yyyy_mm_dd",
    "noncomm_positions_long_all",
    "noncomm_positions_short_all",
    "open_interest_all",
)


def fetch_cot(conn: sqlite3.Connection, limit: int = 260) -> int:
    """Fetch up to `limit` most recent weekly silver COT reports.

    260 weeks ~= 5 years, matching Phase 1's history requirement for prices.
    """
    resp = requests.get(
        COT_API_URL,
        params={
            "$where": f"market_and_exchange_names = '{MARKET_NAME}'",
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": limit,
        },
        timeout=30,
    )
    resp.raise_for_status()
    records = resp.json()
    if not records:
        raise RuntimeError("CFTC API returned no silver COT records")

    rows = []
    for rec in records:
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            raise RuntimeError(f"CFTC COT record missing expected fields: {missing}")

        report_date = rec["report_date_as_yyyy_mm_dd"][:10]
        mm_long = float(rec["noncomm_positions_long_all"])
        mm_short = float(rec["noncomm_positions_short_all"])
        rows.append(
            (report_date, mm_long - mm_short, mm_long, mm_short, float(rec["open_interest_all"]))
        )

    conn.executemany(
        "INSERT OR REPLACE INTO cot (report_date, mm_net, mm_long, mm_short, open_interest) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)
