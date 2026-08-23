"""Command-line entry point for slv.

Phase 1 added `slv ingest`. Phase 2 adds `slv compute`. Later phases add
`brief`, `thesis`, and `journal` subcommands.
"""
from __future__ import annotations

import argparse
import math
import sys

import pandas as pd

from slv import config, db
from slv.compute import indicators
from slv.fetch import calendar, cot, fred, prices

INDICATORS_COLUMNS = (
    "atr14",
    "ema20",
    "ema50",
    "sma200",
    "rsi14",
    "rvol20",
    "gsr",
    "dist_ema20_atr",
    "range_pctile_60d",
)


def ingest() -> None:
    """Run all four raw-data fetchers against the shared DB connection.

    Each fetcher commits its own rows before the next one starts, so a
    failure partway through (e.g. FRED down) leaves already-fetched data
    intact and un-duplicated on the next run — no partial writes within any
    single fetcher, and the run as a whole is safe to retry.
    """
    conn = db.connect()
    try:
        for symbol, n in prices.fetch_all(conn).items():
            print(f"prices ({symbol}): {n} rows")

        n = fred.fetch_all(conn)
        print(f"macro:  {n} rows")

        n = cot.fetch_cot(conn)
        print(f"cot:    {n} rows")

        n = calendar.load_calendar(conn)
        print(f"events: {n} rows")
    finally:
        conn.close()


def _read_prices(conn, symbol: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM prices "
        "WHERE symbol = ? ORDER BY date",
        conn,
        params=(symbol,),
        index_col="date",
    )


def _sql_value(x: float) -> float | None:
    """NaN (not-yet-enough-history, or a missing aligned date) -> NULL."""
    if x is None:
        return None
    if math.isnan(x):
        return None
    return float(x)


def compute() -> None:
    """Rebuild the indicators table from scratch out of stored prices.

    Derived table (see CLAUDE.md) — always safe to drop and recompute, so
    this deletes all rows and reinserts rather than trying to diff/upsert.
    """
    conn = db.connect()
    try:
        silver = _read_prices(conn, config.INSTRUMENT_SYMBOL)
        if silver.empty:
            raise RuntimeError(
                f"no price data for {config.INSTRUMENT_SYMBOL!r}; run `slv ingest` first"
            )
        gold = _read_prices(conn, config.GOLD_SYMBOL)

        close, high, low = silver["close"], silver["high"], silver["low"]
        atr = indicators.atr14(high, low, close)
        ema20 = indicators.ema20(close)

        out = pd.DataFrame(
            {
                "atr14": atr,
                "ema20": ema20,
                "ema50": indicators.ema50(close),
                "sma200": indicators.sma200(close),
                "rsi14": indicators.rsi14(close),
                "rvol20": indicators.rvol20(close),
                # gold's dates can differ slightly from silver's (holiday
                # calendars) -- align to silver's index, NaN where missing.
                "gsr": indicators.gsr(gold["close"].reindex(close.index), close),
                "dist_ema20_atr": indicators.dist_ema20_atr(close, ema20, atr),
                "range_pctile_60d": indicators.range_pctile_60d(high, low, close),
            }
        )

        rows = [
            (date,) + tuple(_sql_value(v) for v in row)
            for date, row in zip(out.index, out.itertuples(index=False))
        ]

        conn.execute("DELETE FROM indicators")
        conn.executemany(
            "INSERT INTO indicators (date, " + ", ".join(INDICATORS_COLUMNS) + ") "
            "VALUES (" + ", ".join("?" * (1 + len(INDICATORS_COLUMNS))) + ")",
            rows,
        )
        conn.commit()
        print(f"indicators: {len(rows)} rows")
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="slv")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="fetch raw data (idempotent)")
    subparsers.add_parser("compute", help="rebuild derived indicators from stored prices")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        ingest()
    elif args.command == "compute":
        compute()

    return 0


if __name__ == "__main__":
    sys.exit(main())
