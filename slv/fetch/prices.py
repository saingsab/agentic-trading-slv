"""Fetch daily OHLCV for the configured instrument via yfinance.

Idempotent: rows are keyed on (date, symbol, source) and written with
INSERT OR REPLACE, so re-fetching an overlapping date range converges on
the same state rather than duplicating rows.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import yfinance as yf

from slv.config import INSTRUMENT_SYMBOL

SOURCE = "yfinance"


def fetch_prices(
    conn: sqlite3.Connection,
    symbol: str = INSTRUMENT_SYMBOL,
    period: str = "5y",
) -> int:
    """Download `period` of daily OHLCV for `symbol` and upsert into prices.

    Returns the number of rows written. Raises if yfinance returns nothing —
    never writes a partial/empty result.
    """
    # auto_adjust=False: we want raw OHLC as printed, not yfinance's
    # split/dividend-adjusted series. Futures like SI=F aren't adjusted
    # anyway, but this keeps prices.py's output well-defined either way.
    df = yf.download(
        symbol, period=period, interval="1d", progress=False, auto_adjust=False
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol!r} (period={period!r})")

    # A single-symbol request can still come back with a MultiIndex on
    # columns (Price, Ticker); flatten to plain OHLCV names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    rows = [
        (
            idx.strftime("%Y-%m-%d"),
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            None if pd.isna(row["Volume"]) else float(row["Volume"]),
            symbol,
            SOURCE,
        )
        for idx, row in df.iterrows()
    ]

    conn.executemany(
        "INSERT OR REPLACE INTO prices "
        "(date, open, high, low, close, volume, symbol, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)
