"""Command-line entry point for slv.

Phase 1 added `slv ingest`. Phase 2 added `slv compute`. Phase 3 added
`slv thesis open/close` and `slv journal score`. Phase 4 adds `slv brief`.
"""
from __future__ import annotations

import argparse
import math
import sys

import pandas as pd

from slv import brief as brief_module
from slv import config, db, journal
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


def _compute_indicators_df(conn) -> pd.DataFrame:
    """Pure-ish assembly of the indicators DataFrame from stored prices --
    split out from compute() so tests can populate a temp DB's indicators
    table the same way `slv compute` does, without going through the CLI.
    """
    silver = _read_prices(conn, config.INSTRUMENT_SYMBOL)
    if silver.empty:
        raise RuntimeError(f"no price data for {config.INSTRUMENT_SYMBOL!r}; run `slv ingest` first")
    gold = _read_prices(conn, config.GOLD_SYMBOL)

    close, high, low = silver["close"], silver["high"], silver["low"]
    atr = indicators.atr14(high, low, close)
    ema20 = indicators.ema20(close)

    return pd.DataFrame(
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


def compute() -> None:
    """Rebuild the indicators table from scratch out of stored prices.

    Derived table (see CLAUDE.md) — always safe to drop and recompute, so
    this deletes all rows and reinserts rather than trying to diff/upsert.
    """
    conn = db.connect()
    try:
        out = _compute_indicators_df(conn)
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


def _cmd_thesis_open(args: argparse.Namespace) -> None:
    conn = db.connect()
    try:
        thesis_id = journal.open_thesis(
            conn,
            direction=args.direction,
            entry_zone=args.entry,
            invalidation=args.invalidation,
            target=args.target,
            size_pct_equity=args.size,
            leverage=args.leverage,
            rationale=args.rationale,
        )
        print(f"thesis {thesis_id} opened: {args.direction} {args.entry}, "
              f"invalidation {args.invalidation}, target {args.target}")
    finally:
        conn.close()


def _cmd_thesis_close(args: argparse.Namespace) -> None:
    conn = db.connect()
    try:
        journal.close_thesis(
            conn, args.id, entry_price=args.entry, exit_price=args.exit, closed_at=args.closed_at
        )
        print(f"thesis {args.id} closed: entry {args.entry}, exit {args.exit}")
    finally:
        conn.close()


def _cmd_journal_score(args: argparse.Namespace) -> None:
    conn = db.connect()
    try:
        results = journal.score_all_unscored(conn)
        if not results:
            print("no unscored closed theses")
        for r in results:
            passed = round(r.score * len(journal.RUBRIC_CHECKS))
            print(
                f"thesis {r.thesis_id}: R={r.r_multiple:+.2f}  "
                f"process_score={r.score:.2f} ({passed}/{len(journal.RUBRIC_CHECKS)})"
            )
    finally:
        conn.close()


def brief() -> None:
    """Assemble today's facts and write a dated markdown brief to briefs/."""
    conn = db.connect()
    try:
        path = brief_module.write_brief(conn)
        print(f"brief written: {path}")
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="slv")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="fetch raw data (idempotent)")
    subparsers.add_parser("compute", help="rebuild derived indicators from stored prices")
    subparsers.add_parser("brief", help="write today's brief to briefs/")

    thesis_parser = subparsers.add_parser("thesis", help="manage theses (immutable once written)")
    thesis_sub = thesis_parser.add_subparsers(dest="thesis_command", required=True)

    open_parser = thesis_sub.add_parser("open", help="log a new thesis before entry")
    open_parser.add_argument("--direction", required=True, choices=["long", "short"])
    open_parser.add_argument(
        "--entry", required=True, metavar="LOW-HIGH", help="planned entry zone, e.g. 72.50-73.20"
    )
    open_parser.add_argument("--invalidation", required=True, type=float)
    open_parser.add_argument("--target", required=True, type=float)
    open_parser.add_argument("--size", required=True, type=float, help="position size, %% of equity")
    open_parser.add_argument("--leverage", required=True, type=float)
    open_parser.add_argument("--rationale", default=None)

    close_parser = thesis_sub.add_parser("close", help="record a closed trade's outcome")
    close_parser.add_argument("id", type=int)
    close_parser.add_argument("--entry", required=True, type=float, help="actual fill price")
    close_parser.add_argument("--exit", required=True, type=float)
    close_parser.add_argument(
        "--closed-at", dest="closed_at", default=None, help="ISO timestamp; defaults to now"
    )

    journal_parser = subparsers.add_parser("journal", help="score closed theses")
    journal_sub = journal_parser.add_subparsers(dest="journal_command", required=True)
    journal_sub.add_parser("score", help="grade every unscored closed thesis on process, not P&L")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        ingest()
    elif args.command == "compute":
        compute()
    elif args.command == "brief":
        brief()
    elif args.command == "thesis":
        if args.thesis_command == "open":
            _cmd_thesis_open(args)
        elif args.thesis_command == "close":
            _cmd_thesis_close(args)
    elif args.command == "journal":
        if args.journal_command == "score":
            _cmd_journal_score(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
