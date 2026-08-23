"""Command-line entry point for slv.

Phase 1 implements `slv ingest` only. Later phases add `compute`, `brief`,
`thesis`, and `journal` subcommands.
"""
from __future__ import annotations

import argparse
import sys

from slv import db
from slv.fetch import calendar, cot, fred, prices


def ingest() -> None:
    """Run all four raw-data fetchers against the shared DB connection.

    Each fetcher commits its own rows before the next one starts, so a
    failure partway through (e.g. FRED down) leaves already-fetched data
    intact and un-duplicated on the next run — no partial writes within any
    single fetcher, and the run as a whole is safe to retry.
    """
    conn = db.connect()
    try:
        n = prices.fetch_prices(conn)
        print(f"prices: {n} rows")

        n = fred.fetch_all(conn)
        print(f"macro:  {n} rows")

        n = cot.fetch_cot(conn)
        print(f"cot:    {n} rows")

        n = calendar.load_calendar(conn)
        print(f"events: {n} rows")
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="slv")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="fetch raw data (idempotent)")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        ingest()

    return 0


if __name__ == "__main__":
    sys.exit(main())
