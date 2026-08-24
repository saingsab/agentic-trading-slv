"""I/O orchestration for the backtest engine: loads prices+indicators,
enforces the sealed train/holdout split, runs the walk-forward simulation
via slv.compute.backtest, and logs every run to backtest_runs.

Sealed holdout (PLAN.md: "sealed holdout is enforced at the tool layer...
unlocking it is a manual CLI action by you, logged, and rare"): the
holdout slice is only ever simulated -- metrics computed, logged, or
returned -- when unlock_holdout=True and a non-empty reason is given.
Every unlock is written to backtest_runs itself (holdout_unlocked,
holdout_unlock_reason), so it's auditable after the fact, not just logged
to a console that scrolls away.

None of this is exposed to the Phase 5 agent: slv/agent/tools.py has no
tool that touches backtest_runs, on purpose.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from slv import config
from slv.compute import backtest as bt

# Confirmed with the user 2026-08-24, not in PLAN.md:
HOLDOUT_FRACTION = 0.20  # most recent 20% of history is sealed
INITIAL_TRAIN_DAYS = 252  # ~1 trading year before the first walk-forward fold
TEST_DAYS = 90  # each fold's out-of-sample window
MIN_TRADES = 20  # below this, compute_metrics refuses to report expectancy/R


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def spec_hash(rule: dict) -> str:
    """Deterministic hash of a rule spec, so repeated runs of the same
    idea (even across process restarts) group together in backtest_runs --
    PLAN.md's "multiple-comparison accounting".
    """
    canonical = json.dumps(rule, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load_price_indicator_series(conn: sqlite3.Connection) -> pd.DataFrame:
    """Silver's prices joined with its computed indicators, one row per
    date both tables have. Requires `slv ingest && slv compute` to have
    run first.
    """
    prices = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM prices WHERE symbol = ? ORDER BY date",
        conn,
        params=(config.INSTRUMENT_SYMBOL,),
        index_col="date",
    )
    indicators = pd.read_sql_query("SELECT * FROM indicators ORDER BY date", conn, index_col="date")
    merged = prices.join(indicators, how="inner")
    if merged.empty:
        raise RuntimeError("no merged price/indicator data; run `slv ingest && slv compute` first")
    return merged


def train_holdout_split(
    df: pd.DataFrame, holdout_fraction: float = HOLDOUT_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Most recent `holdout_fraction` of rows are holdout; everything
    before that is train.
    """
    n = len(df)
    holdout_rows = int(n * holdout_fraction)
    split_idx = n - holdout_rows
    return df.iloc[:split_idx], df.iloc[split_idx:]


def run_backtest(
    conn: sqlite3.Connection,
    rule: dict,
    unlock_holdout: bool = False,
    holdout_reason: Optional[str] = None,
) -> dict:
    """Runs the walk-forward backtest against train data, logs the run,
    and returns its metrics.

    Raises if unlock_holdout is True without a reason -- the manual,
    logged, rare unlock ceremony PLAN.md calls for is not optional.
    Holdout metrics are computed only inside that branch; every other
    invocation of this function never simulates the holdout slice at all.
    """
    if unlock_holdout and not holdout_reason:
        raise ValueError("unlock_holdout requires a non-empty holdout_reason")

    full = load_price_indicator_series(conn)
    train, holdout = train_holdout_split(full)

    folds = bt.walk_forward_folds(train, INITIAL_TRAIN_DAYS, TEST_DAYS)
    if not folds:
        raise RuntimeError(
            f"not enough train data for a single walk-forward fold "
            f"(need >= {INITIAL_TRAIN_DAYS + TEST_DAYS} rows, have {len(train)})"
        )

    fold_metrics = []
    all_trades: list[bt.Trade] = []
    for _, test_df in folds:
        fold_trades = bt.simulate_trades(test_df, rule)
        all_trades.extend(fold_trades)
        fold_metrics.append(bt.compute_metrics(fold_trades, MIN_TRADES))

    train_metrics = bt.compute_metrics(all_trades, MIN_TRADES)
    train_metrics["n_folds"] = len(folds)
    train_metrics["fold_metrics"] = fold_metrics

    holdout_metrics = None
    if unlock_holdout:
        holdout_trades = bt.simulate_trades(holdout, rule)
        holdout_metrics = bt.compute_metrics(holdout_trades, MIN_TRADES)

    params = {
        "rule": rule,
        "holdout_fraction": HOLDOUT_FRACTION,
        "initial_train_days": INITIAL_TRAIN_DAYS,
        "test_days": TEST_DAYS,
        "min_trades": MIN_TRADES,
    }

    conn.execute(
        "INSERT INTO backtest_runs "
        "(spec_hash, params_json, train_metrics_json, created_at, "
        " holdout_unlocked, holdout_unlock_reason, holdout_metrics_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            spec_hash(rule),
            json.dumps(params),
            json.dumps(train_metrics),
            _utc_now_iso(),
            1 if unlock_holdout else 0,
            holdout_reason,
            json.dumps(holdout_metrics) if holdout_metrics is not None else None,
        ),
    )
    conn.commit()

    result = {"train_metrics": train_metrics, "holdout_unlocked": unlock_holdout}
    if holdout_metrics is not None:
        result["holdout_metrics"] = holdout_metrics
    return result
