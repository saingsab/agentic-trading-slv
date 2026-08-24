"""Rule-based backtest simulation. Pure functions only (see CLAUDE.md): no
I/O, no DB access -- everything here takes a DataFrame of prices+indicators
already joined by date and returns plain Python/dataclass results.

Rule spec (confirmed with the user 2026-08-24, not in PLAN.md):
{
    "direction": "long" | "short",
    "entry": ["close > ema20", "rsi14 < 40"],   # ALL must hold (AND)
    "exit": ["close < ema50"],                   # ANY holds (OR); optional
    "stop_atr_mult": 1.5,                        # invalidation = entry -/+ mult * ATR14 at signal
    "target_atr_mult": 3.0,
    "max_holding_days": 20,
}
Conditions are "<column> <op> <value>" where value is a float literal or
another column name -- deliberately not eval(): a small closed vocabulary
of comparisons only, kept safe and exactly hashable via json.dumps.

Timing (avoids lookahead bias): a signal computed from bar i's data fills
at bar i+1's open -- you can only act on a bar's indicators after that bar
has closed. Once in a position, each subsequent bar's high/low is checked
for an intraday stop/target hit before that bar's exit-condition signal
is even considered (stop takes priority over target if both would fire
the same bar -- the conservative assumption). A position can't span past
the end of the DataFrame it's given: it's simulated fold-by-fold (see
walk_forward_folds), and a trade still open at the last bar force-closes
there rather than reading past it.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional

import pandas as pd

_OPS = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
}


def evaluate_condition(condition: str, row: pd.Series) -> bool:
    """'<column> <op> <value>' against one row. value is a column name if
    it matches one, else parsed as a float. NaN on either side -> False
    (a condition can't be satisfied by missing data, e.g. before SMA200
    has 200 days of history).
    """
    parts = condition.split()
    if len(parts) != 3:
        raise ValueError(f"condition must be '<column> <op> <value>', got {condition!r}")
    col, op, value_token = parts

    if col not in row.index:
        raise ValueError(f"unknown column {col!r} in condition {condition!r}")
    if op not in _OPS:
        raise ValueError(f"unsupported operator {op!r} in condition {condition!r}")

    left = row[col]
    if value_token in row.index:
        right = row[value_token]
    else:
        try:
            right = float(value_token)
        except ValueError:
            raise ValueError(f"value must be a column name or a number, got {value_token!r}") from None

    if left != left or right != right:  # NaN check without importing numpy/math here
        return False
    # bool(...): pandas/numpy scalar comparisons return np.bool_, not a
    # plain Python bool -- which e.g. json.dumps() rejects outright.
    return bool(_OPS[op](left, right))


def _all_hold(conditions: list[str], row: pd.Series) -> bool:
    return all([evaluate_condition(c, row) for c in conditions])


def _any_holds(conditions: list[str], row: pd.Series) -> bool:
    if not conditions:
        return False
    return any([evaluate_condition(c, row) for c in conditions])


@dataclass(frozen=True)
class Trade:
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    direction: str
    r_multiple: float
    exit_reason: str  # "stop" | "target" | "rule_exit" | "max_holding" | "data_end"


def simulate_trades(df: pd.DataFrame, rule: dict) -> list[Trade]:
    """Walk `df` (ascending date index; needs open/high/low/close plus
    whatever columns the rule's conditions reference) bar by bar, opening
    and closing at most one position at a time per the timing rules in
    this module's docstring.
    """
    direction = rule["direction"]
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
    entry_conds = rule["entry"]
    exit_conds = rule.get("exit", [])
    stop_mult = rule["stop_atr_mult"]
    target_mult = rule["target_atr_mult"]
    max_holding = rule["max_holding_days"]

    dates = df.index.tolist()
    n = len(df)
    trades: list[Trade] = []

    i = 0
    while i < n - 1:  # need a next bar to fill on
        if not _all_hold(entry_conds, df.iloc[i]):
            i += 1
            continue

        atr_at_signal = df.iloc[i]["atr14"]
        if atr_at_signal != atr_at_signal:  # NaN ATR -- not enough history to size this trade
            i += 1
            continue

        entry_idx = i + 1
        entry_price = df.iloc[entry_idx]["open"]
        if direction == "long":
            invalidation = entry_price - stop_mult * atr_at_signal
            target = entry_price + target_mult * atr_at_signal
        else:
            invalidation = entry_price + stop_mult * atr_at_signal
            target = entry_price - target_mult * atr_at_signal

        exit_idx, exit_price, exit_reason = _find_exit(
            df, entry_idx, direction, invalidation, target, exit_conds, max_holding
        )

        risk = abs(entry_price - invalidation)
        reward = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        r_multiple = reward / risk if risk > 0 else 0.0

        trades.append(
            Trade(
                entry_date=dates[entry_idx],
                entry_price=entry_price,
                exit_date=dates[exit_idx],
                exit_price=exit_price,
                direction=direction,
                r_multiple=r_multiple,
                exit_reason=exit_reason,
            )
        )
        i = exit_idx + 1  # positions don't overlap

    return trades


def _find_exit(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    invalidation: float,
    target: float,
    exit_conds: list[str],
    max_holding: int,
) -> tuple[int, float, str]:
    n = len(df)
    j = entry_idx
    while j < n:
        bar = df.iloc[j]
        if direction == "long":
            stop_hit, target_hit = bar["low"] <= invalidation, bar["high"] >= target
        else:
            stop_hit, target_hit = bar["high"] >= invalidation, bar["low"] <= target

        if stop_hit:
            return j, invalidation, "stop"
        if target_hit:
            return j, target, "target"
        if j > entry_idx and _any_holds(exit_conds, bar):
            if j + 1 < n:
                return j + 1, df.iloc[j + 1]["open"], "rule_exit"
            return j, bar["close"], "rule_exit"  # signalled on the last bar -- can't lag further
        if (j - entry_idx) >= max_holding:
            return j, bar["close"], "max_holding"
        j += 1
    return n - 1, df.iloc[-1]["close"], "data_end"  # ran off the end of this slice


def walk_forward_folds(
    df: pd.DataFrame, initial_train_days: int = 252, test_days: int = 90
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window walk-forward folds: fold k's train is every row up
    to that fold's test window; the test window itself is the next
    `test_days` rows. Training data only grows fold over fold, matching
    that you'd never "forget" earlier history live. A trailing remainder
    shorter than test_days is dropped rather than reported as a partial
    fold.
    """
    n = len(df)
    folds = []
    train_end = initial_train_days
    while train_end + test_days <= n:
        folds.append((df.iloc[:train_end], df.iloc[train_end : train_end + test_days]))
        train_end += test_days
    return folds


def compute_metrics(trades: list[Trade], min_trades: int) -> dict:
    """Expectancy (mean R) and the R-multiple distribution -- never a win
    rate (CLAUDE.md). Refuses to report anything but the trade count if
    there aren't at least `min_trades`, rather than showing a number
    precise-looking enough to be mistaken for meaningful.
    """
    n = len(trades)
    if n < min_trades:
        return {"n_trades": n, "min_trades": min_trades, "reportable": False}

    r_values = [t.r_multiple for t in trades]
    return {
        "n_trades": n,
        "min_trades": min_trades,
        "reportable": True,
        "expectancy_r": statistics.mean(r_values),
        "r_median": statistics.median(r_values),
        "r_stdev": statistics.stdev(r_values) if n > 1 else 0.0,
        "r_min": min(r_values),
        "r_max": max(r_values),
    }
