"""Deterministic technical indicators computed from stored OHLCV.

Pure functions only: no I/O, no DB access, no network (see CLAUDE.md). Every
function takes pandas Series aligned on a date index and returns a Series of
the same length, NaN wherever there isn't enough history yet.

ATR14 and RSI14 use Wilder's smoothing (the original/industry-standard
definition of both indicators, confirmed with the user 2026-08-23): a
14-period simple-average seed, then each subsequent value is
(prev * (period-1) + new) / period. rvol20 is annualized (x sqrt(252)),
also confirmed with the user.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing recurrence, shared by atr14 and rsi14.

    `values[0]` is assumed undefined (e.g. no previous close to diff
    against) and is never read. The seed at index `period` is the simple
    average of values[1..period]; every index after that is
    (prev * (period - 1) + values[i]) / period. Everything before index
    `period` is NaN.
    """
    n = len(values)
    out = np.full(n, np.nan)
    if n <= period:
        return out

    prev = np.mean(values[1 : period + 1])
    out[period] = prev
    for i in range(period + 1, n):
        prev = (prev * (period - 1) + values[i]) / period
        out[i] = prev
    return out


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Greatest of today's high-low range and the gap from yesterday's close.

    First value is NaN (no previous close to compare against). Forced
    explicitly: pandas' default max(axis=1, skipna=True) would otherwise
    quietly fall back to high-low for that row instead of leaving it
    undefined.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    tr.iloc[0] = np.nan
    return tr.rename("true_range")


def atr14(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return pd.Series(_wilder_smooth(tr.to_numpy(), period), index=close.index, name="atr14")


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average, seeded with the simple average of the
    first `span` closes (the standard convention), smoothing factor
    k = 2 / (span + 1) thereafter.
    """
    values = close.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    if n < span:
        return pd.Series(out, index=close.index, name=f"ema{span}")

    k = 2 / (span + 1)
    prev = np.mean(values[:span])
    out[span - 1] = prev
    for i in range(span, n):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return pd.Series(out, index=close.index, name=f"ema{span}")


def ema20(close: pd.Series) -> pd.Series:
    return ema(close, 20).rename("ema20")


def ema50(close: pd.Series) -> pd.Series:
    return ema(close, 50).rename("ema50")


def sma200(close: pd.Series, period: int = 200) -> pd.Series:
    return close.rolling(period).mean().rename("sma200")


def rsi14(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. 100 where losses are zero over the smoothed window and
    there's been any gain; NaN where both average gain and average loss are
    zero (a flat price series — 0/0 is undefined, not "50").
    """
    diff = close.diff().to_numpy()
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    gain[0] = np.nan
    loss[0] = np.nan

    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
    rsi = np.where((avg_gain == 0) & (avg_loss == 0), np.nan, rsi)
    return pd.Series(rsi, index=close.index, name="rsi14")


def rvol20(close: pd.Series, period: int = 20) -> pd.Series:
    """Annualized realized volatility: sample stdev of daily log returns
    over `period` days, x sqrt(252).
    """
    log_returns = np.log(close / close.shift(1))
    return (log_returns.rolling(period).std() * np.sqrt(TRADING_DAYS_PER_YEAR)).rename("rvol20")


def gsr(gold_close: pd.Series, silver_close: pd.Series) -> pd.Series:
    """Gold-silver ratio: gold_close / silver_close."""
    return (gold_close / silver_close).rename("gsr")


def dist_ema20_atr(close: pd.Series, ema20_series: pd.Series, atr14_series: pd.Series) -> pd.Series:
    """Distance of close from EMA20, expressed in ATR14 units."""
    return ((close - ema20_series) / atr14_series).rename("dist_ema20_atr")


def range_pctile_60d(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 60) -> pd.Series:
    """Where today's close sits within the trailing `period`-day high-low
    range, as a 0-1 fraction (0 = at the period low, 1 = at the period high).
    """
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return ((close - lowest) / (highest - lowest)).rename("range_pctile_60d")
