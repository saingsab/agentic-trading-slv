"""Regime classification: real yield trend, DXY trend, GSR percentile, and
COT percentile, packaged into a single RegimeState.

Pure functions only (see CLAUDE.md): no I/O, no DB access.

Locked decisions (confirmed with the user 2026-08-23, not in PLAN.md so
asked rather than assumed):
- Trend = sign of the change over a fixed lookback, no deadband. "flat"
  only fires on an exact tie, which is rare on daily data -- an accepted
  tradeoff for not inventing a deadband threshold.
- Trend lookback: 20 trading days (~1 month), for both real yield and DXY.
- Percentile lookback: trailing 1 year -- 252 trading days for GSR (daily),
  52 reports for COT (weekly).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

TREND_LOOKBACK_DAYS = 20
GSR_PERCENTILE_LOOKBACK = 252
COT_PERCENTILE_LOOKBACK = 52


def trend(series: pd.Series, lookback: int = TREND_LOOKBACK_DAYS) -> pd.Series:
    """'rising' / 'falling' / 'flat' per date, vs. the value `lookback`
    periods earlier. None where there isn't `lookback` periods of history,
    or wherever the comparison is otherwise undefined (a NaN in `series`).
    """
    shifted = series.shift(lookback)
    diff = series - shifted
    labels = pd.Series(
        np.select([diff > 0, diff < 0], ["rising", "falling"], default="flat"),
        index=series.index,
        dtype=object,
    )
    labels[diff.isna()] = None
    return labels.rename("trend")


def percentile_rank(series: pd.Series, lookback: int) -> pd.Series:
    """For each date, the fraction of the trailing `lookback`-window
    (including today) that is <= today's value. 1.0 = today is the highest
    value seen in the window. NaN until `lookback` periods of history exist.
    """

    def _rank_last(window: np.ndarray) -> float:
        return float((window <= window[-1]).sum()) / len(window)

    return series.rolling(lookback).apply(_rank_last, raw=True).rename("percentile_rank")


def gsr_percentile(gsr: pd.Series) -> pd.Series:
    return percentile_rank(gsr, GSR_PERCENTILE_LOOKBACK).rename("gsr_percentile")


def cot_percentile(mm_net: pd.Series) -> pd.Series:
    return percentile_rank(mm_net, COT_PERCENTILE_LOOKBACK).rename("cot_percentile")


@dataclass(frozen=True)
class RegimeState:
    date: str
    real_yield_trend: Optional[str]
    dxy_trend: Optional[str]
    gsr_percentile: Optional[float]
    cot_percentile: Optional[float]


def _clean(value: float) -> Optional[float]:
    """NaN (insufficient lookback history) becomes None, not a fake 0.0."""
    return None if value != value else float(value)


def build_regime_state(
    date: str,
    real_yield_trend_label: Optional[str],
    dxy_trend_label: Optional[str],
    gsr_pctile: float,
    cot_pctile: float,
) -> RegimeState:
    """Package one date's already-computed classifier values into a
    RegimeState. Callers (cli.py's `compute`) pull "today" out of each
    Series before calling this -- keeping this function trivial and pure.
    """
    return RegimeState(
        date=date,
        real_yield_trend=real_yield_trend_label,
        dxy_trend=dxy_trend_label,
        gsr_percentile=_clean(gsr_pctile),
        cot_percentile=_clean(cot_pctile),
    )
