"""Event risk sizing: given historical event-day moves and a leverage
level, what's the median/p90 move and what does that do to equity.

Pure functions only (see CLAUDE.md): no I/O, no DB access. "The move" for
a given day is close-to-close return magnitude (confirmed with the user
2026-08-23, over true-range-as-%-of-prior-close). Equity impact is
move_pct * leverage, matching CLAUDE.md's framing directly: "Silver's
daily ATR runs ~2%... At 3x leverage that is 6-15% of equity."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def daily_returns(close: pd.Series) -> pd.Series:
    """abs((close / prior_close) - 1) for every date. First value is NaN
    (no prior close).
    """
    return ((close / close.shift(1)) - 1).abs().rename("daily_return_abs")


def event_day_moves(returns: pd.Series, event_dates: list[str]) -> pd.Series:
    """Look up `returns` on each date in `event_dates`.

    NaN wherever an event date has no corresponding return -- most often
    because it's a seeded future event that hasn't happened yet, or falls
    on a non-trading day. Callers must drop NaNs before treating the
    result as a historical sample (estimate_event_risk does this).
    """
    return returns.reindex(event_dates).rename("event_day_move")


@dataclass(frozen=True)
class EventRiskEstimate:
    event_name: str
    n_observations: int
    leverage: float
    median_move_pct: float
    p90_move_pct: float
    median_equity_impact_pct: float
    p90_equity_impact_pct: float


def estimate_event_risk(
    event_name: str, historical_moves_pct: pd.Series, leverage: float
) -> EventRiskEstimate:
    """Median and 90th-percentile historical move for this event type, and
    what each does to equity at `leverage`.

    `historical_moves_pct` is a sample of abs daily-return fractions from
    past instances of this event (e.g. from event_day_moves); NaNs (future
    events with no price data yet) are dropped before computing statistics.
    Raises if nothing usable is left -- never reports a made-up number.
    """
    moves = np.asarray(historical_moves_pct, dtype=float)
    moves = moves[~np.isnan(moves)]
    if len(moves) == 0:
        raise ValueError(f"no historical moves available for event {event_name!r}")

    median_move = float(np.median(moves))
    p90_move = float(np.percentile(moves, 90))
    return EventRiskEstimate(
        event_name=event_name,
        n_observations=len(moves),
        leverage=leverage,
        median_move_pct=median_move,
        p90_move_pct=p90_move,
        median_equity_impact_pct=median_move * leverage,
        p90_equity_impact_pct=p90_move * leverage,
    )
