import numpy as np
import pandas as pd
import pytest

from slv.compute import eventrisk


def test_daily_returns():
    close = pd.Series([100.0, 105.0, 100.0, 130.0])
    # idx1: |105/100 - 1| = 0.05
    # idx2: |100/105 - 1| = 0.047619047619...
    # idx3: |130/100 - 1| = 0.3
    r = eventrisk.daily_returns(close)
    assert r.iloc[0] != r.iloc[0]  # NaN
    assert r.iloc[1] == pytest.approx(0.05)
    assert r.iloc[2] == pytest.approx(5 / 105)
    assert r.iloc[3] == pytest.approx(0.3)


def test_event_day_moves_looks_up_by_date_and_nans_missing():
    dates = ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
    close = pd.Series([100.0, 105.0, 100.0, 130.0], index=dates)
    returns = eventrisk.daily_returns(close)

    # '2026-01-10' isn't in the price history (a future/seeded event) -> NaN
    moves = eventrisk.event_day_moves(returns, ["2026-01-03", "2026-01-05", "2026-01-10"])
    assert moves.loc["2026-01-03"] == pytest.approx(0.05)
    assert moves.loc["2026-01-05"] == pytest.approx(0.3)
    assert np.isnan(moves.loc["2026-01-10"])


def test_estimate_event_risk_median_p90_and_equity_impact():
    # sorted: 0.02, 0.03, 0.04, 0.05, 0.10 -> median (middle of 5) = 0.04
    # p90 (numpy linear interpolation): position = 0.9*(5-1) = 3.6,
    # between sorted[3]=0.05 and sorted[4]=0.10 -> 0.05 + 0.6*0.05 = 0.08
    moves = pd.Series([0.03, 0.10, 0.02, 0.05, 0.04])
    est = eventrisk.estimate_event_risk("CPI", moves, leverage=3)

    assert est.event_name == "CPI"
    assert est.n_observations == 5
    assert est.median_move_pct == pytest.approx(0.04)
    assert est.p90_move_pct == pytest.approx(0.08)
    assert est.median_equity_impact_pct == pytest.approx(0.12)
    assert est.p90_equity_impact_pct == pytest.approx(0.24)


def test_estimate_event_risk_drops_nan_observations():
    moves = pd.Series([0.03, np.nan, 0.02, 0.05, 0.04, np.nan])
    est = eventrisk.estimate_event_risk("NFP", moves, leverage=1)
    assert est.n_observations == 4  # the two NaNs are dropped, not counted


def test_estimate_event_risk_raises_when_nothing_usable():
    moves = pd.Series([np.nan, np.nan])
    with pytest.raises(ValueError, match="no historical moves"):
        eventrisk.estimate_event_risk("FOMC", moves, leverage=3)
