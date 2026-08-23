import math

import pandas as pd
import pytest

from slv.compute import regime


def test_trend_rising_falling_and_flat():
    series = pd.Series([10, 11, 12, 9, 9, 15, 9])
    # lookback=2, comparing each value to the one 2 periods earlier:
    # idx2: 12 vs 10 -> rising
    # idx3: 9 vs 11 -> falling
    # idx4: 9 vs 12 -> falling
    # idx5: 15 vs 9 -> rising
    # idx6: 9 vs 9 -> flat (exact tie)
    labels = regime.trend(series, lookback=2)
    assert list(labels.iloc[:2]) == [None, None]
    assert list(labels.iloc[2:]) == ["rising", "falling", "falling", "rising", "flat"]


def test_percentile_rank():
    series = pd.Series([5, 3, 8, 6, 8])
    # lookback=3:
    # idx2: window [5,3,8], last=8, 3/3 values <= 8 -> 1.0
    # idx3: window [3,8,6], last=6, 2/3 values <= 6 (3 and 6) -> 0.6667
    # idx4: window [8,6,8], last=8, 3/3 values <= 8 -> 1.0
    ranks = regime.percentile_rank(series, lookback=3)
    assert ranks.iloc[:2].isna().all()
    assert ranks.iloc[2] == pytest.approx(1.0)
    assert ranks.iloc[3] == pytest.approx(2 / 3)
    assert ranks.iloc[4] == pytest.approx(1.0)


def test_gsr_percentile_uses_252_day_lookback():
    assert regime.GSR_PERCENTILE_LOOKBACK == 252
    series = pd.Series(range(300))  # strictly increasing -> today is always the max
    ranks = regime.gsr_percentile(series)
    assert ranks.iloc[251] == pytest.approx(1.0)  # first index with a full window
    assert ranks.iloc[:251].isna().all()


def test_cot_percentile_uses_52_report_lookback():
    assert regime.COT_PERCENTILE_LOOKBACK == 52
    series = pd.Series(range(60))
    ranks = regime.cot_percentile(series)
    assert ranks.iloc[51] == pytest.approx(1.0)
    assert ranks.iloc[:51].isna().all()


def test_build_regime_state_packages_fields():
    state = regime.build_regime_state(
        date="2026-08-21",
        real_yield_trend_label="rising",
        dxy_trend_label="falling",
        gsr_pctile=0.75,
        cot_pctile=0.10,
    )
    assert state == regime.RegimeState(
        date="2026-08-21",
        real_yield_trend="rising",
        dxy_trend="falling",
        gsr_percentile=0.75,
        cot_percentile=0.10,
    )


def test_build_regime_state_turns_nan_percentile_into_none():
    state = regime.build_regime_state(
        date="2026-08-21",
        real_yield_trend_label=None,
        dxy_trend_label=None,
        gsr_pctile=math.nan,
        cot_pctile=math.nan,
    )
    assert state.gsr_percentile is None
    assert state.cot_percentile is None
