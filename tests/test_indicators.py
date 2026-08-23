"""Every expected value here is derived independently of indicators.py —
either by hand (documented inline) or, for rvol20, with plain arithmetic
that doesn't call the function under test — never by running the code and
snapshotting its output.
"""
import math

import numpy as np
import pandas as pd
import pytest

from slv.compute import indicators as ind


def test_true_range():
    high = pd.Series([10, 12, 11, 13])
    low = pd.Series([8, 9, 9, 10])
    close = pd.Series([9, 11, 10, 12])
    # row0: no prev close -> NaN
    # row1: max(12-9, |12-9|, |9-9|) = max(3,3,0) = 3
    # row2: max(11-9, |11-11|, |9-11|) = max(2,0,2) = 2
    # row3: max(13-10, |13-10|, |10-10|) = max(3,3,0) = 3
    tr = ind.true_range(high, low, close)
    assert tr.iloc[0] != tr.iloc[0]  # NaN
    assert list(tr.iloc[1:]) == [3, 2, 3]


def test_atr14_with_small_period():
    # close held flat at 10 so |high-prev_close| and |low-prev_close|
    # reduce to a plain deviation from 10; makes the true range obvious.
    high = pd.Series([11, 13, 12, 14, 15])
    low = pd.Series([9, 8, 9, 8, 7])
    close = pd.Series([10, 10, 10, 10, 10])
    # TR = [NaN, 5, 3, 6, 8] (row1: max(5,3,2)=5; row2: max(3,2,1)=3;
    # row3: max(6,4,2)=6; row4: max(8,5,3)=8)
    # Wilder period=3: seed = mean(TR[1..3]) = mean(5,3,6) = 14/3
    # next = (14/3 * 2 + 8) / 3 = (28/3 + 8) / 3 = (52/3) / 3 = 52/9
    atr = ind.atr14(high, low, close, period=3)
    assert atr.iloc[:3].isna().all()
    assert atr.iloc[3] == pytest.approx(14 / 3)
    assert atr.iloc[4] == pytest.approx(52 / 9)


def test_ema_seeds_with_simple_average_then_smooths():
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    # span=3 -> k=0.5; seed = mean(10,11,12) = 11 at index 2
    # index3 = 13*0.5 + 11*0.5 = 12
    # index4 = 14*0.5 + 12*0.5 = 13
    # index5 = 15*0.5 + 13*0.5 = 14
    e = ind.ema(close, span=3)
    assert e.iloc[:2].isna().all()
    assert list(e.iloc[2:]) == [11, 12, 13, 14]


def test_sma200_with_small_period():
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    s = ind.sma200(close, period=3)
    # rolling mean of a linear sequence = the window's midpoint
    assert s.iloc[:2].isna().all()
    assert list(s.iloc[2:]) == [11, 12, 13, 14]


def test_rsi14_with_small_period():
    close = pd.Series([10, 11, 12, 11, 10, 11, 12, 13])
    # diff = [NaN,1,1,-1,-1,1,1,1]; gain=[NaN,1,1,0,0,1,1,1]; loss=[NaN,0,0,1,1,0,0,0]
    # Wilder period=3:
    # avg_gain seed (idx3) = mean(1,1,0) = 2/3; avg_loss seed = mean(0,0,1) = 1/3
    #   RS = 2 -> RSI = 100 - 100/3 = 66.666...
    # idx4: avg_gain=(2/3*2+0)/3=4/9; avg_loss=(1/3*2+1)/3=5/9; RS=0.8 -> RSI=44.444...
    # idx5: avg_gain=(4/9*2+1)/3=17/27; avg_loss=(5/9*2+0)/3=10/27; RS=1.7 -> RSI=62.963...
    # idx6: avg_gain=(17/27*2+1)/3=61/81; avg_loss=(10/27*2+0)/3=20/81; RS=3.05 -> RSI=75.309...
    # idx7: avg_gain=(61/81*2+1)/3=203/243; avg_loss=(20/81*2+0)/3=40/243; RS=5.075 -> RSI=83.539...
    rsi = ind.rsi14(close, period=3)
    assert rsi.iloc[:3].isna().all()
    expected = [
        100 - 100 / 3,
        100 - 100 / 1.8,
        100 - 100 / 2.7,
        100 - 100 / 4.05,
        100 - 100 / 6.075,
    ]
    for actual, exp in zip(rsi.iloc[3:], expected):
        assert actual == pytest.approx(exp)


def test_rsi14_is_nan_on_flat_price_not_fifty():
    close = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
    rsi = ind.rsi14(close, period=3)
    # 0/0 (no gains, no losses) is undefined -- must not silently read as 50.
    assert rsi.iloc[3:].isna().all()


def test_rvol20_annualized_sample_stdev():
    close = pd.Series([100.0, 110.0, 100.0, 121.0])
    # log returns: ln(1.1), ln(1/1.1), ln(1.21) = 0.09531018, -0.09531018, 0.19062036
    # sample stdev (ddof=1) of a rolling 2-window, then x sqrt(252) -- computed
    # independently with plain mean/variance below, not via the function under test.
    r = [math.log(110 / 100), math.log(100 / 110), math.log(121 / 100)]

    def sample_std(vals):
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        return var**0.5

    expected_idx2 = sample_std(r[0:2]) * math.sqrt(252)
    expected_idx3 = sample_std(r[1:3]) * math.sqrt(252)

    rv = ind.rvol20(close, period=2)
    assert rv.iloc[:2].isna().all()
    assert rv.iloc[2] == pytest.approx(expected_idx2)
    assert rv.iloc[3] == pytest.approx(expected_idx3)


def test_gsr_is_gold_over_silver():
    gold = pd.Series([2000.0, 2000.0])
    silver = pd.Series([25.0, 20.0])
    g = ind.gsr(gold, silver)
    assert list(g) == [80.0, 100.0]


def test_dist_ema20_atr():
    close = pd.Series([110.0, 95.0])
    ema20 = pd.Series([100.0, 100.0])
    atr14 = pd.Series([5.0, 2.5])
    d = ind.dist_ema20_atr(close, ema20, atr14)
    assert list(d) == [2.0, -2.0]


def test_range_pctile_60d_with_small_period():
    high = pd.Series([12, 14, 13])
    low = pd.Series([8, 9, 9])
    close = pd.Series([10, 13, 9])
    # period=2: idx1 window highs=[12,14]->14, lows=[8,9]->8; (13-8)/(14-8)=5/6
    # idx2 window highs=[14,13]->14, lows=[9,9]->9; (9-9)/(14-9)=0
    p = ind.range_pctile_60d(high, low, close, period=2)
    assert np.isnan(p.iloc[0])
    assert p.iloc[1] == pytest.approx(5 / 6)
    assert p.iloc[2] == pytest.approx(0.0)
