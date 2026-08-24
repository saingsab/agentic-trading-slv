import math

import pandas as pd
import pytest

from slv.compute import backtest as bt


# ---------------------------------------------------------------------------
# evaluate_condition
# ---------------------------------------------------------------------------


def test_evaluate_condition_column_vs_constant():
    row = pd.Series({"close": 70.0, "ema20": 65.0, "rsi14": 35.0})
    assert bt.evaluate_condition("rsi14 < 40", row) is True
    assert bt.evaluate_condition("rsi14 < 30", row) is False
    assert bt.evaluate_condition("close == 70", row) is True


def test_evaluate_condition_column_vs_column():
    row = pd.Series({"close": 70.0, "ema20": 65.0})
    assert bt.evaluate_condition("close > ema20", row) is True
    assert bt.evaluate_condition("close < ema20", row) is False


def test_evaluate_condition_nan_is_never_true():
    row = pd.Series({"close": float("nan"), "ema20": 65.0})
    assert bt.evaluate_condition("close > ema20", row) is False
    assert bt.evaluate_condition("close < ema20", row) is False


def test_evaluate_condition_rejects_unknown_column():
    row = pd.Series({"close": 70.0})
    with pytest.raises(ValueError, match="unknown column"):
        bt.evaluate_condition("nonexistent > 1", row)


def test_evaluate_condition_rejects_bad_operator():
    row = pd.Series({"close": 70.0})
    with pytest.raises(ValueError, match="unsupported operator"):
        bt.evaluate_condition("close ~= 1", row)


def test_evaluate_condition_rejects_bad_value():
    row = pd.Series({"close": 70.0})
    with pytest.raises(ValueError, match="column name or a number"):
        bt.evaluate_condition("close > banana", row)


# ---------------------------------------------------------------------------
# simulate_trades -- one hand-verified scenario per exit path
# ---------------------------------------------------------------------------


def _df(rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("date")


def test_simulate_trades_target_hit():
    # entry signal on d0 (close 100 > ema20 95) -> fill at d1's open (101)
    # invalidation = 101 - 1.0*atr(2.0) = 99; target = 101 + 2.0*2.0 = 105
    # d1..d2 touch neither; d3's high (106) clears the target -> exit at 105
    df = _df({
        "date":  ["d0", "d1", "d2", "d3", "d4"],
        "open":  [99.0, 101.0, 101.5, 104.0, 105.0],
        "high":  [100.0, 102.0, 103.0, 106.0, 107.0],
        "low":   [98.0, 100.0, 99.5, 100.0, 104.0],
        "close": [100.0, 101.0, 102.0, 105.0, 106.0],
        "atr14": [2.0, 2.1, 2.0, 2.0, 2.0],
        "ema20": [95.0, 96.0, 96.5, 97.0, 98.0],
    })
    rule = {"direction": "long", "entry": ["close > ema20"], "exit": [],
            "stop_atr_mult": 1.0, "target_atr_mult": 2.0, "max_holding_days": 10}

    trades = bt.simulate_trades(df, rule)
    assert len(trades) == 1
    t = trades[0]
    assert (t.entry_date, t.entry_price) == ("d1", 101.0)
    assert (t.exit_date, t.exit_price, t.exit_reason) == ("d3", 105.0, "target")
    assert t.r_multiple == pytest.approx(2.0)  # (105-101)/(101-99)


def test_simulate_trades_stop_hit_on_fill_bar():
    # invalidation = 101 - 1.0*2.0 = 99; d1's low (98.5) hits it immediately
    df = _df({
        "date":  ["d0", "d1", "d2"],
        "open":  [99.0, 101.0, 99.0],
        "high":  [100.0, 101.5, 100.0],
        "low":   [98.0, 98.5, 98.0],
        "close": [100.0, 99.0, 99.0],
        "atr14": [2.0, 2.0, 2.0],
        "ema20": [95.0, 96.0, 96.0],
    })
    rule = {"direction": "long", "entry": ["close > ema20"], "exit": [],
            "stop_atr_mult": 1.0, "target_atr_mult": 5.0, "max_holding_days": 10}

    trades = bt.simulate_trades(df, rule)
    assert len(trades) == 1
    t = trades[0]
    assert (t.entry_date, t.exit_date) == ("d1", "d1")  # stopped out same bar as fill
    assert t.exit_reason == "stop"
    assert t.r_multiple == pytest.approx(-1.0)  # hitting your stop is always R=-1


def test_simulate_trades_max_holding():
    # stop/target set unreachably far; forced out after 2 bars held
    df = _df({
        "date":  ["d0", "d1", "d2", "d3"],
        "open":  [99.0, 101.0, 101.0, 102.0],
        "high":  [100.0, 102.0, 102.0, 103.0],
        "low":   [98.0, 100.0, 100.0, 101.0],
        "close": [100.0, 101.0, 102.0, 103.0],
        "atr14": [1.0, 1.0, 1.0, 1.0],
        "ema20": [95.0, 96.0, 96.0, 97.0],
    })
    rule = {"direction": "long", "entry": ["close > ema20"], "exit": [],
            "stop_atr_mult": 100.0, "target_atr_mult": 100.0, "max_holding_days": 2}

    trades = bt.simulate_trades(df, rule)
    assert len(trades) == 1
    t = trades[0]
    assert (t.entry_date, t.exit_date, t.exit_reason) == ("d1", "d3", "max_holding")
    # risk = |101 - 1| = 100 (invalidation = 101 - 100*1.0); reward = 103-101 = 2
    assert t.r_multiple == pytest.approx(2 / 100)


def test_simulate_trades_rule_exit():
    # stop/target unreachably far; exit condition (close < ema20) fires on
    # d2, filling at d3's open
    df = _df({
        "date":  ["d0", "d1", "d2", "d3"],
        "open":  [99.0, 101.0, 102.0, 93.0],
        "high":  [100.0, 103.0, 103.0, 94.0],
        "low":   [98.0, 100.0, 93.5, 92.0],
        "close": [100.0, 102.0, 94.0, 93.0],
        "atr14": [1.0, 1.0, 1.0, 1.0],
        "ema20": [95.0, 96.0, 96.0, 95.0],
    })
    rule = {"direction": "long", "entry": ["close > ema20"], "exit": ["close < ema20"],
            "stop_atr_mult": 100.0, "target_atr_mult": 100.0, "max_holding_days": 30}

    trades = bt.simulate_trades(df, rule)
    assert len(trades) == 1
    t = trades[0]
    assert (t.entry_date, t.exit_date, t.exit_reason) == ("d1", "d3", "rule_exit")
    assert (t.entry_price, t.exit_price) == (101.0, 93.0)
    assert t.r_multiple == pytest.approx(-8 / 100)  # reward=-8, risk=|101-1|=100


def test_simulate_trades_skips_signal_with_nan_atr():
    df = _df({
        "date":  ["d0", "d1"],
        "open":  [99.0, 89.0],
        "high":  [100.0, 91.0],
        "low":   [98.0, 88.0],
        "close": [100.0, 90.0],
        "atr14": [float("nan"), 1.0],
        "ema20": [95.0, 95.0],
    })
    rule = {"direction": "long", "entry": ["close > ema20"], "exit": [],
            "stop_atr_mult": 1.0, "target_atr_mult": 2.0, "max_holding_days": 10}

    assert bt.simulate_trades(df, rule) == []


def test_simulate_trades_short_direction():
    # entry: close < ema20 on d0 (90 < 95) -> fill at d1's open (89)
    # invalidation = 89 + 1.0*2.0 = 91; target = 89 - 2.0*2.0 = 85
    # d1's high (92) clears the (short) stop -> loss
    df = _df({
        "date":  ["d0", "d1"],
        "open":  [91.0, 89.0],
        "high":  [92.0, 92.0],
        "low":   [88.0, 88.0],
        "close": [90.0, 90.0],
        "atr14": [2.0, 2.0],
        "ema20": [95.0, 95.0],
    })
    rule = {"direction": "short", "entry": ["close < ema20"], "exit": [],
            "stop_atr_mult": 1.0, "target_atr_mult": 2.0, "max_holding_days": 10}

    trades = bt.simulate_trades(df, rule)
    assert len(trades) == 1
    t = trades[0]
    assert (t.exit_price, t.exit_reason) == (91.0, "stop")
    assert t.r_multiple == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# walk_forward_folds
# ---------------------------------------------------------------------------


def test_walk_forward_folds_boundaries():
    df = pd.DataFrame({"close": range(30)})
    folds = bt.walk_forward_folds(df, initial_train_days=10, test_days=5)
    # train_end steps 10, 15, 20, 25 (25+5=30 fits); 30+5=35 > 30 stops -> 4 folds
    assert len(folds) == 4

    train0, test0 = folds[0]
    assert len(train0) == 10 and list(train0.index) == list(range(0, 10))
    assert len(test0) == 5 and list(test0.index) == list(range(10, 15))

    train3, test3 = folds[3]
    assert len(train3) == 25 and list(train3.index) == list(range(0, 25))
    assert len(test3) == 5 and list(test3.index) == list(range(25, 30))


def test_walk_forward_folds_drops_short_trailing_remainder():
    df = pd.DataFrame({"close": range(32)})  # 2 rows past the last full fold
    folds = bt.walk_forward_folds(df, initial_train_days=10, test_days=5)
    assert len(folds) == 4  # the trailing 2 rows (30, 31) don't form a fold


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_reportable():
    trades = [
        bt.Trade("d", 1, "d", 1, "long", r, "target") for r in [1.0, -1.0, 2.0, 0.5, -0.5]
    ]
    stats = bt.compute_metrics(trades, min_trades=3)
    assert stats["reportable"] is True
    assert stats["n_trades"] == 5
    assert stats["expectancy_r"] == pytest.approx(0.4)  # (1-1+2+0.5-0.5)/5
    assert stats["r_median"] == pytest.approx(0.5)
    assert stats["r_stdev"] == pytest.approx(math.sqrt(1.425))  # sample variance, hand-derived
    assert stats["r_min"] == pytest.approx(-1.0)
    assert stats["r_max"] == pytest.approx(2.0)
    assert "win_rate" not in stats  # CLAUDE.md: never report win rate


def test_compute_metrics_insufficient_trades_refuses_to_report():
    trades = [bt.Trade("d", 1, "d", 1, "long", 1.0, "target") for _ in range(2)]
    stats = bt.compute_metrics(trades, min_trades=20)
    assert stats == {"n_trades": 2, "min_trades": 20, "reportable": False}
