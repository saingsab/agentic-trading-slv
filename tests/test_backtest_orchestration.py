import json

import pytest

from slv import backtest, db


SIMPLE_RULE = {
    "direction": "long",
    "entry": ["close > ema20"],
    "exit": [],
    "stop_atr_mult": 100.0,   # deliberately unreachable -- exercises data_end, not stop/target
    "target_atr_mult": 100.0,
    "max_holding_days": 1000,
}


def _seed(conn, n=60):
    """close rises 1/day; ema20 is always 5 below close, so the entry
    condition fires on the first bar of every slice it's given -- enough
    to exercise fold aggregation and the train/holdout split without
    needing to hand-verify individual R values here (that's test_backtest.py's job).
    """
    for i in range(n):
        date = f"2026-01-{i:03d}"  # not a real calendar date; index order is all that matters
        price = 100.0 + i
        conn.execute(
            "INSERT INTO prices (date, open, high, low, close, volume, symbol, source) "
            "VALUES (?, ?, ?, ?, ?, 1000, 'SI=F', 'test')",
            (date, price, price, price, price),
        )
        conn.execute(
            "INSERT INTO indicators (date, atr14, ema20) VALUES (?, 2.0, ?)",
            (date, price - 5.0),
        )
    conn.commit()


def _use_small_windows(monkeypatch):
    monkeypatch.setattr(backtest, "INITIAL_TRAIN_DAYS", 10)
    monkeypatch.setattr(backtest, "TEST_DAYS", 5)
    monkeypatch.setattr(backtest, "MIN_TRADES", 1)


def test_spec_hash_is_deterministic_and_order_independent():
    a = backtest.spec_hash({"direction": "long", "entry": ["x"]})
    b = backtest.spec_hash({"entry": ["x"], "direction": "long"})
    c = backtest.spec_hash({"direction": "short", "entry": ["x"]})
    assert a == b
    assert a != c


def test_train_holdout_split_is_most_recent_fraction(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        _seed(conn, n=60)
        full = backtest.load_price_indicator_series(conn)
        train, holdout = backtest.train_holdout_split(full, holdout_fraction=0.2)
        assert len(train) == 48
        assert len(holdout) == 12
        assert train.index[-1] < holdout.index[0]  # holdout is strictly the later slice
    finally:
        conn.close()


def test_run_backtest_without_unlock_never_computes_holdout_metrics(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "slv.db")
    try:
        _seed(conn, n=60)
        _use_small_windows(monkeypatch)

        result = backtest.run_backtest(conn, SIMPLE_RULE)
        assert result["holdout_unlocked"] is False
        assert "holdout_metrics" not in result
        assert result["train_metrics"]["reportable"] is True
        assert result["train_metrics"]["n_trades"] > 0

        row = conn.execute(
            "SELECT holdout_unlocked, holdout_unlock_reason, holdout_metrics_json FROM backtest_runs"
        ).fetchone()
        assert row == (0, None, None)
    finally:
        conn.close()


def test_run_backtest_requires_a_reason_to_unlock(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "slv.db")
    try:
        _seed(conn, n=60)
        _use_small_windows(monkeypatch)
        with pytest.raises(ValueError, match="requires a non-empty"):
            backtest.run_backtest(conn, SIMPLE_RULE, unlock_holdout=True)
    finally:
        conn.close()


def test_run_backtest_unlock_computes_and_logs_holdout_metrics(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "slv.db")
    try:
        _seed(conn, n=60)
        _use_small_windows(monkeypatch)

        result = backtest.run_backtest(
            conn, SIMPLE_RULE, unlock_holdout=True, holdout_reason="verifying phase 6"
        )
        assert result["holdout_unlocked"] is True
        assert "holdout_metrics" in result

        row = conn.execute(
            "SELECT holdout_unlocked, holdout_unlock_reason, holdout_metrics_json FROM backtest_runs"
        ).fetchone()
        assert row[0] == 1
        assert row[1] == "verifying phase 6"
        assert json.loads(row[2]) == result["holdout_metrics"]
    finally:
        conn.close()


def test_run_backtest_logs_spec_hash_and_params(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "slv.db")
    try:
        _seed(conn, n=60)
        _use_small_windows(monkeypatch)
        backtest.run_backtest(conn, SIMPLE_RULE)

        row = conn.execute("SELECT spec_hash, params_json FROM backtest_runs").fetchone()
        assert row[0] == backtest.spec_hash(SIMPLE_RULE)
        params = json.loads(row[1])
        assert params["rule"] == SIMPLE_RULE
        assert params["min_trades"] == 1  # picked up the monkeypatched value
    finally:
        conn.close()


def test_run_backtest_raises_with_too_little_train_data(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        _seed(conn, n=20)  # far short of the default 252+90 fold requirement
        with pytest.raises(RuntimeError, match="not enough train data"):
            backtest.run_backtest(conn, SIMPLE_RULE)
    finally:
        conn.close()
