import json

import pandas as pd
import pytest

from slv import brief, cli, db, journal


def _insert_prices(conn, symbol, dates, closes):
    for date, close in zip(dates, closes):
        conn.execute(
            "INSERT INTO prices (date, open, high, low, close, symbol, source) "
            "VALUES (?, ?, ?, ?, ?, ?, 'test')",
            (date, close, close, close, close, symbol),
        )


def _seed_full_dataset(conn, n_days=310):
    """Enough business-day history to clear every lookback window
    (trend=20d, gsr percentile=252d, cot percentile=52 reports), so the
    regime section renders real values instead of "n/a" across the board.
    """
    dates = pd.bdate_range("2025-01-01", periods=n_days).strftime("%Y-%m-%d").tolist()

    silver_close = [60.0 + 0.05 * i for i in range(n_days)]  # steadily rising
    gold_close = [2000.0 + 0.5 * i for i in range(n_days)]
    dxy_close = [100.0 - 0.02 * i for i in range(n_days)]  # steadily falling

    _insert_prices(conn, "SI=F", dates, silver_close)
    _insert_prices(conn, "GC=F", dates, gold_close)
    _insert_prices(conn, "DX-Y.NYB", dates, dxy_close)

    for date, i in zip(dates, range(n_days)):
        conn.execute(
            "INSERT INTO macro (date, series_id, value) VALUES (?, 'DFII10', ?)",
            (date, 1.5 + 0.001 * i),
        )

    # weekly COT reports, every 5th business day -> > 52 reports over 310 days
    for i in range(0, n_days, 5):
        conn.execute(
            "INSERT INTO cot (report_date, mm_net, mm_long, mm_short, open_interest) "
            "VALUES (?, ?, ?, ?, ?)",
            (dates[i], 1000.0 + i, 5000.0, 4000.0, 100000.0),
        )

    # a past CPI (history for event-risk stats) and one CPI inside the
    # 14-day lookahead window from the last price date.
    conn.execute("INSERT INTO events (date, name) VALUES (?, 'CPI')", (dates[100],))
    conn.execute("INSERT INTO events (date, name) VALUES (?, 'CPI')", (dates[150],))
    upcoming_date = (pd.Timestamp(dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    conn.execute("INSERT INTO events (date, name) VALUES (?, 'CPI')", (upcoming_date,))
    conn.commit()

    out = cli._compute_indicators_df(conn)
    rows = [
        (date,) + tuple(cli._sql_value(v) for v in row)
        for date, row in zip(out.index, out.itertuples(index=False))
    ]
    conn.executemany(
        "INSERT INTO indicators (date, " + ", ".join(cli.INDICATORS_COLUMNS) + ") "
        "VALUES (" + ", ".join("?" * (1 + len(cli.INDICATORS_COLUMNS))) + ")",
        rows,
    )
    conn.commit()
    return dates


def test_build_facts_happy_path(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        dates = _seed_full_dataset(conn)
        journal.open_thesis(conn, "long", "70-72", 68, 80, 8, 3)

        facts = brief.build_facts(conn, brief_date="2026-08-23")

        assert facts["as_of"] == dates[-1]
        assert facts["price"]["symbol"] == "SI=F"
        assert facts["price"]["close"] == pytest.approx(60.0 + 0.05 * (len(dates) - 1))

        # rising silver prices over the whole history -> trend is "rising"
        # DXY was constructed steadily falling
        assert facts["regime"]["dxy_trend"] == "falling"
        assert facts["regime"]["real_yield_trend"] == "rising"
        assert 0.0 <= facts["regime"]["gsr_percentile"] <= 1.0
        assert 0.0 <= facts["regime"]["cot_percentile"] <= 1.0

        assert len(facts["upcoming_events"]) == 1
        assert facts["upcoming_events"][0]["name"] == "CPI"
        assert facts["upcoming_events"][0]["n_observations"] == 2

        assert len(facts["open_theses"]) == 1
        assert facts["open_theses"][0]["direction"] == "long"

        # facts_json must actually be valid JSON (NaN would break this)
        json.dumps(facts)
    finally:
        conn.close()


def test_build_facts_raises_without_indicators(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        with pytest.raises(RuntimeError, match="run `slv compute`"):
            brief.build_facts(conn, brief_date="2026-08-23")
    finally:
        conn.close()


def test_render_markdown_handles_missing_values_as_na():
    facts = {
        "date": "2026-08-23",
        "as_of": "2026-08-21",
        "price": {"symbol": "SI=F", "close": None},
        "indicators": {
            "atr14": None, "ema20": None, "ema50": None, "sma200": None,
            "rsi14": None, "rvol20": None, "gsr": None,
            "dist_ema20_atr": None, "range_pctile_60d": None,
        },
        "regime": {
            "real_yield_trend": None, "dxy_trend": None,
            "gsr_percentile": None, "cot_percentile": None,
        },
        "upcoming_events": [],
        "open_theses": [],
    }
    md = brief.render_markdown(facts)
    assert "n/a" in md
    assert "_None scheduled._" in md
    assert "_None open._" in md
    assert "# Silver (XAG/USD) Brief — 2026-08-23" in md


def test_render_markdown_formats_real_values():
    facts = {
        "date": "2026-08-23",
        "as_of": "2026-08-21",
        "price": {"symbol": "SI=F", "close": 69.53},
        "indicators": {
            "atr14": 1.85, "ema20": 63.87, "ema50": 63.98, "sma200": 71.05,
            "rsi14": 67.81, "rvol20": 0.335, "gsr": 67.32,
            "dist_ema20_atr": 3.06, "range_pctile_60d": 0.691,
        },
        "regime": {
            "real_yield_trend": "rising", "dxy_trend": "falling",
            "gsr_percentile": 0.42, "cot_percentile": 0.77,
        },
        "upcoming_events": [
            {"date": "2026-09-04", "name": "NFP", "days_away": 12,
             "median_move_pct": 0.021, "p90_move_pct": 0.045, "n_observations": 55},
        ],
        "open_theses": [
            {"id": 1, "direction": "long", "entry_zone": "68-69", "invalidation": 65.8,
             "target": 74.0, "size_pct_equity": 8, "leverage": 3, "created_at": "2026-08-20T00:00:00"},
        ],
    }
    md = brief.render_markdown(facts)
    assert "69.53" in md
    assert "67.8" in md  # rsi14 to 1 decimal
    assert "**rising**" in md
    assert "**falling**" in md
    assert "42%" in md  # gsr_percentile as a whole-number percent
    assert "NFP" in md and "2.1%" in md and "4.5%" in md
    assert "68-69" in md and "long" in md


def test_write_brief_creates_file_and_db_row(tmp_path, monkeypatch):
    monkeypatch.setattr(brief.config, "BRIEFS_DIR", tmp_path / "briefs")
    monkeypatch.setattr(brief.config, "REPO_ROOT", tmp_path)

    conn = db.connect(tmp_path / "slv.db")
    try:
        _seed_full_dataset(conn)
        path = brief.write_brief(conn, brief_date="2026-08-23")

        assert path.exists()
        assert path.name == "2026-08-23.md"
        assert "Silver (XAG/USD) Brief" in path.read_text()

        row = conn.execute("SELECT date, path, facts_json FROM briefs WHERE date = '2026-08-23'").fetchone()
        assert row is not None
        assert row[1] == "briefs/2026-08-23.md"
        assert json.loads(row[2])["date"] == "2026-08-23"
    finally:
        conn.close()


def test_write_brief_is_idempotent_for_same_date(tmp_path, monkeypatch):
    monkeypatch.setattr(brief.config, "BRIEFS_DIR", tmp_path / "briefs")
    monkeypatch.setattr(brief.config, "REPO_ROOT", tmp_path)

    conn = db.connect(tmp_path / "slv.db")
    try:
        _seed_full_dataset(conn)
        brief.write_brief(conn, brief_date="2026-08-23")
        brief.write_brief(conn, brief_date="2026-08-23")
        count = conn.execute("SELECT COUNT(*) FROM briefs WHERE date = '2026-08-23'").fetchone()[0]
        assert count == 1
    finally:
        conn.close()
