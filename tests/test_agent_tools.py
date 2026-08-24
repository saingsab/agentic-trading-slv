"""Tests the tool functions directly, in-process, no Docker/Ollama needed.

Points SLV_DB_PATH at a temp DB -- see tools.py's module docstring for why
that's the only way to inject a path into something that has to stay
self-contained enough to ship into a sandbox.
"""
import json

import pytest

from slv import db, journal
from slv.agent.tools import get_indicators, get_regime, search_journal


def test_get_indicators_returns_latest_row(tmp_path, monkeypatch):
    db_path = tmp_path / "slv.db"
    conn = db.connect(db_path)
    conn.execute(
        "INSERT INTO indicators (date, atr14, ema20, ema50, sma200, rsi14, rvol20, "
        "gsr, dist_ema20_atr, range_pctile_60d) VALUES "
        "('2026-08-20', 1.5, 60.0, 59.0, 58.0, 55.0, 0.3, 70.0, 1.0, 0.5)"
    )
    conn.execute(
        "INSERT INTO indicators (date, atr14, ema20, ema50, sma200, rsi14, rvol20, "
        "gsr, dist_ema20_atr, range_pctile_60d) VALUES "
        "('2026-08-21', 1.85, 63.87, 63.98, 71.05, 67.81, 0.335, 67.32, 3.06, 0.691)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("SLV_DB_PATH", str(db_path))

    result = json.loads(get_indicators(""))
    assert result["date"] == "2026-08-21"
    assert result["rsi14"] == 67.81


def test_get_indicators_by_specific_date(tmp_path, monkeypatch):
    db_path = tmp_path / "slv.db"
    conn = db.connect(db_path)
    conn.execute(
        "INSERT INTO indicators (date, atr14) VALUES ('2026-08-20', 1.5), ('2026-08-21', 1.85)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("SLV_DB_PATH", str(db_path))

    result = json.loads(get_indicators("2026-08-20"))
    assert result["atr14"] == 1.5


def test_get_indicators_missing_date_returns_error(tmp_path, monkeypatch):
    db_path = tmp_path / "slv.db"
    db.connect(db_path).close()
    monkeypatch.setenv("SLV_DB_PATH", str(db_path))

    result = json.loads(get_indicators("2099-01-01"))
    assert "error" in result


def test_get_regime_reads_latest_brief(tmp_path, monkeypatch):
    db_path = tmp_path / "slv.db"
    conn = db.connect(db_path)
    facts_old = {"as_of": "2026-08-20", "regime": {"real_yield_trend": "falling", "dxy_trend": "falling", "gsr_percentile": 0.1, "cot_percentile": 0.2}}
    facts_new = {"as_of": "2026-08-21", "regime": {"real_yield_trend": "rising", "dxy_trend": "falling", "gsr_percentile": 0.54, "cot_percentile": 0.23}}
    conn.execute("INSERT INTO briefs (date, path, facts_json) VALUES ('2026-08-20', 'x', ?)", (json.dumps(facts_old),))
    conn.execute("INSERT INTO briefs (date, path, facts_json) VALUES ('2026-08-21', 'x', ?)", (json.dumps(facts_new),))
    conn.commit()
    conn.close()
    monkeypatch.setenv("SLV_DB_PATH", str(db_path))

    result = json.loads(get_regime())
    assert result["as_of"] == "2026-08-21"
    assert result["real_yield_trend"] == "rising"
    assert result["gsr_percentile"] == 0.54


def test_get_regime_no_briefs_returns_error(tmp_path, monkeypatch):
    db_path = tmp_path / "slv.db"
    db.connect(db_path).close()
    monkeypatch.setenv("SLV_DB_PATH", str(db_path))

    result = json.loads(get_regime())
    assert "error" in result


def test_search_journal_matches_all_keywords_case_insensitive(tmp_path, monkeypatch):
    db_path = tmp_path / "slv.db"
    conn = db.connect(db_path)
    journal.open_thesis(conn, "long", "68-69", 65.8, 74.0, 8, 3, rationale="Pullback to EMA20, RSI oversold")
    journal.open_thesis(conn, "short", "80-81", 83.0, 75.0, 5, 2, rationale="Breakdown below range")
    monkeypatch.setenv("SLV_DB_PATH", str(db_path))

    results = json.loads(search_journal("pullback ema20"))
    assert len(results) == 1
    assert "Pullback" in results[0]["rationale"]
    assert results[0]["status"] == "open"


def test_search_journal_empty_query_returns_all_most_recent_first(tmp_path, monkeypatch):
    db_path = tmp_path / "slv.db"
    conn = db.connect(db_path)
    journal.open_thesis(conn, "long", "68-69", 65.8, 74.0, 8, 3, rationale="first")
    journal.open_thesis(conn, "long", "68-69", 65.8, 74.0, 8, 3, rationale="second")
    monkeypatch.setenv("SLV_DB_PATH", str(db_path))

    results = json.loads(search_journal(""))
    assert [r["rationale"] for r in results] == ["second", "first"]


def test_search_journal_respects_limit(tmp_path, monkeypatch):
    db_path = tmp_path / "slv.db"
    conn = db.connect(db_path)
    for i in range(5):
        journal.open_thesis(conn, "long", "68-69", 65.8, 74.0, 8, 3, rationale=f"thesis {i}")
    monkeypatch.setenv("SLV_DB_PATH", str(db_path))

    results = json.loads(search_journal("", limit=2))
    assert len(results) == 2


def test_search_journal_includes_score_only_for_closed_and_scored(tmp_path, monkeypatch):
    db_path = tmp_path / "slv.db"
    conn = db.connect(db_path)
    conn.execute(
        "INSERT INTO prices (date, open, high, low, close, symbol, source) "
        "VALUES ('2026-08-20', 68, 69, 67, 68, 'SI=F', 'test')"
    )
    conn.commit()

    open_id = journal.open_thesis(conn, "long", "68-69", 65.8, 74.0, 8, 3, rationale="pullback ema20 setup")
    closed_unscored_id = journal.open_thesis(conn, "long", "68-69", 65.8, 74.0, 8, 3, rationale="pullback ema20 also")
    journal.close_thesis(conn, closed_unscored_id, entry_price=68.5, exit_price=71.0, closed_at="2026-08-21T00:00:00")

    scored_id = journal.open_thesis(conn, "long", "68-69", 65.8, 74.0, 8, 3, rationale="pullback ema20 scored")
    journal.close_thesis(conn, scored_id, entry_price=68.5, exit_price=71.0, closed_at="2026-08-21T00:00:00")
    journal.score_thesis(conn, scored_id)
    monkeypatch.setenv("SLV_DB_PATH", str(db_path))

    results = {r["id"]: r for r in json.loads(search_journal("pullback ema20"))}
    assert results[open_id]["status"] == "open"
    assert "r_multiple" not in results[open_id]
    assert results[closed_unscored_id]["status"] == "closed"
    assert "r_multiple" not in results[closed_unscored_id]  # closed but not yet scored
    assert results[scored_id]["status"] == "closed"
    # risk = entry(68.5) - invalidation(65.8) = 2.7; reward = exit(71.0) - entry(68.5) = 2.5
    assert results[scored_id]["r_multiple"] == pytest.approx(2.5 / 2.7)
    assert "checks" in results[scored_id]
