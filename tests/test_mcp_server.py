"""These are thin wrappers around slv.agent.tools (already tested in
test_agent_tools.py) -- the only thing worth checking here is that the
wiring is correct: each MCP tool passes its args through unchanged and
returns exactly what the underlying tool returns, and that SLV_DB_PATH
defaults to the real DB.
"""
import json

from slv import config, db, journal
from slv.mcp_server import get_indicators, get_regime, search_journal


def test_slv_db_path_defaults_to_the_real_db():
    import os

    assert os.environ.get("SLV_DB_PATH") == str(config.DB_PATH)


def test_get_indicators_wrapper_matches_underlying_tool(tmp_path, monkeypatch):
    db_path = tmp_path / "slv.db"
    conn = db.connect(db_path)
    conn.execute("INSERT INTO indicators (date, atr14) VALUES ('2026-08-21', 1.85)")
    conn.commit()
    conn.close()
    monkeypatch.setenv("SLV_DB_PATH", str(db_path))

    result = json.loads(get_indicators(""))
    assert result["date"] == "2026-08-21"
    assert result["atr14"] == 1.85


def test_get_regime_wrapper_matches_underlying_tool(tmp_path, monkeypatch):
    db_path = tmp_path / "slv.db"
    conn = db.connect(db_path)
    facts = {"as_of": "2026-08-21", "regime": {"real_yield_trend": "rising", "dxy_trend": "falling",
                                                "gsr_percentile": 0.5, "cot_percentile": 0.2}}
    conn.execute("INSERT INTO briefs (date, path, facts_json) VALUES ('2026-08-21', 'x', ?)", (json.dumps(facts),))
    conn.commit()
    conn.close()
    monkeypatch.setenv("SLV_DB_PATH", str(db_path))

    result = json.loads(get_regime())
    assert result["real_yield_trend"] == "rising"
    assert result["gsr_percentile"] == 0.5


def test_search_journal_wrapper_defaults_to_empty_query_matching_everything(tmp_path, monkeypatch):
    db_path = tmp_path / "slv.db"
    conn = db.connect(db_path)
    journal.open_thesis(conn, "long", "68-69", 65.8, 74.0, 8, 3, rationale="anything at all")
    monkeypatch.setenv("SLV_DB_PATH", str(db_path))

    results = json.loads(search_journal())  # no args -- query="" limit=5 defaults
    assert len(results) == 1
    assert results[0]["rationale"] == "anything at all"


def test_search_journal_wrapper_respects_limit_arg(tmp_path, monkeypatch):
    db_path = tmp_path / "slv.db"
    conn = db.connect(db_path)
    for i in range(3):
        journal.open_thesis(conn, "long", "68-69", 65.8, 74.0, 8, 3, rationale=f"thesis {i}")
    monkeypatch.setenv("SLV_DB_PATH", str(db_path))

    results = json.loads(search_journal("", 2))
    assert len(results) == 2
