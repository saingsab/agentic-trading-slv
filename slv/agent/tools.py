"""The agent's three read-only tools (PLAN.md Phase 5): get_indicators,
get_regime, search_journal.

Each function's source gets shipped into wherever CodeAgent executes code
-- the Docker sandbox, when loop.py runs with sandbox=True -- because
that's how smolagents' CodeAgent paradigm works: tool calls are just
Python function calls inside LLM-generated code, so the function itself
has to exist in that execution context. That means every import must
happen inside the function body (nothing at module level is carried
over), and nothing here can import from `slv` -- the package isn't
installed in the sandbox image.

The DB path is read from SLV_DB_PATH, defaulting to /data/slv.db (where
loop.py mounts data/ read-only inside the sandbox). Overriding that env
var is also how tests exercise these functions directly, in-process, with
a temp DB -- the only way to unit-test something this self-contained,
since there's no slv.config to inject a path through.

Never write through these: every query is a SELECT. The agent proposes,
it doesn't decide (CLAUDE.md) -- these tools don't give it a way to.
"""
from __future__ import annotations

from smolagents import tool


@tool
def get_indicators(date: str = "") -> str:
    """Returns silver's computed technical indicators for one date.

    Args:
        date: ISO date (YYYY-MM-DD) to look up, or "" for the most recent
            date with computed indicators.

    Returns:
        A raw JSON string -- call json.loads() on the result before
        indexing into it. Once parsed, it's an object with keys: date,
        atr14, ema20, ema50, sma200, rsi14, rvol20, gsr, dist_ema20_atr,
        range_pctile_60d. A value is null wherever there wasn't enough
        price history yet to compute it. If no row matches, the parsed
        object has an "error" key instead.
    """
    import json
    import os
    import sqlite3

    db_path = os.environ.get("SLV_DB_PATH", "/data/slv.db")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        if date:
            cursor = conn.execute("SELECT * FROM indicators WHERE date = ?", (date,))
        else:
            cursor = conn.execute("SELECT * FROM indicators ORDER BY date DESC LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            return json.dumps({"error": f"no indicators row for date={date or 'latest'}"})
        columns = [d[0] for d in cursor.description]
        return json.dumps(dict(zip(columns, row)))
    finally:
        conn.close()


@tool
def get_regime() -> str:
    """Returns the most recent regime snapshot: real yield trend, DXY
    trend, gold/silver ratio percentile, and COT managed-money net
    percentile. This is read from the last brief `slv brief` generated,
    not recomputed here -- run `slv brief` first if it's stale.

    Returns:
        A raw JSON string -- call json.loads() on the result before
        indexing into it. Once parsed, it's an object with keys: as_of,
        real_yield_trend, dxy_trend, gsr_percentile, cot_percentile. Trend
        values are "rising", "falling", or null; percentiles are 0.0-1.0
        or null. If no brief has ever been generated, the parsed object
        has an "error" key instead.
    """
    import json
    import os
    import sqlite3

    db_path = os.environ.get("SLV_DB_PATH", "/data/slv.db")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT facts_json FROM briefs ORDER BY date DESC LIMIT 1").fetchone()
        if row is None:
            return json.dumps({"error": "no brief has been generated yet; run `slv brief`"})
        facts = json.loads(row[0])
        result = {"as_of": facts["as_of"]}
        result.update(facts["regime"])
        return json.dumps(result)
    finally:
        conn.close()


@tool
def search_journal(query: str, limit: int = 5) -> str:
    """Searches theses by keyword against their rationale text.

    There's no structured setup-type/tag field in the journal schema, only
    the free-text rationale written when a thesis was opened -- so this
    only finds a thesis if its rationale happens to mention the words
    you're searching for.

    Args:
        query: Space-separated keywords, e.g. "pullback EMA20". A thesis
            matches only if its rationale contains every keyword
            (case-insensitive substring match).
        limit: Maximum number of matches to return, most recently created
            first.

    Returns:
        A raw JSON string -- call json.loads() on the result before
        indexing into it. Once parsed, it's a list of objects, each with
        id, created_at, direction, entry_zone, invalidation, target,
        rationale, and status ("open" or "closed"). Closed theses that
        have been scored (via `slv journal score`) also include
        r_multiple, process_score, and checks (the 5-item process rubric,
        true/false per item) -- process_score is independent of
        r_multiple by design (CLAUDE.md: report expectancy/R, never win
        rate; grade process, not P&L).
    """
    import json
    import os
    import sqlite3

    db_path = os.environ.get("SLV_DB_PATH", "/data/slv.db")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute(
            "SELECT t.id, t.created_at, t.direction, t.entry_zone, t.invalidation, "
            "t.target, t.rationale, o.thesis_id, o.r_multiple, o.process_score, "
            "o.process_detail_json "
            "FROM theses t LEFT JOIN thesis_outcomes o ON o.thesis_id = t.id "
            # id DESC as a tiebreaker: created_at only has second precision
            # (journal.py), so theses opened within the same second would
            # otherwise sort in an undefined order.
            "ORDER BY t.created_at DESC, t.id DESC"
        )
        keywords = [w.lower() for w in query.split() if w]
        results = []
        for (tid, created_at, direction, entry_zone, invalidation, target,
             rationale, outcome_id, r_multiple, process_score, detail_json) in cursor:
            haystack = (rationale or "").lower()
            # a bare generator expression here (`all(k in haystack for k in
            # keywords)`) trips smolagents' tool-source validator, which
            # only special-cases list/dict/set comprehensions, not
            # GeneratorExp -- it misreports the loop variable as
            # undefined. A list comprehension inside all() sidesteps it.
            if keywords and not all([k in haystack for k in keywords]):
                continue

            entry = {
                "id": tid,
                "created_at": created_at,
                "direction": direction,
                "entry_zone": entry_zone,
                "invalidation": invalidation,
                "target": target,
                "rationale": rationale,
                "status": "closed" if outcome_id is not None else "open",
            }
            if outcome_id is not None and process_score is not None:
                entry["r_multiple"] = r_multiple
                entry["process_score"] = process_score
                entry["checks"] = json.loads(detail_json)["checks"] if detail_json else None
            results.append(entry)
            if len(results) >= limit:
                break
        return json.dumps(results)
    finally:
        conn.close()
