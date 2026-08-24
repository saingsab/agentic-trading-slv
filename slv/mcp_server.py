"""MCP server exposing the same three read tools as Phase 5's agent --
get_indicators, get_regime, search_journal -- directly to Claude Code, so
it reads slv's computed state instead of running a web search for
technicals (PLAN.md Phase 7).

These are thin wrappers around slv.agent.tools' exact implementations,
not a reimplementation: those functions are already read-only, already
self-contained, and already return the right JSON shape. The wrappers
exist only because MCP's @server.tool() introspects a plain function's
signature/docstring to build its schema -- it can't do that against a
smolagents SimpleTool instance directly, even though that instance is
itself callable.

Unlike Phase 5's tools, this process is NOT sandboxed: it's a normal
trusted local process you (or Claude Code) launch directly, not somewhere
LLM-generated code executes -- so SLV_DB_PATH is pointed at the real
data/slv.db, not a sandbox mount, and set here before slv.agent.tools is
even imported.
"""
from __future__ import annotations

import os

from slv import config

os.environ.setdefault("SLV_DB_PATH", str(config.DB_PATH))

from mcp.server.mcpserver import MCPServer  # noqa: E402 (after SLV_DB_PATH is set)

from slv.agent.tools import get_indicators as _get_indicators  # noqa: E402
from slv.agent.tools import get_regime as _get_regime  # noqa: E402
from slv.agent.tools import search_journal as _search_journal  # noqa: E402

server = MCPServer("slv")


@server.tool()
def get_indicators(date: str = "") -> str:
    """Returns silver's computed technical indicators for one date.

    Args:
        date: ISO date (YYYY-MM-DD) to look up, or "" for the most
            recent date with computed indicators.

    Returns:
        A raw JSON string -- parse it before indexing into it. Keys:
        date, atr14, ema20, ema50, sma200, rsi14, rvol20, gsr,
        dist_ema20_atr, range_pctile_60d. A value is null wherever there
        wasn't enough price history yet to compute it. If no row
        matches, the parsed object has an "error" key instead.
    """
    return _get_indicators(date)


@server.tool()
def get_regime() -> str:
    """Returns the most recent regime snapshot: real yield trend, DXY
    trend, gold/silver ratio percentile, and COT managed-money net
    percentile. Read from the last `slv brief` generated, not recomputed
    here -- run `slv brief` first if it's stale.

    Returns:
        A raw JSON string -- parse it before indexing into it. Keys:
        as_of, real_yield_trend, dxy_trend, gsr_percentile,
        cot_percentile. Trend values are "rising", "falling", or null;
        percentiles are 0.0-1.0 or null. If no brief has ever been
        generated, the parsed object has an "error" key instead.
    """
    return _get_regime()


@server.tool()
def search_journal(query: str = "", limit: int = 5) -> str:
    """Searches theses by keyword against their rationale text.

    There's no structured setup-type/tag field in the journal schema,
    only the free-text rationale written when a thesis was opened -- so
    this only finds a thesis if its rationale happens to mention the
    words you're searching for.

    Args:
        query: Space-separated keywords, e.g. "pullback EMA20". A thesis
            matches only if its rationale contains every keyword
            (case-insensitive substring match). Empty string matches
            every thesis.
        limit: Maximum number of matches to return, most recently
            created first.

    Returns:
        A raw JSON string -- parse it before indexing into it. Once
        parsed, a list of objects with id, created_at, direction,
        entry_zone, invalidation, target, rationale, and status ("open"
        or "closed"). Closed theses that have been scored (via
        `slv journal score`) also include r_multiple, process_score, and
        checks (the 5-item process rubric) -- process_score is
        independent of r_multiple by design (CLAUDE.md: report
        expectancy/R, never win rate; grade process, not P&L).
    """
    return _search_journal(query, limit)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
