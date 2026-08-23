"""Assembles today's facts object from computed state and renders it as
templated markdown. No LLM here (see CLAUDE.md / PLAN.md Phase 4) -- plain
string formatting only, brief.py never calls out to any model.

Facts come from three places: the indicators table (already computed by
`slv compute`), raw macro/prices/cot for regime classification, and theses
for a compact "what's currently open" section. Scope call (not in
PLAN.md, low-stakes/easily revisited so made without stopping to ask):
event lookahead is 14 days, matching the typical CPI/NFP/FOMC monthly
cadence, and open theses are included since they're cheap to pull and
directly useful in a morning read.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date as date_cls
from pathlib import Path
from typing import Optional

import pandas as pd

from slv import config
from slv.compute import eventrisk, regime

EVENT_LOOKAHEAD_DAYS = 14


def _clean(value):
    """NaN -> None; everything else passed through. Keeps facts_json valid
    JSON (json.dumps chokes on NaN by default being non-standard) and
    lets the renderer treat "no value" uniformly.
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    return value


def _as_of(series: pd.Series, as_of: str):
    """Latest value at or before `as_of`. None if there's nothing yet."""
    eligible = series.loc[:as_of]
    if eligible.empty:
        return None
    return _clean(eligible.iloc[-1])


def _price_series(conn: sqlite3.Connection, symbol: str) -> pd.Series:
    df = pd.read_sql_query(
        "SELECT date, close FROM prices WHERE symbol = ? ORDER BY date",
        conn,
        params=(symbol,),
        index_col="date",
    )
    return df["close"]


def _macro_series(conn: sqlite3.Connection, series_id: str) -> pd.Series:
    df = pd.read_sql_query(
        "SELECT date, value FROM macro WHERE series_id = ? ORDER BY date",
        conn,
        params=(series_id,),
        index_col="date",
    )
    return df["value"]


def _cot_mm_net_series(conn: sqlite3.Connection) -> pd.Series:
    df = pd.read_sql_query(
        "SELECT report_date AS date, mm_net FROM cot ORDER BY report_date", conn, index_col="date"
    )
    return df["mm_net"]


def _latest_indicators_row(conn: sqlite3.Connection) -> dict:
    cursor = conn.execute("SELECT * FROM indicators ORDER BY date DESC LIMIT 1")
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("indicators table is empty; run `slv compute` first")
    columns = [d[0] for d in cursor.description]
    return {col: _clean(val) for col, val in zip(columns, row)}


def build_regime_snapshot(conn: sqlite3.Connection, as_of: str) -> regime.RegimeState:
    real_yield = _macro_series(conn, "DFII10")
    dxy = _price_series(conn, config.DXY_SYMBOL)
    mm_net = _cot_mm_net_series(conn)
    gsr = pd.read_sql_query(
        "SELECT date, gsr FROM indicators ORDER BY date", conn, index_col="date"
    )["gsr"]

    gsr_pctile = _as_of(regime.gsr_percentile(gsr), as_of)
    cot_pctile = _as_of(regime.cot_percentile(mm_net), as_of)
    return regime.build_regime_state(
        date=as_of,
        real_yield_trend_label=_as_of(regime.trend(real_yield), as_of),
        dxy_trend_label=_as_of(regime.trend(dxy), as_of),
        # _as_of returns None (not NaN) when there's no value yet;
        # build_regime_state's own NaN check would let a plain None
        # through unchanged, so normalize here instead of using
        # `x or float("nan")`, which would also catch a legitimate 0.0.
        gsr_pctile=float("nan") if gsr_pctile is None else gsr_pctile,
        cot_pctile=float("nan") if cot_pctile is None else cot_pctile,
    )


def build_upcoming_events(conn: sqlite3.Connection, as_of: str, lookahead_days: int = EVENT_LOOKAHEAD_DAYS) -> list[dict]:
    """Every seeded event within (as_of, as_of + lookahead_days], with
    historical move stats from prior occurrences of the same event name.
    Events with no prior history to compare against are still listed, with
    move stats as None -- omitting them would hide event risk you can't
    yet quantify, which is worse than showing "unknown".
    """
    horizon = (pd.Timestamp(as_of) + pd.Timedelta(days=lookahead_days)).strftime("%Y-%m-%d")
    upcoming = conn.execute(
        "SELECT date, name FROM events WHERE date > ? AND date <= ? ORDER BY date",
        (as_of, horizon),
    ).fetchall()
    if not upcoming:
        return []

    returns = eventrisk.daily_returns(_price_series(conn, config.INSTRUMENT_SYMBOL))

    results = []
    for event_date, name in upcoming:
        prior_dates = [
            d
            for (d,) in conn.execute(
                "SELECT date FROM events WHERE name = ? AND date < ? ORDER BY date", (name, event_date)
            ).fetchall()
        ]
        moves = eventrisk.event_day_moves(returns, prior_dates).dropna()
        days_away = (pd.Timestamp(event_date) - pd.Timestamp(as_of)).days

        if moves.empty:
            results.append(
                {"date": event_date, "name": name, "days_away": days_away,
                 "median_move_pct": None, "p90_move_pct": None, "n_observations": 0}
            )
        else:
            est = eventrisk.estimate_event_risk(name, moves, leverage=1.0)
            results.append(
                {"date": event_date, "name": name, "days_away": days_away,
                 "median_move_pct": est.median_move_pct, "p90_move_pct": est.p90_move_pct,
                 "n_observations": est.n_observations}
            )
    return results


def build_open_theses(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT t.id, t.created_at, t.direction, t.entry_zone, t.invalidation, "
        "t.target, t.size_pct_equity, t.leverage FROM theses t "
        "LEFT JOIN thesis_outcomes o ON o.thesis_id = t.id "
        "WHERE o.thesis_id IS NULL ORDER BY t.created_at"
    ).fetchall()
    columns = ["id", "created_at", "direction", "entry_zone", "invalidation", "target", "size_pct_equity", "leverage"]
    return [dict(zip(columns, row)) for row in rows]


def build_facts(conn: sqlite3.Connection, brief_date: str) -> dict:
    indicators_row = _latest_indicators_row(conn)
    as_of = indicators_row["date"]
    price = _as_of(_price_series(conn, config.INSTRUMENT_SYMBOL), as_of)
    regime_state = build_regime_snapshot(conn, as_of)

    return {
        "date": brief_date,
        "as_of": as_of,
        "price": {"symbol": config.INSTRUMENT_SYMBOL, "close": price},
        "indicators": {k: v for k, v in indicators_row.items() if k != "date"},
        "regime": {
            "real_yield_trend": regime_state.real_yield_trend,
            "dxy_trend": regime_state.dxy_trend,
            "gsr_percentile": regime_state.gsr_percentile,
            "cot_percentile": regime_state.cot_percentile,
        },
        "upcoming_events": build_upcoming_events(conn, as_of),
        "open_theses": build_open_theses(conn),
    }


def _fmt(value, spec: str = "", default: str = "n/a") -> str:
    if value is None:
        return default
    return format(value, spec)


def render_markdown(facts: dict) -> str:
    ind = facts["indicators"]
    reg = facts["regime"]
    lines = [
        f"# Silver (XAG/USD) Brief — {facts['date']}",
        "",
        f"_As of {facts['as_of']} close: {_fmt(facts['price']['close'], '.2f')} "
        f"({facts['price']['symbol']}, a futures proxy for spot — see CLAUDE.md domain notes)_",
        "",
        "## Indicators",
        "",
        "| Indicator | Value |",
        "|---|---|",
        f"| ATR14 | {_fmt(ind['atr14'], '.2f')} |",
        f"| EMA20 | {_fmt(ind['ema20'], '.2f')} |",
        f"| EMA50 | {_fmt(ind['ema50'], '.2f')} |",
        f"| SMA200 | {_fmt(ind['sma200'], '.2f')} |",
        f"| RSI14 | {_fmt(ind['rsi14'], '.1f')} |",
        f"| RVol20 (annualized) | {_fmt(ind['rvol20'], '.1%')} |",
        f"| Gold/Silver Ratio | {_fmt(ind['gsr'], '.1f')} |",
        f"| Distance from EMA20 (ATR units) | {_fmt(ind['dist_ema20_atr'], '+.2f')} |",
        f"| 60d Range Position | {_fmt(ind['range_pctile_60d'], '.0%')} |",
        "",
        "## Regime",
        "",
        f"- Real yield (DFII10) trend: **{reg['real_yield_trend'] or 'n/a'}**",
        f"- DXY trend: **{reg['dxy_trend'] or 'n/a'}**",
        f"- Gold/Silver ratio percentile (1y): {_fmt(reg['gsr_percentile'], '.0%')}",
        f"- COT managed-money net percentile (1y): {_fmt(reg['cot_percentile'], '.0%')}",
        "",
        f"## Upcoming events (next {EVENT_LOOKAHEAD_DAYS} days)",
        "",
    ]

    events = facts["upcoming_events"]
    if not events:
        lines.append("_None scheduled._")
    else:
        lines += ["| Date | Event | Days away | Median move | P90 move | N |", "|---|---|---|---|---|---|"]
        for e in events:
            lines.append(
                f"| {e['date']} | {e['name']} | {e['days_away']} | "
                f"{_fmt(e['median_move_pct'], '.1%')} | {_fmt(e['p90_move_pct'], '.1%')} | {e['n_observations']} |"
            )
    lines.append("")

    lines.append("## Open theses")
    lines.append("")
    theses = facts["open_theses"]
    if not theses:
        lines.append("_None open._")
    else:
        lines += [
            "| ID | Direction | Entry zone | Invalidation | Target | Size | Leverage | Opened |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for t in theses:
            lines.append(
                f"| {t['id']} | {t['direction']} | {t['entry_zone']} | {t['invalidation']} | "
                f"{t['target']} | {t['size_pct_equity']}% | {t['leverage']}x | {t['created_at'][:10]} |"
            )
    lines.append("")

    return "\n".join(lines)


def write_brief(conn: sqlite3.Connection, brief_date: Optional[str] = None) -> Path:
    """Build facts, render markdown, write it to briefs/, and record the
    result in the briefs table (INSERT OR REPLACE -- re-running for the
    same date regenerates deterministically rather than erroring or
    duplicating).
    """
    brief_date = brief_date or date_cls.today().isoformat()
    facts = build_facts(conn, brief_date)
    markdown = render_markdown(facts)

    config.BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.BRIEFS_DIR / f"{brief_date}.md"
    out_path.write_text(markdown)

    rel_path = out_path.relative_to(config.REPO_ROOT).as_posix()
    conn.execute(
        "INSERT OR REPLACE INTO briefs (date, path, facts_json) VALUES (?, ?, ?)",
        (brief_date, rel_path, json.dumps(facts, indent=2)),
    )
    conn.commit()
    return out_path
