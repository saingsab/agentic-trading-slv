"""The prediction journal: immutable theses, append-only outcomes, and
process scoring independent of P&L.

theses rows are immutable by omission, not by a runtime guard: this module
exposes no update/delete for theses at all. open_thesis() is the only way
to write one, and it only ever INSERTs. "Corrections create a new row"
(CLAUDE.md) -- there is no other code path.

thesis_outcomes is not immutable in that same sense: close_thesis() inserts
the raw outcome (entry, exit, closed_at); score_thesis()/score_all() UPDATE
that same row later to fill in r_multiple/process_score/process_detail_json.
That's a deliberate two-step design matching PLAN.md's cadence -- closing a
trade and grading its process are different rituals done at different times
(close whenever a trade ends; `slv journal score` weekly, in batch).

theses.status is written once at creation ('open') and never updated.
Whether a thesis is open or closed is derived by checking for a matching
thesis_outcomes row -- this was a deliberate choice (confirmed with the
user 2026-08-23) specifically so nothing in this module ever needs an
UPDATE against theses, which would risk breaking immutability by accident.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from slv.compute import eventrisk

VALID_DIRECTIONS = ("long", "short")

# "Sized" rubric check: an event's p90 equity impact, scaled by this
# thesis's actual size_pct_equity, must stay at or below this to pass.
# Confirmed with the user 2026-08-23 -- the high end of CLAUDE.md's own
# worked example ("3x leverage -> 6-15% of equity").
SIZED_MAX_EQUITY_IMPACT_PCT = 15.0

RUBRIC_CHECKS = ("falsifiable", "sized", "honored", "in_condition", "pre_committed")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_entry_zone(entry_zone: str) -> tuple[float, float]:
    """'72.50-73.20' -> (72.50, 73.20), low first regardless of input order."""
    parts = entry_zone.split("-")
    if len(parts) != 2:
        raise ValueError(f"entry_zone must be 'low-high', got {entry_zone!r}")
    a, b = float(parts[0]), float(parts[1])
    return (a, b) if a <= b else (b, a)


def r_multiple(direction: str, entry_price: float, exit_price: float, invalidation: float) -> float:
    """P&L expressed in units of the risk you accepted at entry.

    risk = distance from entry to invalidation; reward = distance from
    entry to exit, same sign convention. R=1 means the trade lost exactly
    your planned risk; R=-1 the same, in the other direction; R=2 means it
    made twice what you were risking.
    """
    if direction == "long":
        risk = entry_price - invalidation
        reward = exit_price - entry_price
    elif direction == "short":
        risk = invalidation - entry_price
        reward = entry_price - exit_price
    else:
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    if risk <= 0:
        raise ValueError(
            f"invalidation ({invalidation}) is on the wrong side of entry ({entry_price}) "
            f"for a {direction} -- risk must be positive"
        )
    return reward / risk


# ---------------------------------------------------------------------------
# theses: write-once
# ---------------------------------------------------------------------------


def open_thesis(
    conn: sqlite3.Connection,
    direction: str,
    entry_zone: str,
    invalidation: float,
    target: float,
    size_pct_equity: float,
    leverage: float,
    rationale: Optional[str] = None,
    claim_types: Optional[str] = None,
    provenance: Optional[str] = None,
) -> int:
    """Write a new thesis. There is no corresponding update function --
    ever. Get it wrong and the fix is a new row, per CLAUDE.md.

    Rejects a thesis whose invalidation can never actually invalidate it
    (e.g. an invalidation above the entry zone on a long) -- "Falsifiable"
    is enforced here at write time, not just checked later at scoring time.
    """
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {VALID_DIRECTIONS}, got {direction!r}")

    lo, hi = parse_entry_zone(entry_zone)
    if direction == "long" and not invalidation < lo:
        raise ValueError(
            f"invalidation ({invalidation}) must be below the entry zone ({lo}-{hi}) for a long"
        )
    if direction == "short" and not invalidation > hi:
        raise ValueError(
            f"invalidation ({invalidation}) must be above the entry zone ({lo}-{hi}) for a short"
        )

    cursor = conn.execute(
        "INSERT INTO theses "
        "(created_at, direction, entry_zone, invalidation, target, "
        " size_pct_equity, leverage, rationale, claim_types, provenance, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')",
        (
            _utc_now_iso(),
            direction,
            f"{lo}-{hi}",
            invalidation,
            target,
            size_pct_equity,
            leverage,
            rationale,
            claim_types,
            provenance,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_thesis(conn: sqlite3.Connection, thesis_id: int) -> dict:
    """Returns a plain dict rather than using conn.row_factory=sqlite3.Row,
    which would mutate a setting on a connection this module doesn't own
    (check_sized() reuses the same connection for pd.read_sql_query).
    """
    cursor = conn.execute("SELECT * FROM theses WHERE id = ?", (thesis_id,))
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"no thesis with id {thesis_id}")
    columns = [d[0] for d in cursor.description]
    return dict(zip(columns, row))


def is_closed(conn: sqlite3.Connection, thesis_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM thesis_outcomes WHERE thesis_id = ?", (thesis_id,)
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# thesis_outcomes: raw outcome now, score later
# ---------------------------------------------------------------------------


def close_thesis(
    conn: sqlite3.Connection,
    thesis_id: int,
    entry_price: float,
    exit_price: float,
    closed_at: Optional[str] = None,
) -> None:
    """Record the raw outcome of a closed trade. r_multiple/process_score
    are left NULL -- `slv journal score` fills those in later.
    """
    get_thesis(conn, thesis_id)  # raises if it doesn't exist
    if is_closed(conn, thesis_id):
        raise ValueError(f"thesis {thesis_id} is already closed")

    conn.execute(
        "INSERT INTO thesis_outcomes (thesis_id, closed_at, entry_price, exit_price) "
        "VALUES (?, ?, ?, ?)",
        (thesis_id, closed_at or _utc_now_iso(), entry_price, exit_price),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# process rubric -- five checks, independent of whether the trade made money
# ---------------------------------------------------------------------------


def check_falsifiable(direction: str, entry_zone: str, invalidation: float) -> bool:
    """A specific invalidation price was stated, on the side of the entry
    zone that can actually invalidate the thesis.

    Always true for anything written through open_thesis() (it enforces
    this at creation) -- kept as an explicit, visible rubric item rather
    than assumed, in case a row ever got in some other way.
    """
    lo, hi = parse_entry_zone(entry_zone)
    if direction == "long":
        return invalidation < lo
    if direction == "short":
        return invalidation > hi
    raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")


def check_honored(direction: str, invalidation: float, exit_price: float) -> bool:
    """Exit was at or better than the stated invalidation -- the loss
    never ran past your predetermined worst case.
    """
    if direction == "long":
        return exit_price >= invalidation
    if direction == "short":
        return exit_price <= invalidation
    raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")


def check_in_condition(entry_zone: str, entry_price: float) -> bool:
    """The actual fill fell inside the stated entry zone."""
    lo, hi = parse_entry_zone(entry_zone)
    return lo <= entry_price <= hi


def check_pre_committed() -> bool:
    """Always true for a thesis scored by this tool: created_at is set by
    the DB at INSERT time (never user-supplied), and close_thesis() cannot
    run without a pre-existing thesis row to reference. There's no code
    path that lets a trade get logged before its thesis exists.
    """
    return True


def check_sized(
    conn: sqlite3.Connection,
    thesis_id: int,
    created_at: str,
    closed_at: str,
    leverage: float,
    size_pct_equity: float,
) -> tuple[bool, dict]:
    """For every scheduled event inside [created_at, closed_at] (the
    thesis's holding window -- created_at is used as a proxy for entry
    time, since we don't capture a separate entry timestamp, so this errs
    toward flagging slightly more events, not fewer), does the p90
    historical move, scaled by this thesis's actual leverage and
    size_pct_equity, stay within SIZED_MAX_EQUITY_IMPACT_PCT?

    Passes vacuously if no events fall in the window. An event with no
    prior historical occurrences to compare against is skipped (noted in
    the returned detail, not scored either way) rather than failing the
    thesis on missing data.
    """
    start_date, end_date = created_at[:10], closed_at[:10]
    event_rows = conn.execute(
        "SELECT date, name FROM events WHERE date BETWEEN ? AND ? ORDER BY date",
        (start_date, end_date),
    ).fetchall()

    prices = pd.read_sql_query(
        "SELECT date, close FROM prices WHERE symbol = 'SI=F' ORDER BY date",
        conn,
        index_col="date",
    )["close"]
    returns = eventrisk.daily_returns(prices)

    checked = []
    skipped = []
    for event_date, event_name in event_rows:
        prior_dates = conn.execute(
            "SELECT date FROM events WHERE name = ? AND date < ? ORDER BY date",
            (event_name, event_date),
        ).fetchall()
        moves = eventrisk.event_day_moves(returns, [d for (d,) in prior_dates]).dropna()
        if moves.empty:
            skipped.append({"date": event_date, "name": event_name, "reason": "no prior history"})
            continue

        est = eventrisk.estimate_event_risk(event_name, moves, leverage)
        equity_impact_pct = est.p90_equity_impact_pct * 100 * (size_pct_equity / 100)
        checked.append(
            {
                "date": event_date,
                "name": event_name,
                "p90_equity_impact_pct": equity_impact_pct,
                "within_cap": equity_impact_pct <= SIZED_MAX_EQUITY_IMPACT_PCT,
            }
        )

    passed = all(c["within_cap"] for c in checked)
    return passed, {"cap_pct": SIZED_MAX_EQUITY_IMPACT_PCT, "checked": checked, "skipped": skipped}


@dataclass(frozen=True)
class ProcessScore:
    thesis_id: int
    r_multiple: float
    score: float  # fraction of the 5 checks that passed, 0.0-1.0
    detail: dict


def score_thesis(conn: sqlite3.Connection, thesis_id: int) -> ProcessScore:
    """Compute r_multiple and the 5-check process score for one closed
    thesis, and persist both into thesis_outcomes.
    """
    thesis = get_thesis(conn, thesis_id)
    cursor = conn.execute("SELECT * FROM thesis_outcomes WHERE thesis_id = ?", (thesis_id,))
    outcome_row = cursor.fetchone()
    if outcome_row is None:
        raise ValueError(f"thesis {thesis_id} has no outcome yet -- close it first")
    outcome = dict(zip([d[0] for d in cursor.description], outcome_row))

    rm = r_multiple(thesis["direction"], outcome["entry_price"], outcome["exit_price"], thesis["invalidation"])

    sized_pass, sized_detail = check_sized(
        conn,
        thesis_id,
        thesis["created_at"],
        outcome["closed_at"],
        thesis["leverage"],
        thesis["size_pct_equity"],
    )

    checks = {
        "falsifiable": check_falsifiable(thesis["direction"], thesis["entry_zone"], thesis["invalidation"]),
        "sized": sized_pass,
        "honored": check_honored(thesis["direction"], thesis["invalidation"], outcome["exit_price"]),
        "in_condition": check_in_condition(thesis["entry_zone"], outcome["entry_price"]),
        "pre_committed": check_pre_committed(),
    }
    score = sum(checks.values()) / len(RUBRIC_CHECKS)
    detail = {"checks": checks, "sized_detail": sized_detail}

    conn.execute(
        "UPDATE thesis_outcomes SET r_multiple = ?, process_score = ?, process_detail_json = ? "
        "WHERE thesis_id = ?",
        (rm, score, json.dumps(detail), thesis_id),
    )
    conn.commit()
    return ProcessScore(thesis_id=thesis_id, r_multiple=rm, score=score, detail=detail)


def score_all_unscored(conn: sqlite3.Connection) -> list[ProcessScore]:
    """Score every closed thesis that doesn't have a process_score yet."""
    rows = conn.execute(
        "SELECT thesis_id FROM thesis_outcomes WHERE process_score IS NULL ORDER BY thesis_id"
    ).fetchall()
    return [score_thesis(conn, thesis_id) for (thesis_id,) in rows]
