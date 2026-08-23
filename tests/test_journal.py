import json

import pytest

from slv import db, journal


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_parse_entry_zone_sorts_low_first():
    assert journal.parse_entry_zone("72.50-73.20") == (72.50, 73.20)
    assert journal.parse_entry_zone("73.20-72.50") == (72.50, 73.20)


def test_parse_entry_zone_rejects_bad_format():
    with pytest.raises(ValueError, match="low-high"):
        journal.parse_entry_zone("72.50")


def test_r_multiple_long():
    # risk = 100-90 = 10; reward = 120-100 = 20 -> R = 2.0
    assert journal.r_multiple("long", entry_price=100, exit_price=120, invalidation=90) == 2.0
    # a loss: reward = 88-100 = -12 -> R = -1.2
    assert journal.r_multiple("long", entry_price=100, exit_price=88, invalidation=90) == pytest.approx(-1.2)


def test_r_multiple_short():
    # risk = 110-100 = 10; reward = 100-80 = 20 -> R = 2.0
    assert journal.r_multiple("short", entry_price=100, exit_price=80, invalidation=110) == 2.0


def test_r_multiple_rejects_nonsensical_invalidation():
    with pytest.raises(ValueError, match="wrong side"):
        journal.r_multiple("long", entry_price=100, exit_price=120, invalidation=105)


def test_check_falsifiable():
    assert journal.check_falsifiable("long", "90-95", invalidation=85) is True
    assert journal.check_falsifiable("long", "90-95", invalidation=92) is False
    assert journal.check_falsifiable("short", "90-95", invalidation=100) is True
    assert journal.check_falsifiable("short", "90-95", invalidation=92) is False


def test_check_honored():
    assert journal.check_honored("long", invalidation=90, exit_price=91) is True
    assert journal.check_honored("long", invalidation=90, exit_price=89) is False
    assert journal.check_honored("short", invalidation=110, exit_price=109) is True
    assert journal.check_honored("short", invalidation=110, exit_price=111) is False


def test_check_in_condition():
    assert journal.check_in_condition("90-95", entry_price=92) is True
    assert journal.check_in_condition("90-95", entry_price=95) is True  # inclusive boundary
    assert journal.check_in_condition("90-95", entry_price=89.99) is False


def test_check_pre_committed_is_always_true():
    assert journal.check_pre_committed() is True


# ---------------------------------------------------------------------------
# theses: write-once
# ---------------------------------------------------------------------------


def test_open_thesis_rejects_invalid_direction(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        with pytest.raises(ValueError, match="direction"):
            journal.open_thesis(conn, "sideways", "70-73", 68, 79, 8, 3)
    finally:
        conn.close()


def test_open_thesis_rejects_invalidation_on_wrong_side(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        with pytest.raises(ValueError, match="below the entry zone"):
            journal.open_thesis(conn, "long", "70-73", invalidation=71, target=79, size_pct_equity=8, leverage=3)
    finally:
        conn.close()


def test_open_thesis_writes_a_row_and_has_no_update_path(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        thesis_id = journal.open_thesis(conn, "long", "72.50-73.20", 70.80, 79.00, 8, 3)
        assert thesis_id == 1

        thesis = journal.get_thesis(conn, thesis_id)
        assert thesis["direction"] == "long"
        assert thesis["entry_zone"] == "72.5-73.2"
        assert thesis["status"] == "open"

        # journal.py exposes no function that can UPDATE the theses table --
        # this is the actual enforcement mechanism, not a runtime check.
        assert not hasattr(journal, "update_thesis")
        assert not hasattr(journal, "edit_thesis")
    finally:
        conn.close()


def test_open_thesis_second_call_creates_a_new_row_not_an_overwrite(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        first = journal.open_thesis(conn, "long", "72.50-73.20", 70.80, 79.00, 8, 3)
        second = journal.open_thesis(conn, "long", "72.50-73.20", 70.80, 79.00, 8, 3)
        assert first != second
        assert conn.execute("SELECT COUNT(*) FROM theses").fetchone()[0] == 2
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# thesis_outcomes
# ---------------------------------------------------------------------------


def test_close_thesis_requires_an_existing_thesis(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        with pytest.raises(ValueError, match="no thesis"):
            journal.close_thesis(conn, thesis_id=999, entry_price=73.0, exit_price=70.60)
    finally:
        conn.close()


def test_close_thesis_refuses_to_close_twice(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        thesis_id = journal.open_thesis(conn, "long", "72.50-73.20", 70.80, 79.00, 8, 3)
        journal.close_thesis(conn, thesis_id, entry_price=73.0, exit_price=70.60)
        with pytest.raises(ValueError, match="already closed"):
            journal.close_thesis(conn, thesis_id, entry_price=73.0, exit_price=71.0)
    finally:
        conn.close()


def test_close_thesis_leaves_score_null_until_journal_score_runs(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        thesis_id = journal.open_thesis(conn, "long", "72.50-73.20", 70.80, 79.00, 8, 3)
        journal.close_thesis(conn, thesis_id, entry_price=73.0, exit_price=70.60)
        row = conn.execute(
            "SELECT r_multiple, process_score FROM thesis_outcomes WHERE thesis_id = ?",
            (thesis_id,),
        ).fetchone()
        assert row == (None, None)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# check_sized + score_thesis: needs prices/events in the DB
# ---------------------------------------------------------------------------


def _seed_prices_and_events(conn):
    # D3 and D5 are prior CPI occurrences (used as history); D7 is the one
    # that falls inside the thesis's holding window and gets checked.
    prices = [
        ("2025-01-01", 100.0),
        ("2025-01-08", 100.0),
        ("2025-01-09", 103.0),  # CPI #1: |103/100-1| = 0.03
        ("2025-01-16", 103.0),
        ("2025-01-17", 108.0),  # CPI #2: |108/103-1| = 0.048543...
        ("2025-01-24", 108.0),
        ("2025-01-25", 115.0),  # CPI #3: inside the holding window, being checked
    ]
    for date, close in prices:
        conn.execute(
            "INSERT INTO prices (date, open, high, low, close, symbol, source) "
            "VALUES (?, ?, ?, ?, ?, 'SI=F', 'test')",
            (date, close, close, close, close),
        )
    for date in ("2025-01-09", "2025-01-17", "2025-01-25"):
        conn.execute(
            "INSERT INTO events (date, name, actual, consensus, prior, surprise_z) "
            "VALUES (?, 'CPI', NULL, NULL, NULL, NULL)",
            (date,),
        )
    conn.commit()


def test_check_sized_passes_within_cap(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        _seed_prices_and_events(conn)
        thesis_id = journal.open_thesis(conn, "long", "108-108", 90, 120, size_pct_equity=50, leverage=3)
        # p90 of [0.03, 0.048543...] with numpy linear interp:
        # position = 0.9*(2-1) = 0.9 -> 0.03 + 0.9*(0.048543-0.03) = 0.046689...
        # equity_impact = 0.046689 * 3 * 100 * 0.5 = 7.0034% -> under the 15% cap
        passed, detail = journal.check_sized(
            conn, thesis_id, created_at="2025-01-20T00:00:00", closed_at="2025-01-26T00:00:00",
            leverage=3, size_pct_equity=50,
        )
        assert passed is True
        assert len(detail["checked"]) == 1
        assert detail["checked"][0]["p90_equity_impact_pct"] == pytest.approx(7.0034, abs=0.01)
        assert detail["checked"][0]["within_cap"] is True
    finally:
        conn.close()


def test_check_sized_fails_over_cap(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        _seed_prices_and_events(conn)
        thesis_id = journal.open_thesis(conn, "long", "108-108", 90, 120, size_pct_equity=100, leverage=20)
        passed, detail = journal.check_sized(
            conn, thesis_id, created_at="2025-01-20T00:00:00", closed_at="2025-01-26T00:00:00",
            leverage=20, size_pct_equity=100,
        )
        assert passed is False
        assert detail["checked"][0]["within_cap"] is False
    finally:
        conn.close()


def test_check_sized_skips_events_with_no_prior_history(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        conn.execute(
            "INSERT INTO prices (date, open, high, low, close, symbol, source) "
            "VALUES ('2025-01-25', 115, 115, 115, 115, 'SI=F', 'test')"
        )
        conn.execute(
            "INSERT INTO events (date, name, actual, consensus, prior, surprise_z) "
            "VALUES ('2025-01-25', 'FOMC', NULL, NULL, NULL, NULL)"
        )
        conn.commit()
        thesis_id = journal.open_thesis(conn, "long", "108-108", 90, 120, size_pct_equity=50, leverage=3)
        passed, detail = journal.check_sized(
            conn, thesis_id, created_at="2025-01-20T00:00:00", closed_at="2025-01-26T00:00:00",
            leverage=3, size_pct_equity=50,
        )
        assert passed is True  # vacuously -- nothing checkable failed
        assert detail["checked"] == []
        assert len(detail["skipped"]) == 1
        assert detail["skipped"][0]["name"] == "FOMC"
    finally:
        conn.close()


def test_score_thesis_end_to_end(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        _seed_prices_and_events(conn)
        thesis_id = journal.open_thesis(
            conn, "long", "107-109", invalidation=100, target=120, size_pct_equity=50, leverage=3
        )
        journal.close_thesis(
            conn, thesis_id, entry_price=108, exit_price=115, closed_at="2025-01-26T00:00:00"
        )
        # backdate created_at into the fixture's window without a code path
        # that mutates theses -- direct SQL here is the test poking at
        # fixture setup, not journal.py exposing a way to do this.
        conn.execute(
            "UPDATE theses SET created_at = ? WHERE id = ?", ("2025-01-20T00:00:00", thesis_id)
        )
        conn.commit()

        result = journal.score_thesis(conn, thesis_id)

        # r_multiple: risk = 108-100 = 8; reward = 115-108 = 7 -> R = 0.875
        assert result.r_multiple == pytest.approx(0.875)
        # all 5 checks should pass: falsifiable (100<107), sized (~7% < 15%),
        # honored (115 >= 100), in_condition (107<=108<=109), pre_committed (always)
        assert result.score == 1.0
        assert all(result.detail["checks"].values())

        # persisted, not just returned
        row = conn.execute(
            "SELECT r_multiple, process_score, process_detail_json FROM thesis_outcomes WHERE thesis_id = ?",
            (thesis_id,),
        ).fetchone()
        assert row[0] == pytest.approx(0.875)
        assert row[1] == 1.0
        assert json.loads(row[2])["checks"]["honored"] is True
    finally:
        conn.close()


def test_score_thesis_flags_a_failed_check(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        _seed_prices_and_events(conn)
        thesis_id = journal.open_thesis(
            conn, "long", "107-109", invalidation=100, target=120, size_pct_equity=50, leverage=3
        )
        # exit below invalidation: the stop was not honored
        journal.close_thesis(
            conn, thesis_id, entry_price=108, exit_price=95, closed_at="2025-01-26T00:00:00"
        )
        conn.execute(
            "UPDATE theses SET created_at = ? WHERE id = ?", ("2025-01-20T00:00:00", thesis_id)
        )
        conn.commit()

        result = journal.score_thesis(conn, thesis_id)
        assert result.detail["checks"]["honored"] is False
        assert result.score == pytest.approx(4 / 5)
    finally:
        conn.close()


def test_score_all_unscored_only_scores_closed_theses_missing_a_score(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        _seed_prices_and_events(conn)
        closed_and_scored = journal.open_thesis(conn, "long", "107-109", 100, 120, 50, 3)
        journal.close_thesis(conn, closed_and_scored, 108, 115, closed_at="2025-01-26T00:00:00")
        conn.execute("UPDATE theses SET created_at = ? WHERE id = ?", ("2025-01-20T00:00:00", closed_and_scored))

        closed_unscored = journal.open_thesis(conn, "long", "107-109", 100, 120, 50, 3)
        journal.close_thesis(conn, closed_unscored, 108, 112, closed_at="2025-01-26T00:00:00")
        conn.execute("UPDATE theses SET created_at = ? WHERE id = ?", ("2025-01-20T00:00:00", closed_unscored))

        still_open = journal.open_thesis(conn, "long", "107-109", 100, 120, 50, 3)
        conn.commit()

        journal.score_thesis(conn, closed_and_scored)  # pre-score one of them
        results = journal.score_all_unscored(conn)

        assert [r.thesis_id for r in results] == [closed_unscored]
        assert not journal.is_closed(conn, still_open)
    finally:
        conn.close()
