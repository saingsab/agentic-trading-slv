import pytest

from slv import db
from slv.fetch import calendar


def _write_seed(path, rows):
    path.write_text("date,name\n" + "\n".join(f"{d},{n}" for d, n in rows) + "\n")


def test_load_calendar_inserts_rows(tmp_path):
    seed = tmp_path / "seed.csv"
    _write_seed(seed, [("2026-01-28", "FOMC"), ("2026-01-13", "CPI")])

    conn = db.connect(tmp_path / "slv.db")
    try:
        n = calendar.load_calendar(conn, seed)
        assert n == 2
        rows = conn.execute("SELECT date, name FROM events ORDER BY date").fetchall()
        assert rows == [("2026-01-13", "CPI"), ("2026-01-28", "FOMC")]
    finally:
        conn.close()


def test_load_calendar_does_not_clobber_manual_edits(tmp_path):
    """A re-seed must not wipe actual/consensus/prior a human filled in."""
    seed = tmp_path / "seed.csv"
    _write_seed(seed, [("2026-01-13", "CPI")])

    conn = db.connect(tmp_path / "slv.db")
    try:
        calendar.load_calendar(conn, seed)
        conn.execute(
            "UPDATE events SET actual = 3.1, consensus = 3.0, prior = 2.9, "
            "surprise_z = 0.4 WHERE date = '2026-01-13' AND name = 'CPI'"
        )
        conn.commit()

        # Re-running the fetcher must leave the filled-in row untouched.
        calendar.load_calendar(conn, seed)
        row = conn.execute(
            "SELECT actual, consensus, prior, surprise_z FROM events "
            "WHERE date = '2026-01-13' AND name = 'CPI'"
        ).fetchone()
        assert row == (3.1, 3.0, 2.9, 0.4)
    finally:
        conn.close()


def test_load_calendar_missing_file_raises(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        with pytest.raises(RuntimeError):
            calendar.load_calendar(conn, tmp_path / "does_not_exist.csv")
    finally:
        conn.close()


def test_seeded_calendar_file_loads(tmp_path):
    """The real data/calendar_seed.csv shipped in the repo is well-formed."""
    conn = db.connect(tmp_path / "slv.db")
    try:
        n = calendar.load_calendar(conn)  # default seed_path = data/calendar_seed.csv
        assert n > 0
    finally:
        conn.close()
