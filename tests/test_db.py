import sqlite3

from slv import db


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {row[0] for row in rows}


def test_connect_creates_every_table(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        assert set(db.TABLES) <= _table_names(conn)
    finally:
        conn.close()


def test_connect_is_idempotent(tmp_path):
    db_path = tmp_path / "slv.db"

    conn = db.connect(db_path)
    conn.execute(
        "INSERT INTO prices (date, open, high, low, close, symbol, source) "
        "VALUES ('2024-01-02', 23.5, 23.8, 23.4, 23.7, 'SI=F', 'yfinance')"
    )
    conn.commit()
    conn.close()

    # Reconnecting must not touch existing rows or raise on the re-run of
    # `CREATE TABLE IF NOT EXISTS`.
    conn = db.connect(db_path)
    try:
        rows = conn.execute("SELECT COUNT(*) FROM prices").fetchone()
        assert rows[0] == 1
    finally:
        conn.close()


def test_connect_creates_parent_directory(tmp_path):
    nested = tmp_path / "nested" / "slv.db"
    conn = db.connect(nested)
    conn.close()
    assert nested.exists()
