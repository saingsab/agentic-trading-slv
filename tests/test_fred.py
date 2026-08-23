from unittest.mock import patch

import pytest

from slv import db
from slv.fetch import fred


def _fake_response(observations):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"observations": observations}

    return FakeResponse()


def test_fetch_series_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(fred, "FRED_API_KEY", None)
    conn = db.connect(tmp_path / "slv.db")
    try:
        with pytest.raises(RuntimeError, match="FRED_API_KEY"):
            fred.fetch_series(conn, "DFII10")
    finally:
        conn.close()


def test_fetch_series_skips_missing_marker_and_upserts(tmp_path, monkeypatch):
    monkeypatch.setattr(fred, "FRED_API_KEY", "test-key")
    observations = [
        {"date": "2026-01-02", "value": "1.85"},
        {"date": "2026-01-03", "value": "."},  # FRED's missing-data marker
        {"date": "2026-01-06", "value": "1.90"},
    ]

    conn = db.connect(tmp_path / "slv.db")
    try:
        with patch("slv.fetch.fred.requests.get", return_value=_fake_response(observations)):
            n = fred.fetch_series(conn, "DFII10")
        assert n == 2  # the "." row is dropped, not stored as garbage

        rows = conn.execute(
            "SELECT date, value FROM macro WHERE series_id = 'DFII10' ORDER BY date"
        ).fetchall()
        assert rows == [("2026-01-02", 1.85), ("2026-01-06", 1.90)]

        # Re-fetching the same observations must not duplicate or change rows.
        with patch("slv.fetch.fred.requests.get", return_value=_fake_response(observations)):
            fred.fetch_series(conn, "DFII10")
        count = conn.execute(
            "SELECT COUNT(*) FROM macro WHERE series_id = 'DFII10'"
        ).fetchone()[0]
        assert count == 2
    finally:
        conn.close()


def test_fetch_series_raises_on_no_usable_observations(tmp_path, monkeypatch):
    monkeypatch.setattr(fred, "FRED_API_KEY", "test-key")
    conn = db.connect(tmp_path / "slv.db")
    try:
        with patch("slv.fetch.fred.requests.get", return_value=_fake_response([{"date": "2026-01-02", "value": "."}])):
            with pytest.raises(RuntimeError, match="no usable observations"):
                fred.fetch_series(conn, "DFII10")
    finally:
        conn.close()
