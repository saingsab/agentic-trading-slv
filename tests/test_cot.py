from unittest.mock import patch

import pytest

from slv import db
from slv.fetch import cot


def _fake_response(records):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return records

    return FakeResponse()


def _record(report_date, mm_long, mm_short, open_interest):
    return {
        "report_date_as_yyyy_mm_dd": f"{report_date}T00:00:00.000",
        "noncomm_positions_long_all": str(mm_long),
        "noncomm_positions_short_all": str(mm_short),
        "open_interest_all": str(open_interest),
    }


def test_fetch_cot_computes_mm_net_and_upserts(tmp_path):
    records = [_record("2026-08-18", 38353, 14728, 120117)]

    conn = db.connect(tmp_path / "slv.db")
    try:
        with patch("slv.fetch.cot.requests.get", return_value=_fake_response(records)):
            n = cot.fetch_cot(conn)
        assert n == 1

        row = conn.execute(
            "SELECT report_date, mm_net, mm_long, mm_short, open_interest FROM cot"
        ).fetchone()
        assert row == ("2026-08-18", 38353 - 14728, 38353.0, 14728.0, 120117.0)

        # Re-fetching identical data must converge, not duplicate.
        with patch("slv.fetch.cot.requests.get", return_value=_fake_response(records)):
            cot.fetch_cot(conn)
        count = conn.execute("SELECT COUNT(*) FROM cot").fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def test_fetch_cot_raises_on_missing_required_field(tmp_path):
    bad_record = _record("2026-08-18", 38353, 14728, 120117)
    del bad_record["open_interest_all"]

    conn = db.connect(tmp_path / "slv.db")
    try:
        with patch("slv.fetch.cot.requests.get", return_value=_fake_response([bad_record])):
            with pytest.raises(RuntimeError, match="missing expected fields"):
                cot.fetch_cot(conn)
    finally:
        conn.close()


def test_fetch_cot_raises_on_empty_response(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        with patch("slv.fetch.cot.requests.get", return_value=_fake_response([])):
            with pytest.raises(RuntimeError, match="no silver COT records"):
                cot.fetch_cot(conn)
    finally:
        conn.close()
