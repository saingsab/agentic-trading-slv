from unittest.mock import patch

import pandas as pd
import pytest

from slv import db
from slv.fetch import prices


def _fake_ohlcv():
    return pd.DataFrame(
        {
            "Open": [23.10, 23.40],
            "High": [23.50, 23.60],
            "Low": [23.00, 23.20],
            "Close": [23.35, 23.55],
            "Volume": [1000.0, 1200.0],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
    )


def test_fetch_prices_upserts(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        with patch("slv.fetch.prices.yf.download", return_value=_fake_ohlcv()):
            n = prices.fetch_prices(conn, symbol="SI=F")
        assert n == 2

        rows = conn.execute(
            "SELECT date, open, high, low, close, volume, symbol, source "
            "FROM prices ORDER BY date"
        ).fetchall()
        assert rows == [
            ("2026-01-02", 23.10, 23.50, 23.00, 23.35, 1000.0, "SI=F", "yfinance"),
            ("2026-01-05", 23.40, 23.60, 23.20, 23.55, 1200.0, "SI=F", "yfinance"),
        ]

        # Re-fetching the same window must converge, not duplicate.
        with patch("slv.fetch.prices.yf.download", return_value=_fake_ohlcv()):
            prices.fetch_prices(conn, symbol="SI=F")
        count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        assert count == 2
    finally:
        conn.close()


def test_fetch_prices_raises_on_empty_download(tmp_path):
    conn = db.connect(tmp_path / "slv.db")
    try:
        with patch("slv.fetch.prices.yf.download", return_value=pd.DataFrame()):
            with pytest.raises(RuntimeError, match="no data"):
                prices.fetch_prices(conn, symbol="SI=F")
    finally:
        conn.close()
