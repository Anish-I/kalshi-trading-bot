"""Tests for load_trade_archive / load_quotes_and_trades helpers."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from engine import market_archive


def _write_trades(dir_path, rows):
    dir_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(dir_path / "KXBTC15M_trades.parquet", index=False)


def _write_quotes(dir_path, rows):
    dir_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(dir_path / "KXBTC15M.parquet", index=False)


@pytest.fixture
def fake_archive(tmp_path, monkeypatch):
    root = tmp_path / "market_archive"
    root.mkdir()
    monkeypatch.setattr(market_archive, "ARCHIVE_DIR", root)
    return root


def _sample_trades(n=5):
    return [
        {
            "trade_id": f"t{i}",
            "ticker": "KXBTC15M-26APR0800-B100000",
            "created_time": f"2026-04-07T10:0{i}:00Z",
            "yes_price_cents": 50 + i,
            "no_price_cents": 50 - i,
            "count": 10 + i,
            "taker_side": "yes" if i % 2 == 0 else "no",
        }
        for i in range(n)
    ]


def _sample_quotes(n=3):
    return [
        {
            "timestamp": f"2026-04-07T10:0{i}:00Z",
            "ticker": "KXBTC15M-26APR0800-B100000",
            "yes_bid": 0.5,
            "yes_ask": 0.52,
            "no_bid": 0.48,
            "no_ask": 0.5,
            "volume": 100 + i,
        }
        for i in range(n)
    ]


def test_load_trade_archive_basic(fake_archive):
    _write_trades(fake_archive / "2026-04-07", _sample_trades(5))
    df = market_archive.load_trade_archive("KXBTC15M", "2026-04-07", "2026-04-07")
    assert len(df) == 5
    assert set(market_archive.TRADE_COLUMNS).issubset(df.columns)
    assert list(df["trade_id"]) == [f"t{i}" for i in range(5)]


def test_load_trade_archive_empty_day(fake_archive):
    # Only write to day 1; request a range that includes a day with no file
    _write_trades(fake_archive / "2026-04-07", _sample_trades(2))
    df = market_archive.load_trade_archive("KXBTC15M", "2026-04-06", "2026-04-08")
    assert len(df) == 2  # no exception, just the one day we wrote


def test_load_trade_archive_no_files(fake_archive):
    df = market_archive.load_trade_archive("KXBTC15M", "2026-04-01", "2026-04-03")
    assert df.empty
    assert list(df.columns) == market_archive.TRADE_COLUMNS


def test_load_quotes_and_trades(fake_archive):
    _write_trades(fake_archive / "2026-04-07", _sample_trades(5))
    _write_quotes(fake_archive / "2026-04-07", _sample_quotes(3))
    quotes, trades = market_archive.load_quotes_and_trades(
        "KXBTC15M", "2026-04-07", "2026-04-07"
    )
    assert len(quotes) == 3
    assert len(trades) == 5
    assert "yes_bid" in quotes.columns
    assert "trade_id" in trades.columns


def test_load_archive_still_works(fake_archive):
    _write_quotes(fake_archive / "2026-04-07", _sample_quotes(4))
    df = market_archive.load_archive(
        "KXBTC15M", date(2026, 4, 7), date(2026, 4, 7)
    )
    assert len(df) == 4
    assert "yes_bid" in df.columns
