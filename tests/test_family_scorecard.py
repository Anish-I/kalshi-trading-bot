"""Tests for engine.family_scorecard.FamilyScorecard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from engine.family_scorecard import FamilyScorecard, FamilyHealth


@dataclass
class _Fam:
    name: str
    series_prefix: str
    budget_cents: int = 5000


FIXTURE_REGISTRY = {
    "btc_15m": _Fam("btc", "KXBTC15M"),
    "eth_5m": _Fam("eth", "KXETH5M"),
}


def _make_journal(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "trade_journal.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(p, index=False)
    return p


def _make_ledger(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "order_ledger.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(p, index=False)
    return p


def _now_iso(minutes_ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _good_journal_rows(ticker_prefix: str, n: int) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "time": _now_iso(i),
                "ticker": f"{ticker_prefix}-XYZ-{i}",
                "settled": True,
                "won": True,
                "pnl_cents": 10,
                "gate_reason_code": None,
            }
        )
    return rows


def _bad_journal_rows(ticker_prefix: str, n: int, win_rate: float) -> list[dict]:
    rows = []
    wins = int(n * win_rate)
    for i in range(n):
        won = i < wins
        rows.append(
            {
                "time": _now_iso(i),
                "ticker": f"{ticker_prefix}-XYZ-{i}",
                "settled": True,
                "won": won,
                "pnl_cents": 10 if won else -15,
                "gate_reason_code": None,
            }
        )
    return rows


def _ledger_rows(ticker_prefix: str, n: int, filled_frac: float = 1.0) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "ticker": f"{ticker_prefix}-XYZ-{i}",
                "submitted_count": 1,
                "filled_count": 1 if (i / n) < filled_frac else 0,
                "updated_at": _now_iso(i),
                "created_at": _now_iso(i),
            }
        )
    return rows


def _scorecard(tmp_path, journal, ledger, shadow=False, state_path=None):
    return FamilyScorecard(
        order_ledger_path=ledger,
        trade_journal_path=journal,
        market_registry=FIXTURE_REGISTRY,
        window_hours=48,
        cache_path=tmp_path / "family_scorecard.parquet",
        shadow_mode=shadow,
        combined_state_path=state_path or (tmp_path / "no_state.json"),
    )


def test_happy_path_good_family(tmp_path):
    journal = _make_journal(tmp_path, _good_journal_rows("KXBTC15M", 10))
    ledger = _make_ledger(tmp_path, _ledger_rows("KXBTC15M", 10))
    sc = _scorecard(tmp_path, journal, ledger)
    h = sc.get_family_health("btc_15m")
    assert h.healthy is True
    assert h.score == pytest.approx(1.0)
    assert h.throttle_multiplier == 1.0
    assert h.metrics["trade_count"] == 10
    assert h.metrics["win_rate"] == 1.0


def test_bad_streak_penalized(tmp_path):
    journal = _make_journal(tmp_path, _bad_journal_rows("KXBTC15M", 10, 0.3))
    ledger = _make_ledger(tmp_path, _ledger_rows("KXBTC15M", 10))
    sc = _scorecard(tmp_path, journal, ledger)
    h = sc.get_family_health("btc_15m")
    assert h.score < 1.0
    assert h.healthy is False
    assert h.throttle_multiplier < 1.0


def test_shadow_mode_forces_full_multiplier(tmp_path):
    journal = _make_journal(tmp_path, _bad_journal_rows("KXBTC15M", 10, 0.3))
    ledger = _make_ledger(tmp_path, _ledger_rows("KXBTC15M", 10))
    sc = _scorecard(tmp_path, journal, ledger, shadow=True)
    h = sc.get_family_health("btc_15m")
    assert h.score < 1.0  # score is still computed
    assert h.throttle_multiplier == 1.0  # but multiplier is forced to 1.0


def test_orphan_rate_penalty(tmp_path):
    journal = _make_journal(tmp_path, _good_journal_rows("KXBTC15M", 10))
    ledger = _make_ledger(tmp_path, _ledger_rows("KXBTC15M", 10))
    state = tmp_path / "state.json"
    state.write_text(
        '{"per_series": {"KXBTC15M-A": {"orphans": 8, "completed": 2}}}'
    )
    sc = _scorecard(tmp_path, journal, ledger, state_path=state)
    h = sc.get_family_health("btc_15m")
    assert h.metrics["orphan_rate"] > 0.3
    assert h.score <= 0.7


def test_cache_is_written(tmp_path):
    journal = _make_journal(tmp_path, _good_journal_rows("KXBTC15M", 5))
    ledger = _make_ledger(tmp_path, _ledger_rows("KXBTC15M", 5))
    sc = _scorecard(tmp_path, journal, ledger)
    sc.compute()
    assert sc.cache_path.exists()
    df = pd.read_parquet(sc.cache_path)
    assert "family" in df.columns
    assert set(df["family"]) == set(FIXTURE_REGISTRY.keys())


def test_zero_trades_returns_healthy_noop(tmp_path):
    journal = _make_journal(
        tmp_path, _good_journal_rows("KXBTC15M", 5)
    )  # only btc rows
    ledger = _make_ledger(tmp_path, _ledger_rows("KXBTC15M", 5))
    sc = _scorecard(tmp_path, journal, ledger)
    h = sc.get_family_health("eth_5m")  # no rows for eth
    assert h.metrics["trade_count"] == 0
    assert h.healthy is True
    assert h.throttle_multiplier == 1.0
