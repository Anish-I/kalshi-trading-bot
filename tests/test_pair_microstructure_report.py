"""Tests for scripts/pair_microstructure_report.py — synthetic ledger fixtures."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pair_microstructure_report as pmr  # noqa: E402


def _make_synthetic_ledger(path: Path, n: int = 20) -> None:
    """Create a fake OrderLedger parquet with `n` pair rows."""
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(n):
        rows.append(
            {
                "strategy": "crypto_pair",
                "market_type": "btc_15m_pair",
                "ticker": f"KXBTC15M-25APR08{i:02d}-T{i}",
                "order_id": f"ord-{i}",
                "status": "settled" if i % 2 == 0 else "filled",
                "submitted_side": "yes",
                "submitted_price_cents": 40 + (i % 10),
                "submitted_count": 1,
                "filled_price_cents": 40 + (i % 10),
                "filled_count": 1,
                "settlement_result": "orphan" if i % 5 == 0 else "win",
                "pnl_cents": (-12 if i % 5 == 0 else 3),
                "created_at": (now - timedelta(days=(i % 7), hours=(i % 24))).isoformat(),
                "updated_at": now.isoformat(),
                "session_tag": "test",
                "signal_ref": "pair_signal",
                "slippage_cents": i % 6,
                "time_to_fill_ms": 100 + i,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_report_runs_and_produces_parquet(tmp_path: Path, monkeypatch):
    ledger_path = tmp_path / "order_ledger.parquet"
    reports_dir = tmp_path / "reports"
    _make_synthetic_ledger(ledger_path, n=20)

    rc = pmr.run(
        days=30,
        series_filter=None,
        ledger_path=ledger_path,
        log_path=tmp_path / "nope.log",
        state_path=tmp_path / "nope.json",
        reports_dir=reports_dir,
    )
    assert rc == 0

    out_files = list(reports_dir.glob("pair_microstructure_*.parquet"))
    assert len(out_files) == 1
    df = pd.read_parquet(out_files[0])
    for col in pmr.OUTPUT_COLUMNS:
        assert col in df.columns, f"missing column {col}"
    assert len(df) > 0


def test_empty_ledger_graceful(tmp_path: Path, capsys):
    ledger_path = tmp_path / "order_ledger.parquet"
    pd.DataFrame(
        columns=["strategy", "ticker", "status", "created_at", "signal_ref", "pnl_cents"]
    ).to_parquet(ledger_path, index=False)

    rc = pmr.run(
        days=30,
        series_filter=None,
        ledger_path=ledger_path,
        log_path=tmp_path / "nope.log",
        state_path=tmp_path / "nope.json",
        reports_dir=tmp_path / "reports",
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "No pair data found" in out


def test_spread_bucketing():
    assert pmr.bucket_spread(4) == "3-5c"
    assert pmr.bucket_spread(2.9) == "<3c"
    assert pmr.bucket_spread(5) == "5-7c"
    assert pmr.bucket_spread(7) == "7-10c"
    assert pmr.bucket_spread(10) == ">10c"
    assert pmr.bucket_spread(15.5) == ">10c"
