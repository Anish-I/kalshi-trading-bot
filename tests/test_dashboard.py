"""Tests for dashboard endpoints added in phase 3."""

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from dashboard import app as dashboard_app
from dashboard.app import app

client = TestClient(app)


def test_api_families_empty_gracefully(tmp_path, monkeypatch):
    # Force scorecard to read from empty tmp paths
    from engine import family_scorecard as fs

    orig_init = fs.FamilyScorecard.__init__

    def patched(self, *a, **kw):
        kw["order_ledger_path"] = tmp_path / "order_ledger.parquet"
        kw["trade_journal_path"] = tmp_path / "trade_journal.parquet"
        kw["cache_path"] = tmp_path / "family_scorecard.parquet"
        kw["combined_state_path"] = tmp_path / "combined_trader_state.json"
        orig_init(self, *a, **kw)

    monkeypatch.setattr(fs.FamilyScorecard, "__init__", patched)

    r = client.get("/api/families")
    assert r.status_code == 200
    body = r.json()
    assert "families" in body
    assert isinstance(body["families"], list)
    # registry has entries → each becomes a healthy no-data family
    for f in body["families"]:
        assert "family" in f
        assert "metrics" in f


def test_api_gate_recent_no_state_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dashboard_app,
        "GATE_STATE_FILES",
        {
            "combined_trader": tmp_path / "combined_trader_state.json",
            "ml_trader": tmp_path / "ml_trader_state.json",
            "pair_trader": tmp_path / "pair_trader_state.json",
        },
    )
    r = client.get("/api/gate/recent")
    assert r.status_code == 200
    body = r.json()
    for name in ("combined_trader", "ml_trader", "pair_trader"):
        assert body[name]["gate_last_decision"] is None
        assert body[name].get("error") == "state file missing"


def test_api_gate_recent_with_state(tmp_path, monkeypatch):
    state_path = tmp_path / "combined_trader_state.json"
    state_path.write_text(
        json.dumps(
            {
                "time": "2026-04-08T12:00:00Z",
                "gate_last_decision": {
                    "allowed": True,
                    "reason_code": "ok",
                    "family": "btc_15m",
                },
            }
        )
    )
    monkeypatch.setattr(
        dashboard_app,
        "GATE_STATE_FILES",
        {
            "combined_trader": state_path,
            "ml_trader": tmp_path / "nope1.json",
            "pair_trader": tmp_path / "nope2.json",
        },
    )
    r = client.get("/api/gate/recent")
    assert r.status_code == 200
    body = r.json()
    assert body["combined_trader"]["state_time"] == "2026-04-08T12:00:00Z"
    assert body["combined_trader"]["gate_last_decision"]["allowed"] is True
    assert body["combined_trader"]["gate_last_decision"]["family"] == "btc_15m"
    assert body["ml_trader"].get("error") == "state file missing"


def test_api_families_with_data(tmp_path, monkeypatch):
    from engine import family_scorecard as fs
    from config.market_registry import MARKET_FAMILIES

    # pick a real family + its series prefix
    fam_name, fam_obj = next(iter(MARKET_FAMILIES.items()))
    prefix_field = getattr(fam_obj, "series_prefix", "") or ""
    prefix = [p.strip() for p in str(prefix_field).split(",") if p.strip()][0]
    ticker = f"{prefix}XYZ-TEST"

    journal_df = pd.DataFrame(
        [
            {
                "time": "2026-04-08T11:00:00Z",
                "ticker": ticker,
                "action": "simulated",
                "settled": True,
                "won": True,
                "pnl_cents": 50,
                "rule_name": "test",
            },
            {
                "time": "2026-04-08T11:30:00Z",
                "ticker": ticker,
                "action": "simulated",
                "settled": True,
                "won": True,
                "pnl_cents": 40,
                "rule_name": "test",
            },
        ]
    )
    journal_path = tmp_path / "trade_journal.parquet"
    journal_df.to_parquet(journal_path, index=False)

    orig_init = fs.FamilyScorecard.__init__

    def patched(self, *a, **kw):
        kw["order_ledger_path"] = tmp_path / "order_ledger.parquet"
        kw["trade_journal_path"] = journal_path
        kw["cache_path"] = tmp_path / "family_scorecard.parquet"
        kw["combined_state_path"] = tmp_path / "combined_trader_state.json"
        orig_init(self, *a, **kw)

    monkeypatch.setattr(fs.FamilyScorecard, "__init__", patched)

    r = client.get("/api/families")
    assert r.status_code == 200
    body = r.json()
    assert "families" in body and len(body["families"]) > 0
    match = [f for f in body["families"] if f["family"] == fam_name]
    assert match, f"expected family {fam_name} in response"
    assert match[0]["metrics"]["trade_count"] >= 2
