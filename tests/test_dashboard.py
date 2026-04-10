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


# -----------------------------------------------------------------------------
# Phase A — agents + proposals endpoints
# -----------------------------------------------------------------------------

def _write_fake_proposal(root: Path, pid: str, state: str = "pending") -> Path:
    d = root / state
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{pid}.json"
    p.write_text(json.dumps({
        "id": pid,
        "state": state,
        "agent": "weather-strategist",
        "agent_version": "0.2.0",
        "created_at": "2026-04-09T00:00:00+00:00",
        "snapshot_hash": "deadbeef",
        "current_config": {"ENTRY_PRICE_MIN_CENTS": 0},
        "proposed_config": {"ENTRY_PRICE_MIN_CENTS": 5},
        "rationale": "test",
        "success_criteria_snapshot": {},
        "git_patch": "",
        "codex_review": {"verdict": "approve", "reasoning": "ok"},
        "human_decision": None,
        "applied_at": None,
        "applied_commit": None,
    }))
    return p


def test_api_agents_lists_weather_strategist():
    r = client.get("/api/agents")
    assert r.status_code == 200
    names = [a["name"] for a in r.json()["agents"]]
    assert "weather-strategist" in names


def test_api_agent_config_returns_writable_keys():
    r = client.get("/api/agents/weather-strategist/config")
    assert r.status_code == 200
    body = r.json()
    assert "current_config" in body
    assert "writable_config_keys" in body


def test_api_agent_config_unknown_404():
    r = client.get("/api/agents/no-such-agent/config")
    assert r.status_code == 404


def test_api_proposals_list_and_get(tmp_path, monkeypatch):
    import dashboard.app as da
    monkeypatch.setattr(da, "_AGENT_PROPOSAL_ROOT", tmp_path)
    _write_fake_proposal(tmp_path, "pendprop1", state="pending")
    r = client.get("/api/proposals?state=pending")
    assert r.status_code == 200
    items = r.json()["proposals"]
    assert any(i["id"] == "pendprop1" for i in items)

    r = client.get("/api/proposals/pendprop1")
    assert r.status_code == 200
    assert r.json()["id"] == "pendprop1"

    r = client.get("/api/proposals/nope")
    assert r.status_code == 404


def test_api_proposals_approve_moves_file(tmp_path, monkeypatch):
    import dashboard.app as da
    monkeypatch.setattr(da, "_AGENT_PROPOSAL_ROOT", tmp_path)
    _write_fake_proposal(tmp_path, "appr1", state="pending")
    r = client.post("/api/proposals/appr1/approve")
    assert r.status_code == 200
    assert (tmp_path / "approved" / "appr1.json").exists()
    assert not (tmp_path / "pending" / "appr1.json").exists()


def test_api_proposals_reject_moves_file(tmp_path, monkeypatch):
    import dashboard.app as da
    monkeypatch.setattr(da, "_AGENT_PROPOSAL_ROOT", tmp_path)
    _write_fake_proposal(tmp_path, "rej1", state="pending")
    r = client.post("/api/proposals/rej1/reject?reason=bad")
    assert r.status_code == 200
    assert (tmp_path / "rejected" / "rej1.json").exists()


def test_api_proposals_invalid_state():
    r = client.get("/api/proposals?state=bogus")
    assert r.status_code == 400
