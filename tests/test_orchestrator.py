"""Tests for scripts.agents.orchestrator — offline only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.agents import orchestrator as orch


def test_tiny_yaml_parses_weather_strategist():
    path = Path(__file__).resolve().parents[1] / "agents" / "weather-strategist.yaml"
    cfg = orch._tiny_yaml(path.read_text(encoding="utf-8"))
    assert cfg["name"] == "weather-strategist"
    assert cfg["authority_class"] == "advisory"
    keys = cfg["writable_config_keys"]
    assert isinstance(keys, list) and len(keys) == 2
    assert keys[0]["key"] == "ENTRY_PRICE_MIN_CENTS"


def test_load_yaml_public_wrapper():
    path = Path(__file__).resolve().parents[1] / "agents" / "weather-strategist.yaml"
    cfg = orch._load_yaml(path)
    assert cfg.get("authority_class") == "advisory"


def test_mock_end_to_end_writes_pending_file(tmp_path, monkeypatch):
    monkeypatch.setenv("KALSHI_AGENT_PROPOSAL_ROOT", str(tmp_path / "proposals"))
    rc = orch.main([
        "--agent", "weather-strategist",
        "--mock-codex", "--mock-claude",
    ])
    assert rc == 0
    pending = tmp_path / "proposals" / "pending"
    files = sorted(pending.glob("*.json"))
    assert len(files) == 1
    obj = json.loads(files[0].read_text(encoding="utf-8"))
    # Required shape
    for k in (
        "id", "state", "agent", "agent_version", "created_at", "snapshot_hash",
        "current_config", "proposed_config", "rationale",
        "success_criteria_snapshot", "git_patch", "codex_review",
        "human_decision", "applied_at", "applied_commit",
    ):
        assert k in obj, f"missing key {k}"
    assert obj["state"] == "pending"
    assert obj["agent"] == "weather-strategist"
    assert obj["codex_review"]["verdict"] == "approve"  # mock mode


def test_unknown_agent_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("KALSHI_AGENT_PROPOSAL_ROOT", str(tmp_path / "proposals"))
    rc = orch.main([
        "--agent", "no-such-agent",
        "--mock-codex", "--mock-claude",
    ])
    assert rc != 0


def test_load_current_config_reads_module_constants(tmp_path, monkeypatch):
    fake_mod = tmp_path / "fake_trade.py"
    fake_mod.write_text("FOO_CENTS = 42\nBAR = 1.5\n")
    monkeypatch.setattr(orch, "REPO_ROOT", tmp_path)
    out = orch._load_current_config([
        {"module": "fake_trade.py", "key": "FOO_CENTS", "default": 0},
        {"module": "fake_trade.py", "key": "MISSING", "default": 99},
    ])
    assert out["FOO_CENTS"] == 42
    assert out["MISSING"] == 99  # falls back to default
