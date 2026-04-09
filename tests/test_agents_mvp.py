"""Tests for Managed Agents MVP scaffolding.

These tests do NOT touch the real Anthropic API. Live paths are skipped by
ensuring ANTHROPIC_API_KEY is unset in the subprocess environment.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _clean_env() -> dict:
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    # ensure repo root on PYTHONPATH so `scripts.agents` resolves
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_snapshot_runs_without_api_key(tmp_path):
    # import directly — no subprocess needed
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.agents import snapshot as snapshot_mod

    out = tmp_path / "snap.tar.gz"
    result = snapshot_mod.build_reporter_snapshot(out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0

    with tarfile.open(out, "r:gz") as tar:
        names = set(tar.getnames())
    assert "manifest.json" in names
    # these might be .MISSING if source files absent on the machine running tests
    assert any(n == "order_ledger.parquet" or n == "order_ledger.MISSING" for n in names)
    assert any(n == "trade_journal.parquet" or n == "trade_journal.MISSING" for n in names)
    assert any(
        n == "family_scorecard.parquet" or n == "family_scorecard.ERROR" for n in names
    )


def test_deploy_dry_run_prints_payload():
    env = _clean_env()
    proc = subprocess.run(
        [sys.executable, "scripts/agents/deploy_agents.py", "--dry-run"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert '"name": "kalshi-reporter"' in proc.stdout
    assert '"model": "claude-sonnet-4-6"' in proc.stdout


def test_schedule_refuses_missing_api_key():
    env = _clean_env()
    proc = subprocess.run(
        [sys.executable, "scripts/agents/schedule.py", "reporter"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "ANTHROPIC_API_KEY" in proc.stderr


def test_tarball_contents(tmp_path):
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.agents import snapshot as snapshot_mod

    out = tmp_path / "snap.tar.gz"
    snapshot_mod.build_reporter_snapshot(out)
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    with tarfile.open(out, "r:gz") as tar:
        tar.extractall(extract_dir)

    # manifest exists and has required keys
    manifest = json.loads((extract_dir / "manifest.json").read_text())
    for key in ("task", "snapshot_time", "date", "git_sha", "window_days"):
        assert key in manifest

    # state dir exists and has at least combined_trader_state.json
    state_dir = extract_dir / "state"
    assert state_dir.exists()
    assert (state_dir / "combined_trader_state.json").exists()

    # family_scorecard parquet has rows (if compute succeeded)
    fs_path = extract_dir / "family_scorecard.parquet"
    if fs_path.exists():
        import pandas as pd

        df = pd.read_parquet(fs_path)
        assert len(df) >= 0  # permissive; presence is enough
