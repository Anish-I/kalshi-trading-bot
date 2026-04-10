"""Tests for scripts.agents.promote — offline, uses tmp proposal root."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.agents import promote as pr


def _write_proposal(root: Path, pid: str, patch: str, state: str = "approved") -> Path:
    d = root / state
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{pid}.json"
    p.write_text(json.dumps({
        "id": pid,
        "state": state,
        "agent": "test-agent",
        "git_patch": patch,
        "writable_config_keys": [
            {"module": "scripts/nonexistent_example.py", "key": "X"},
        ],
    }))
    return p


def test_list_approved_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("KALSHI_AGENT_PROPOSAL_ROOT", str(tmp_path))
    assert pr.list_approved() == []


def test_list_approved_reads_files(tmp_path, monkeypatch):
    monkeypatch.setenv("KALSHI_AGENT_PROPOSAL_ROOT", str(tmp_path))
    _write_proposal(tmp_path, "p1", "")
    _write_proposal(tmp_path, "p2", "")
    ids = pr.list_approved()
    assert "p1" in ids and "p2" in ids


def test_main_list_prints_ids(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KALSHI_AGENT_PROPOSAL_ROOT", str(tmp_path))
    _write_proposal(tmp_path, "abc", "")
    rc = pr.main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "abc" in out


def test_missing_proposal_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("KALSHI_AGENT_PROPOSAL_ROOT", str(tmp_path))
    rc = pr.main(["--approved", "nope"])
    assert rc != 0


def test_empty_patch_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("KALSHI_AGENT_PROPOSAL_ROOT", str(tmp_path))
    _write_proposal(tmp_path, "empty", "")
    rc = pr.main(["--approved", "empty", "--dry-run"])
    assert rc != 0


def test_dry_run_calls_git_apply_check(tmp_path, monkeypatch):
    monkeypatch.setenv("KALSHI_AGENT_PROPOSAL_ROOT", str(tmp_path))
    # Use a minimally valid-looking patch; git apply --check will reject it,
    # which is fine — we only assert that the code path runs git apply --check.
    patch = (
        "--- a/does_not_exist.py\n"
        "+++ b/does_not_exist.py\n"
        "@@ -0,0 +1 @@\n"
        "+hello\n"
    )
    _write_proposal(tmp_path, "dryp", patch)

    called = {"count": 0, "check": False}
    real_run = subprocess.run

    def fake_run(cmd, *a, **kw):
        if isinstance(cmd, list) and cmd[:2] == ["git", "apply"]:
            called["count"] += 1
            if "--check" in cmd:
                called["check"] = True

            class FakeProc:
                returncode = 0
                stdout = ""
                stderr = ""
            return FakeProc()
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = pr.main(["--approved", "dryp", "--dry-run"])
    assert rc == 0
    assert called["check"] is True
