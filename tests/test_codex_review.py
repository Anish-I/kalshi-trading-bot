"""Tests for scripts.agents.codex_review — no real codex invocation."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest import mock

from scripts.agents import codex_review as cr


def test_mock_mode_returns_approve():
    v = cr.review(
        agent_name="weather-strategist",
        proposed_config={"X": 1},
        current_config={"X": 0},
        rationale="because",
        writable_keys=[{"module": "scripts/x.py", "key": "X"}],
        stats_snapshot={},
        mock=True,
    )
    assert v.verdict == "approve"
    assert "MOCK" in v.reasoning
    assert v.model == "mock"
    d = v.to_json()
    assert d["verdict"] == "approve"


def test_build_prompt_includes_all_fields():
    prompt = cr._build_review_prompt(
        agent_name="a1",
        proposed_config={"K": 2},
        current_config={"K": 1},
        rationale="rat",
        writable_keys=[{"module": "m.py", "key": "K"}],
        stats_snapshot={"s": 3},
    )
    assert "a1" in prompt
    assert "\"K\": 2" in prompt
    assert "\"K\": 1" in prompt
    assert "rat" in prompt
    assert "m.py" in prompt
    assert "\"s\": 3" in prompt


def test_schema_has_additional_properties_false():
    # Hard requirement of codex --output-schema
    assert cr.VERDICT_SCHEMA["additionalProperties"] is False
    assert cr.VERDICT_SCHEMA["properties"]["suggested_revision"]["additionalProperties"] is False


def test_review_handles_missing_output_file(tmp_path, monkeypatch):
    """Mock subprocess.run to return 0 but create no output file."""

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, timeout):
        # intentionally do not create the --output-last-message file
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    v = cr.review(
        agent_name="a",
        proposed_config={},
        current_config={},
        rationale="r",
        writable_keys=[],
        stats_snapshot={},
        mock=False,
    )
    assert v.verdict == "review_failed"
    assert "exit 0" in v.reasoning or "empty" in v.reasoning or "not JSON" in v.reasoning


def test_review_handles_timeout(monkeypatch):
    def fake_run(cmd, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(subprocess, "run", fake_run)
    v = cr.review(
        agent_name="a",
        proposed_config={},
        current_config={},
        rationale="r",
        writable_keys=[],
        stats_snapshot={},
        mock=False,
        timeout_s=5,
    )
    assert v.verdict == "review_failed"
    assert "timed out" in v.reasoning


def test_review_parses_valid_output(tmp_path, monkeypatch):
    """Simulate codex writing a valid verdict JSON."""
    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, timeout):
        # Find --output-last-message flag and write a valid verdict
        idx = cmd.index("--output-last-message")
        out = Path(cmd[idx + 1])
        out.write_text(json.dumps({"verdict": "reject", "reasoning": "nope"}))
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    v = cr.review(
        agent_name="a",
        proposed_config={},
        current_config={},
        rationale="r",
        writable_keys=[],
        stats_snapshot={},
        mock=False,
    )
    assert v.verdict == "reject"
    assert v.reasoning == "nope"
    assert "--json" in captured["cmd"]
    assert "--sandbox" in captured["cmd"]
