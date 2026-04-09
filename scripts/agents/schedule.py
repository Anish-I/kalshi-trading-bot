"""Scheduler entrypoint for Managed Agents tasks.

Usage:
    python scripts/agents/schedule.py reporter
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = REPO_ROOT / "agents"
USAGE_PATH = AGENTS_DIR / ".usage.jsonl"
HEARTBEAT_PATH = AGENTS_DIR / ".last_run.json"
OBSIDIAN_DIR = Path("C:/Users/ivatu/ObsidianVault/Wingman/Kalshi/Daily Reports")

DAILY_TOKEN_CAP = 500_000

# make scripts.agents importable
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agents import snapshot as snapshot_mod  # noqa: E402
from scripts.agents import run_session  # noqa: E402


def _todays_tokens() -> int:
    if not USAGE_PATH.exists():
        return 0
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total = 0
    try:
        for line in USAGE_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            ts = row.get("timestamp", "")
            if ts.startswith(today):
                total += int(row.get("input_tokens") or 0)
                total += int(row.get("output_tokens") or 0)
    except Exception:
        return 0
    return total


def _write_heartbeat(task: str, status: str, exit_code: int) -> None:
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_PATH.write_text(
        json.dumps(
            {
                "task": task,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "exit_code": exit_code,
            },
            indent=2,
        )
    )


def run_reporter() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY not set; cannot run scheduled reporter",
            file=sys.stderr,
        )
        _write_heartbeat("reporter", "missing_api_key", 2)
        return 2

    used = _todays_tokens()
    if used > DAILY_TOKEN_CAP:
        print(
            f"daily token cap exceeded ({used} > {DAILY_TOKEN_CAP})",
            file=sys.stderr,
        )
        _write_heartbeat("reporter", "budget_exceeded", 6)
        return 6

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tarball = tmp_path / "snap.tar.gz"
        snapshot_mod.build_reporter_snapshot(tarball)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        code = run_session.run("kalshi-reporter", tarball, out_dir)

        # Copy report.md / stats.json into Obsidian (if produced)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        OBSIDIAN_DIR.mkdir(parents=True, exist_ok=True)
        for candidate in out_dir.rglob("report.md"):
            shutil.copy2(candidate, OBSIDIAN_DIR / f"report_{today}.md")
            break
        for candidate in out_dir.rglob("stats.json"):
            shutil.copy2(candidate, OBSIDIAN_DIR / f"stats_{today}.json")
            break

    status = "ok" if code == 0 else f"exit_{code}"
    _write_heartbeat("reporter", status, code)
    return code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["reporter"])
    args = ap.parse_args()
    if args.task == "reporter":
        return run_reporter()
    return 1


if __name__ == "__main__":
    sys.exit(main())
