"""Headless runner that invokes `claude -p` to process a snapshot.

This is the `claude -p` alternative to `run_session.py` (which targets the
Managed Agents beta API). It uses the existing Claude Code OAuth auth via the
`claude` CLI, so no ANTHROPIC_API_KEY is required — billing goes against the
Claude Max subscription.

Why this exists: the daily PnL reporter MVP doesn't actually need a cloud
container. It's a 50KB parquet → 800 words of markdown transformation. A
local subprocess is strictly better than a cloud session for this job.

Tradeoffs vs Managed Agents:
  - No cloud isolation (fine, data is local anyway)
  - No SSE event stream (don't need it for batch reports)
  - No `beta.agents.create` versioning (YAML still documents intent)
  + Zero API key, zero per-session cost
  + Zero new infrastructure
  + Runs in ~30s end-to-end

Usage:
    from scripts.agents.run_claude_headless import run_headless_reporter
    code = run_headless_reporter(tarball_path, output_dir, timeout_s=600)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = REPO_ROOT / "agents"
HEADLESS_SYSTEM_PROMPT = AGENTS_DIR / "kalshi-reporter.headless.system.md"
USAGE_PATH = AGENTS_DIR / ".usage.jsonl"

# Hard walls
DEFAULT_TIMEOUT_S = 600          # 10 min wall clock
DEFAULT_MAX_BUDGET_USD = 2.0     # subscription-backed but belt-and-suspenders
DEFAULT_MODEL = "sonnet"         # claude -p resolves alias to current Sonnet


def _extract_tarball(tarball: Path, dest: Path) -> None:
    """Extract tarball into dest (must exist)."""
    with tarfile.open(tarball, mode="r:gz") as tf:
        tf.extractall(dest)


def _log_usage(event: dict) -> None:
    """Append a telemetry line to agents/.usage.jsonl."""
    USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USAGE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def run_headless_reporter(
    tarball_path: Path,
    output_dir: Path,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_budget_usd: float = DEFAULT_MAX_BUDGET_USD,
    model: str = DEFAULT_MODEL,
) -> int:
    """Extract the snapshot, invoke `claude -p`, copy outputs, return exit code.

    The claude CLI uses existing Claude Code OAuth (no ANTHROPIC_API_KEY).
    Writes results directly to output_dir. Returns 0 on success, non-zero on
    any failure (subprocess timeout, missing report.md, CLI not found, etc).
    """
    started = datetime.now(timezone.utc)
    tarball_path = Path(tarball_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tarball_path.exists():
        print(f"ERROR: tarball not found: {tarball_path}", file=sys.stderr)
        return 3

    if not HEADLESS_SYSTEM_PROMPT.exists():
        print(
            f"ERROR: system prompt missing: {HEADLESS_SYSTEM_PROMPT}",
            file=sys.stderr,
        )
        return 4

    claude_bin = shutil.which("claude")
    if not claude_bin:
        print("ERROR: `claude` CLI not on PATH", file=sys.stderr)
        return 5

    # Extract snapshot to a working dir we own
    with tempfile.TemporaryDirectory(prefix="kalshi-report-") as work_root:
        work = Path(work_root).resolve()
        staging = work / "inputs"
        staging.mkdir()
        try:
            _extract_tarball(tarball_path, staging)
        except Exception as exc:
            print(f"ERROR: failed to extract tarball: {exc}", file=sys.stderr)
            return 6

        report_path = output_dir / "report.md"
        stats_path = output_dir / "stats.json"
        system_prompt_text = HEADLESS_SYSTEM_PROMPT.read_text(encoding="utf-8")

        user_prompt = (
            f"Generate the daily PnL + shadow-mode report per your system prompt.\n\n"
            f"INPUT_DIR: {staging}\n"
            f"OUTPUT_REPORT: {report_path}\n"
            f"OUTPUT_STATS: {stats_path}\n\n"
            f"Read every file under INPUT_DIR. Write the full markdown report to "
            f"OUTPUT_REPORT and the machine-readable summary to OUTPUT_STATS. "
            f"When done, print the confirmation line and stop."
        )

        cmd = [
            claude_bin,
            "-p",
            "--permission-mode", "acceptEdits",
            "--model", model,
            "--allowedTools",
            "Read,Write,Bash",
            "--add-dir", str(staging),
            "--add-dir", str(output_dir),
            "--append-system-prompt", system_prompt_text,
            "--max-budget-usd", str(max_budget_usd),
            "--no-session-persistence",
            "--output-format", "json",
            user_prompt,
        ]

        print(
            f"[run_claude_headless] invoking claude -p "
            f"(timeout={timeout_s}s, budget=${max_budget_usd}, model={model})",
            file=sys.stderr,
        )

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                # Inherit parent env so claude CLI finds its creds
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            print(f"ERROR: claude -p timed out after {elapsed:.0f}s", file=sys.stderr)
            _log_usage(
                {
                    "timestamp": started.isoformat(),
                    "runner": "claude_headless",
                    "agent": "kalshi-reporter",
                    "status": "timeout",
                    "elapsed_s": elapsed,
                    "exit_code": 124,
                }
            )
            return 124

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()

        # Try to parse JSON output for telemetry; don't hard-fail if it's free text
        input_tokens = output_tokens = None
        cost_usd = None
        try:
            # When --output-format=json, stdout is a single JSON object
            result = json.loads(proc.stdout)
            if isinstance(result, dict):
                usage = result.get("usage") or {}
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
                cost_usd = result.get("total_cost_usd") or result.get("cost_usd")
        except Exception:
            pass

        _log_usage(
            {
                "timestamp": started.isoformat(),
                "runner": "claude_headless",
                "agent": "kalshi-reporter",
                "status": "exit_" + str(proc.returncode),
                "elapsed_s": elapsed,
                "exit_code": proc.returncode,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
            }
        )

        if proc.returncode != 0:
            print(
                f"ERROR: claude -p exit {proc.returncode}",
                file=sys.stderr,
            )
            if proc.stderr:
                print(proc.stderr[-2000:], file=sys.stderr)
            return proc.returncode

        if not report_path.exists():
            print(
                f"ERROR: claude exited 0 but report not written at {report_path}",
                file=sys.stderr,
            )
            if proc.stdout:
                print("stdout tail:", proc.stdout[-1500:], file=sys.stderr)
            return 7

        # Success path — surface the confirmation line
        print(
            f"[run_claude_headless] wrote {report_path} "
            f"({report_path.stat().st_size} bytes) in {elapsed:.1f}s",
            file=sys.stderr,
        )
        return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tarball", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--budget-usd", type=float, default=DEFAULT_MAX_BUDGET_USD)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    return run_headless_reporter(
        args.tarball,
        args.out,
        timeout_s=args.timeout,
        max_budget_usd=args.budget_usd,
        model=args.model,
    )


if __name__ == "__main__":
    sys.exit(main())
