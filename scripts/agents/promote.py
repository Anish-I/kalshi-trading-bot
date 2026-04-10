"""Apply an approved agent proposal to the live repo.

Usage:
    python scripts/agents/promote.py --approved <proposal_id>
    python scripts/agents/promote.py --list
    python scripts/agents/promote.py --dry-run --approved <id>

DOES NOT auto-git-commit. Leaves modified files staged for human review.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROPOSAL_ROOT = Path(
    os.environ.get("KALSHI_AGENT_PROPOSAL_ROOT", "D:/kalshi-data/agent_proposals")
)


def _proposal_root() -> Path:
    # Re-read env at call time so tests can monkeypatch.
    return Path(os.environ.get("KALSHI_AGENT_PROPOSAL_ROOT", str(DEFAULT_PROPOSAL_ROOT)))


def list_approved() -> list:
    root = _proposal_root() / "approved"
    if not root.exists():
        return []
    return sorted([p.stem for p in root.glob("*.json")])


def load_proposal(proposal_id: str) -> tuple[Path, dict]:
    path = _proposal_root() / "approved" / f"{proposal_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"approved proposal not found: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _git_apply(patch_text: str, check_only: bool, cwd: Path) -> tuple[int, str]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(patch_text)
        patch_path = tf.name
    try:
        cmd = ["git", "apply"]
        if check_only:
            cmd.append("--check")
        cmd.append(patch_path)
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
        return proc.returncode, (proc.stderr or proc.stdout or "")
    finally:
        try:
            os.unlink(patch_path)
        except Exception:
            pass


def _affected_modules(proposal: dict) -> list:
    mods = []
    for k in proposal.get("writable_config_keys") or []:
        if isinstance(k, dict) and k.get("module"):
            mods.append(k["module"])
    return mods


def promote(
    proposal_id: str,
    *,
    dry_run: bool = False,
    run_tests: bool = True,
    cwd: Optional[Path] = None,
) -> int:
    cwd = cwd or REPO_ROOT
    try:
        path, proposal = load_proposal(proposal_id)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    patch = proposal.get("git_patch", "")
    if not patch.strip():
        print("ERROR: proposal has no git_patch", file=sys.stderr)
        return 3

    rc, msg = _git_apply(patch, check_only=True, cwd=cwd)
    if rc != 0:
        print(f"ERROR: git apply --check failed:\n{msg}", file=sys.stderr)
        return 4

    if dry_run:
        print("[promote] dry-run OK. Patch preview:")
        print(patch)
        return 0

    rc, msg = _git_apply(patch, check_only=False, cwd=cwd)
    if rc != 0:
        print(f"ERROR: git apply failed:\n{msg}", file=sys.stderr)
        return 5

    affected = _affected_modules(proposal)
    if run_tests and affected:
        test_filter = " or ".join(Path(m).stem for m in affected)
        print(f"[promote] running targeted tests: {test_filter}")
        tp = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-q", "-k", test_filter],
            cwd=str(cwd),
        )
        if tp.returncode != 0:
            print("[promote] tests failed — reverting patch", file=sys.stderr)
            for m in affected:
                subprocess.run(["git", "restore", m], cwd=str(cwd))
            return 6

    # Move approved -> applied
    applied_dir = _proposal_root() / "applied"
    applied_dir.mkdir(parents=True, exist_ok=True)
    proposal["state"] = "applied"
    proposal["applied_at"] = datetime.now(timezone.utc).isoformat()
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd)
        ).decode().strip()
    except Exception:
        sha = "unknown"
    proposal["applied_commit"] = sha

    applied_path = applied_dir / f"{proposal_id}.json"
    applied_path.write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    try:
        path.unlink()
    except Exception:
        pass

    print(f"[promote] applied {proposal_id}. Affected modules:")
    for m in affected:
        print(f"  - {m}")
    print("[promote] NOTE: trader processes using those modules must be restarted.")
    print("[promote] Changes are staged but NOT committed. Review and commit manually.")
    return 0


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--approved", help="proposal id to apply")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-tests", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        items = list_approved()
        if not items:
            print("(no approved proposals)")
            return 0
        for i in items:
            print(i)
        return 0

    if not args.approved:
        ap.error("must pass --approved <id> or --list")

    return promote(
        args.approved,
        dry_run=args.dry_run,
        run_tests=not args.no_tests,
    )


if __name__ == "__main__":
    sys.exit(main())
