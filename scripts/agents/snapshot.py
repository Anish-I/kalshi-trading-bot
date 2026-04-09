"""Build input snapshot tarballs for Managed Agents.

Read-only: pulls data from D:/kalshi-data, never modifies it.
Current task: `reporter` — last 7 days of order_ledger + full trade_journal +
trader state JSONs + a fresh FamilyScorecard snapshot.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path("D:/kalshi-data")

STATE_FILES = [
    "combined_trader_state.json",
    "ml_trader_state.json",
    "pair_trader_state.json",
    "fed_trader_state.json",
    "weather_trader_state.json",
]


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def build_reporter_snapshot(out_path: Path, window_days: int = 7) -> Path:
    """Build the reporter task snapshot at ``out_path`` (tar.gz). Returns ``out_path``."""
    import pandas as pd  # lazy

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "snap"
        staging.mkdir()
        (staging / "state").mkdir()

        # 1) order_ledger — filter last N days
        ol_src = DATA_DIR / "order_ledger.parquet"
        if ol_src.exists():
            ol = pd.read_parquet(ol_src)
            if "created_at" in ol.columns and len(ol) > 0:
                ts = pd.to_datetime(ol["created_at"], utc=True, errors="coerce")
                cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
                mask = ts >= cutoff
                ol = ol.loc[mask].copy()
            ol.to_parquet(staging / "order_ledger.parquet", index=False)
        else:
            # write empty marker file so consumers can detect absence
            (staging / "order_ledger.MISSING").write_text("source file not found\n")

        # 2) trade_journal — copy as-is
        tj_src = DATA_DIR / "trade_journal.parquet"
        if tj_src.exists():
            shutil.copy2(tj_src, staging / "trade_journal.parquet")
        else:
            (staging / "trade_journal.MISSING").write_text("source file not found\n")

        # 3) state json files
        for name in STATE_FILES:
            src = DATA_DIR / name
            if src.exists():
                shutil.copy2(src, staging / "state" / name)

        # 4) family scorecard — compute fresh
        try:
            # Ensure repo root is importable
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            from engine.family_scorecard import FamilyScorecard  # type: ignore

            sc = FamilyScorecard(shadow_mode=True)
            healths = sc.compute()
            rows = []
            for family, h in healths.items():
                row = {
                    "family": getattr(h, "family", family),
                    "healthy": bool(getattr(h, "healthy", False)),
                    "score": float(getattr(h, "score", 0.0) or 0.0),
                    "throttle_multiplier": float(
                        getattr(h, "throttle_multiplier", 1.0) or 1.0
                    ),
                }
                metrics = getattr(h, "metrics", {}) or {}
                # flatten numeric metrics
                for k, v in metrics.items():
                    try:
                        row[f"metric_{k}"] = float(v)
                    except (TypeError, ValueError):
                        row[f"metric_{k}"] = str(v)
                rows.append(row)
            df = pd.DataFrame(rows)
            df.to_parquet(staging / "family_scorecard.parquet", index=False)
        except Exception as e:  # don't fail the whole snapshot
            (staging / "family_scorecard.ERROR").write_text(
                f"FamilyScorecard.compute() failed: {type(e).__name__}: {e}\n"
            )

        # 5) manifest
        now = datetime.now(timezone.utc)
        manifest = {
            "task": "reporter",
            "snapshot_time": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "git_sha": _git_sha(),
            "window_days": window_days,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # tar gz
        with tarfile.open(out_path, "w:gz") as tar:
            for p in sorted(staging.rglob("*")):
                tar.add(p, arcname=str(p.relative_to(staging)))

    return out_path


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["reporter"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-days", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    if args.task == "reporter":
        build_reporter_snapshot(out, window_days=args.window_days)
    size_mb = out.stat().st_size / (1024 * 1024)
    with tarfile.open(out, "r:gz") as tar:
        members = tar.getnames()
    print(f"wrote {out} ({size_mb:.2f} MB, {len(members)} members)")
    for m in members[:10]:
        print(f"  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
