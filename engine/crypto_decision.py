"""Calibration-aware crypto decision helpers."""

from __future__ import annotations

import json
from pathlib import Path

from engine.crypto_calibration import bucket_price_cents, normalize_side


DEFAULT_BUCKET_SIZE_CENTS = 5


def bucket_floor_cents(price_cents: int, bucket_size_cents: int = DEFAULT_BUCKET_SIZE_CENTS) -> int:
    """Return the lower bound for a bucket of size N cents."""
    if price_cents < 0:
        raise ValueError("price_cents must be >= 0")
    return (price_cents // bucket_size_cents) * bucket_size_cents


def price_bucket_label(price_cents: int, bucket_size_cents: int = DEFAULT_BUCKET_SIZE_CENTS) -> str:
    """Return a human-readable bucket label like ``45c``."""
    return f"{bucket_price_cents(price_cents, bucket_size_cents)}c"


def gross_ev_cents_per_contract(p_win: float, entry_cents: int) -> float:
    """Binary-contract expected value in cents before execution buffer."""
    return p_win * (100 - entry_cents) - (1.0 - p_win) * entry_cents


def load_crypto_calibration(path: str | Path) -> dict:
    """Load the calibration artifact and build a side/bucket index."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        return {
            "path": str(artifact_path),
            "exists": False,
            "version": "",
            "generated_at": "",
            "summary": {},
            "rows": [],
            "rows_by_key": {},
        }

    raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    rows = raw.get("rows", raw.get("cells", [])) or []
    rows_by_key = {}

    for row in rows:
        raw_side = str(row.get("side", "")).lower()
        try:
            side = normalize_side(raw_side)
        except ValueError:
            continue
        bucket = row.get("price_bucket")
        if not side or not bucket:
            continue
        rows_by_key[(side, bucket)] = row
        if side == "up":
            rows_by_key[("yes", bucket)] = row
        elif side == "down":
            rows_by_key[("no", bucket)] = row

    return {
        "path": str(artifact_path),
        "exists": True,
        "version": raw.get("version", raw.get("calibration_version", raw.get("metadata", {}).get("artifact_id", ""))),
        "generated_at": raw.get("generated_at", raw.get("timestamp", raw.get("metadata", {}).get("generated_at", ""))),
        "summary": raw.get("summary", {}),
        "bucket_size_cents": int(
            raw.get(
                "bucket_size_cents",
                raw.get("metadata", {}).get("bucket_width_cents", DEFAULT_BUCKET_SIZE_CENTS),
            )
        ),
        "rows": rows,
        "rows_by_key": rows_by_key,
        "metadata": raw,
    }


def lookup_calibration_row(
    artifact: dict,
    side: str,
    entry_cents: int,
    min_trades: int,
) -> dict | None:
    """Return a calibration row if the bucket exists and clears support."""
    if not artifact.get("exists"):
        return None

    bucket_size = artifact.get("bucket_size_cents", DEFAULT_BUCKET_SIZE_CENTS)
    bucket_center = bucket_price_cents(entry_cents, bucket_size)
    bucket_label = price_bucket_label(entry_cents, bucket_size)
    row = artifact.get("rows_by_key", {}).get((side.lower(), bucket_center))
    if not row:
        row = artifact.get("rows_by_key", {}).get((side.lower(), bucket_label))
    if not row:
        return None

    n_trades = int(row.get("n_trades", 0) or 0)
    tradable = bool(row.get("tradable", False))
    if n_trades < min_trades or not tradable:
        return None

    return row


def evaluate_calibrated_trade(
    artifact: dict,
    side: str,
    entry_cents: int,
    min_trades: int,
    ev_buffer_cents: float,
    min_net_ev_cents: float,
) -> dict:
    """Evaluate a side/price candidate against the calibration artifact."""
    bucket = price_bucket_label(entry_cents, artifact.get("bucket_size_cents", DEFAULT_BUCKET_SIZE_CENTS))
    row = lookup_calibration_row(artifact, side, entry_cents, min_trades)
    if not row:
        return {
            "status": "no_calibration",
            "price_bucket": bucket,
            "calibrated_p_win": None,
            "gross_ev_cents_per_contract": None,
            "net_ev_cents_per_contract": None,
            "bucket_trade_count": 0,
            "bucket_tradable": False,
        }

    p_win = float(row.get("win_rate", 0.0))
    gross_ev = gross_ev_cents_per_contract(p_win, entry_cents)
    net_ev = gross_ev - float(ev_buffer_cents)
    return {
        "status": "trading" if net_ev >= float(min_net_ev_cents) else "no_edge",
        "price_bucket": bucket,
        "calibrated_p_win": p_win,
        "gross_ev_cents_per_contract": gross_ev,
        "net_ev_cents_per_contract": net_ev,
        "bucket_trade_count": int(row.get("n_trades", 0) or 0),
        "bucket_tradable": bool(row.get("tradable", False)),
        "row": row,
    }


def calibration_summary_view(artifact: dict) -> dict:
    """Return a compact summary for dashboard/log state."""
    if not artifact.get("exists"):
        return {
            "exists": False,
            "version": "",
            "generated_at": "",
            "tradable_buckets": 0,
            "best_bucket": None,
            "worst_bucket": None,
        }

    summary = artifact.get("summary", {}) or {}
    return {
        "exists": True,
        "version": artifact.get("version", ""),
        "generated_at": artifact.get("generated_at", ""),
        "tradable_buckets": summary.get(
            "tradable_buckets",
            summary.get(
                "tradable_bucket_count",
            sum(1 for row in artifact.get("rows", []) if row.get("tradable")),
            ),
        ),
        "best_bucket": summary.get("best_bucket"),
        "worst_bucket": summary.get("worst_bucket"),
    }
