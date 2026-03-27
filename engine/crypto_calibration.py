"""Offline crypto calibration helpers.

This module stays isolated from the live trader. It only provides pure
helpers for building and querying a bucketed calibration artifact from
historical conjunction events.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PRICE_BUCKET_WIDTH_CENTS = 5
DEFAULT_FORWARD_BARS = 15
DEFAULT_THRESHOLD_BPS = 0.001


@dataclass(frozen=True)
class CalibrationEvent:
    timestamp: str
    session_tag: str
    side: str
    entry_price_cents: int
    won: bool
    pnl_cents: int
    xgb_vote: str
    xgb_conf: float
    momentum_vote: str
    momentum_conf: float
    label: int


def normalize_side(side: str) -> str:
    value = str(side).strip().lower()
    if value in {"up", "yes"}:
        return "up"
    if value in {"down", "no"}:
        return "down"
    raise ValueError(f"Unsupported side: {side!r}")


def bucket_price_cents(price_cents: float | int, bucket_width_cents: int = PRICE_BUCKET_WIDTH_CENTS) -> int:
    """Round an executable price to the nearest bucket center.

    The rounding is deterministic and resolves ties away from zero.
    """
    if bucket_width_cents <= 0:
        raise ValueError("bucket_width_cents must be positive")

    cents = int(math.floor(float(price_cents) + 0.5))
    bucket = int(math.floor((cents + (bucket_width_cents / 2.0)) / bucket_width_cents) * bucket_width_cents)
    return max(0, min(100, bucket))


def session_tag_for_timestamp(dt: datetime | pd.Timestamp | str) -> str:
    """Mirror the runtime session tagging rules for a historical timestamp."""
    if not isinstance(dt, datetime):
        dt = pd.Timestamp(dt).to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)

    if dt.weekday() >= 5:
        return "weekend"

    hour = dt.hour + dt.minute / 60.0
    if 14.5 <= hour < 21.0:
        return "us_core"
    return "us_off"


def make_honest_labels(
    close: pd.Series,
    forward_bars: int = DEFAULT_FORWARD_BARS,
    threshold_bps: float = DEFAULT_THRESHOLD_BPS,
) -> pd.Series:
    """Create the 3-class label used by the honest crypto model."""
    fwd_ret = np.log(close.shift(-forward_bars) / close)
    labels = pd.Series(
        np.where(
            fwd_ret > threshold_bps,
            2,
            np.where(fwd_ret < -threshold_bps, 0, 1),
        ),
        index=close.index,
    )
    return labels


def _coerce_events(events: Iterable[CalibrationEvent | dict[str, Any]]) -> list[CalibrationEvent]:
    coerced: list[CalibrationEvent] = []
    for event in events:
        if isinstance(event, CalibrationEvent):
            coerced.append(event)
            continue
        coerced.append(
            CalibrationEvent(
                timestamp=str(event["timestamp"]),
                session_tag=str(event["session_tag"]),
                side=normalize_side(event["side"]),
                entry_price_cents=int(event["entry_price_cents"]),
                won=bool(event["won"]),
                pnl_cents=int(round(float(event["pnl_cents"]))),
                xgb_vote=str(event.get("xgb_vote", "")).upper(),
                xgb_conf=float(event.get("xgb_conf", 0.0)),
                momentum_vote=str(event.get("momentum_vote", "")).upper(),
                momentum_conf=float(event.get("momentum_conf", 0.0)),
                label=int(event.get("label", 1)),
            )
        )
    return coerced


def _weighted_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_calibration_artifact(
    events: Iterable[CalibrationEvent | dict[str, Any]],
    *,
    metadata: dict[str, Any],
    bucket_width_cents: int = PRICE_BUCKET_WIDTH_CENTS,
    min_trades: int = 50,
    ev_buffer_cents: float = 2.0,
    min_net_ev_cents: float = 1.0,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Aggregate event-level results into a price-bucket calibration table."""
    events_list = _coerce_events(events)
    if not events_list:
        raise ValueError("No calibration events provided")

    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    total_events = len(events_list)

    grouped: dict[tuple[str, int], list[CalibrationEvent]] = {}
    for event in events_list:
        bucket = bucket_price_cents(event.entry_price_cents, bucket_width_cents=bucket_width_cents)
        grouped.setdefault((event.side, bucket), []).append(event)

    session_groups: dict[str, list[CalibrationEvent]] = {}
    for event in events_list:
        session_groups.setdefault(event.session_tag, []).append(event)

    cells: list[dict[str, Any]] = []
    for (side, bucket), bucket_events in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        n_trades = len(bucket_events)
        wins = sum(1 for e in bucket_events if e.won)
        win_rate = wins / n_trades if n_trades else 0.0
        avg_entry = _weighted_mean([float(e.entry_price_cents) for e in bucket_events])
        avg_pnl = _weighted_mean([float(e.pnl_cents) for e in bucket_events])
        gross_ev = win_rate * (100.0 - float(bucket)) - (1.0 - win_rate) * float(bucket)
        net_ev = gross_ev - float(ev_buffer_cents)
        coverage_pct = round(100.0 * n_trades / total_events, 4)
        tradable = n_trades >= min_trades and gross_ev > 0.0

        cells.append(
            {
                "side": side,
                "price_bucket": bucket,
                "n_trades": n_trades,
                "wins": wins,
                "losses": n_trades - wins,
                "win_rate": round(win_rate, 6),
                "avg_entry_price_cents": round(avg_entry, 4),
                "avg_pnl_cents": round(avg_pnl, 4),
                "gross_ev_cents_per_contract": round(gross_ev, 4),
                "net_ev_cents_per_contract": round(net_ev, 4),
                "coverage_pct": coverage_pct,
                "tradable": tradable,
            }
        )

    session_stats: dict[str, dict[str, Any]] = {}
    for session_tag, session_events in sorted(session_groups.items()):
        n_session = len(session_events)
        wins = sum(1 for e in session_events if e.won)
        total_pnl = sum(float(e.pnl_cents) for e in session_events)
        session_stats[session_tag] = {
            "n_trades": n_session,
            "wins": wins,
            "losses": n_session - wins,
            "win_rate": round(wins / n_session if n_session else 0.0, 6),
            "avg_pnl_cents": round(total_pnl / n_session if n_session else 0.0, 4),
            "coverage_pct": round(100.0 * n_session / total_events, 4),
        }

    tradable_cells = [cell for cell in cells if cell["tradable"]]
    summary_pool = tradable_cells or cells
    best_cell = max(
        summary_pool,
        key=lambda cell: (cell["net_ev_cents_per_contract"], cell["n_trades"], -cell["price_bucket"]),
    )
    worst_cell = min(
        summary_pool,
        key=lambda cell: (cell["net_ev_cents_per_contract"], -cell["n_trades"], cell["price_bucket"]),
    )

    core_metadata = {
        **metadata,
        "generated_at": generated_at,
        "artifact_type": "crypto_conjunction_calibration",
        "artifact_version": 1,
        "bucket_width_cents": bucket_width_cents,
        "min_trades": min_trades,
        "min_gross_ev_cents": 0.0,
        "min_net_ev_cents": min_net_ev_cents,
        "ev_buffer_cents": float(ev_buffer_cents),
        "total_events": total_events,
        "tradable_bucket_count": len(tradable_cells),
        "artifact_id": "",
    }

    artifact = {
        "metadata": core_metadata,
        "summary": {
            "total_events": total_events,
            "bucket_count": len(cells),
            "tradable_bucket_count": len(tradable_cells),
            "best_bucket": best_cell,
            "worst_bucket": worst_cell,
        },
        "session_stats": session_stats,
        "cells": cells,
    }
    for cell in artifact["cells"]:
        cell["tradable"] = bool(cell["tradable"])
    artifact["metadata"]["artifact_id"] = _stable_hash(
        {
            "metadata": {k: v for k, v in artifact["metadata"].items() if k != "generated_at"},
            "summary": artifact["summary"],
            "cells": artifact["cells"],
            "session_stats": artifact["session_stats"],
        }
    )
    return artifact


def lookup_bucket(artifact: dict[str, Any], side: str, price_cents: float | int) -> dict[str, Any] | None:
    """Return the calibration cell for a given side and executable price."""
    normalized_side = normalize_side(side)
    bucket = bucket_price_cents(price_cents, bucket_width_cents=int(artifact["metadata"]["bucket_width_cents"]))
    for cell in artifact.get("cells", []):
        if cell["side"] == normalized_side and cell["price_bucket"] == bucket:
            return cell
    return None


def save_artifact(artifact: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out
