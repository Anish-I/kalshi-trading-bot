#!/usr/bin/env python3
"""Build an offline calibration artifact for the crypto XGB+Momentum conjunction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from config.settings import settings
from engine.crypto_calibration import (
    CalibrationEvent,
    build_calibration_artifact,
    make_honest_labels,
    save_artifact,
    session_tag_for_timestamp,
)
from features.honest_features import compute_honest_features


DEFAULT_DATA_FILE = Path("D:/kalshi-data/binance_history/klines_90d.parquet")
DEFAULT_MODEL_PATH = Path("D:/kalshi-models/latest_model.json")


def load_schema(schema_path: str | Path) -> dict:
    raw = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {"features": raw}
    return raw


def _vectorized_xgb_votes(featured: pd.DataFrame, model_path: str | Path, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    model_df = featured.copy()
    for col in feature_cols:
        if col not in model_df.columns:
            model_df[col] = 0.0

    x = model_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    proba = booster.predict(xgb.DMatrix(x))

    p_down = proba[:, 0].astype(float)
    p_up = proba[:, 2].astype(float)
    denom = p_up + p_down

    p_binary_up = np.divide(p_up, denom, out=np.full_like(p_up, 0.5), where=denom >= 0.01)
    p_binary_down = np.divide(p_down, denom, out=np.full_like(p_down, 0.5), where=denom >= 0.01)

    votes = np.full(len(model_df), "flat", dtype=object)
    confs = np.full(len(model_df), 0.50, dtype=float)

    up_mask = (denom >= 0.01) & (p_binary_up > 0.52)
    down_mask = (denom >= 0.01) & (p_binary_down > 0.52) & ~up_mask

    votes[up_mask] = "up"
    confs[up_mask] = p_binary_up[up_mask]
    votes[down_mask] = "down"
    confs[down_mask] = p_binary_down[down_mask]

    return votes, confs


def _vectorized_momentum_votes(featured: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    m5 = featured.get("ret_5m", 0).fillna(0)
    m10 = featured.get("ret_10m", 0).fillna(0)
    vwap_dev = featured.get("vwap_deviation", 0).fillna(0)
    donch = featured.get("donchian_position", 0.5).fillna(0.5)
    ema_s = featured.get("ema_9_slope", 0).fillna(0)
    vol_ratio = featured.get("volume_sma_ratio", 1.0).fillna(1.0)

    signals = np.zeros(len(featured), dtype=int)
    signals += np.where(m5 > 0.0007, 1, np.where(m5 < -0.0007, -1, 0))
    signals += np.where(m10 > 0.0012, 1, np.where(m10 < -0.0012, -1, 0))
    signals += np.where(vwap_dev > 0.002, 1, np.where(vwap_dev < -0.002, -1, 0))
    signals += np.where(donch > 0.75, 1, np.where(donch < 0.25, -1, 0))
    signals += np.where(ema_s > 0.00012, 1, np.where(ema_s < -0.00012, -1, 0))

    amplify = (vol_ratio > 2.0) & (np.abs(signals) >= 2)
    signals = np.where(amplify, np.trunc(signals * 1.5).astype(int), signals)

    votes = np.full(len(featured), "flat", dtype=object)
    confs = np.full(len(featured), 0.50, dtype=float)

    up_mask = signals >= 3
    down_mask = signals <= -3
    votes[up_mask] = "up"
    votes[down_mask] = "down"
    confs[up_mask] = np.minimum(0.80, 0.55 + signals[up_mask] * 0.05)
    confs[down_mask] = np.minimum(0.80, 0.55 + np.abs(signals[down_mask]) * 0.05)

    return votes, confs


def _make_events(
    featured: pd.DataFrame,
    xgb_votes: np.ndarray,
    xgb_confs: np.ndarray,
    momentum_votes: np.ndarray,
    momentum_confs: np.ndarray,
    *,
    price_source: str,
) -> list[CalibrationEvent]:
    events: list[CalibrationEvent] = []
    conjunction_mask = (xgb_votes == momentum_votes) & (xgb_votes != "flat")

    for idx in np.flatnonzero(conjunction_mask):
        side = str(xgb_votes[idx])
        label = int(featured.iloc[idx]["label"])
        won = (side == "up" and label == 2) or (side == "down" and label == 0)

        timestamp = pd.Timestamp(featured.iloc[idx]["timestamp"]).to_pydatetime()
        if price_source == "hypothetical":
            entry_prices = range(20, 61, 5)
        elif price_source == "close":
            entry_prices = [max(1, min(99, int(round(float(featured.iloc[idx]["close"])) % 100)))]
        else:
            proxy = (float(xgb_confs[idx]) + float(momentum_confs[idx])) / 2.0
            entry_prices = [max(1, min(99, int(round(proxy * 100.0))))]

        for entry_price_cents in entry_prices:
            pnl_cents = (100 - entry_price_cents) if won else -entry_price_cents
            events.append(
                CalibrationEvent(
                    timestamp=timestamp.isoformat(),
                    session_tag=session_tag_for_timestamp(timestamp),
                    side=side,
                    entry_price_cents=entry_price_cents,
                    won=won,
                    pnl_cents=int(pnl_cents),
                    xgb_vote=side.upper(),
                    xgb_conf=float(xgb_confs[idx]),
                    momentum_vote=side.upper(),
                    momentum_conf=float(momentum_confs[idx]),
                    label=label,
                )
            )

    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=Path(settings.CRYPTO_CALIBRATION_PATH))
    parser.add_argument("--price-source", choices=["hypothetical", "proxy", "close"], default="hypothetical")
    parser.add_argument("--forward-bars", type=int, default=15)
    parser.add_argument("--threshold-bps", type=float, default=0.001)
    parser.add_argument("--min-trades", type=int, default=int(settings.CRYPTO_CALIBRATION_MIN_TRADES))
    parser.add_argument("--ev-buffer-cents", type=float, default=float(settings.CRYPTO_EV_BUFFER_CENTS))
    parser.add_argument("--min-net-ev-cents", type=float, default=float(settings.CRYPTO_MIN_NET_EV_CENTS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data_file.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    schema_path = args.model_path.with_suffix(".schema.json")
    schema = load_schema(schema_path)
    feature_cols = list(schema.get("features", []))
    if not feature_cols:
        raise ValueError(f"Schema missing features: {schema_path}")

    raw = pd.read_parquet(args.data_file)
    featured = compute_honest_features(raw)
    featured = featured.copy()
    featured["label"] = make_honest_labels(featured["close"], forward_bars=args.forward_bars, threshold_bps=args.threshold_bps)
    featured = featured.dropna(subset=["label"]).reset_index(drop=True)
    featured = featured.iloc[:-args.forward_bars].reset_index(drop=True)
    featured["label"] = featured["label"].astype(int)

    xgb_votes, xgb_confs = _vectorized_xgb_votes(featured, args.model_path, feature_cols)
    momentum_votes, momentum_confs = _vectorized_momentum_votes(featured)

    events = _make_events(
        featured,
        xgb_votes,
        xgb_confs,
        momentum_votes,
        momentum_confs,
        price_source=args.price_source,
    )

    artifact = build_calibration_artifact(
        events,
        metadata={
            "data_file": str(args.data_file),
            "model_path": str(args.model_path),
            "schema_path": str(schema_path),
            "source_model_type": schema.get("model_type", ""),
            "source_model_timestamp": schema.get("timestamp", ""),
            "training_samples": schema.get("training_samples", 0),
            "cv_accuracy": schema.get("cv_accuracy"),
            "cv_log_loss": schema.get("cv_log_loss"),
            "schema_feature_count": len(feature_cols),
            "source_mode": "bars_90d_replay",
            "price_source": args.price_source,
            "forward_bars": args.forward_bars,
            "threshold_bps": args.threshold_bps,
            "signal_rule": "xgb+momentum_conjunction",
            "features_source": "features.honest_features.compute_honest_features",
            "source_rows": len(featured),
            "conjunction_events": len(events),
        },
        min_trades=args.min_trades,
        ev_buffer_cents=args.ev_buffer_cents,
        min_net_ev_cents=args.min_net_ev_cents,
    )

    out = save_artifact(artifact, args.output)
    summary = artifact["summary"]
    print(f"Wrote calibration artifact: {out}")
    print(f"Conjunction events: {len(events)}")
    print(f"Tradable buckets: {summary['tradable_bucket_count']}")
    best = summary["best_bucket"]
    worst = summary["worst_bucket"]
    print(
        "Best: {side}@{bucket}c net={net:+.2f} gross={gross:+.2f}".format(
            side=best["side"],
            bucket=best["price_bucket"],
            net=best["net_ev_cents_per_contract"],
            gross=best["gross_ev_cents_per_contract"],
        )
    )
    print(
        "Worst: {side}@{bucket}c net={net:+.2f} gross={gross:+.2f}".format(
            side=worst["side"],
            bucket=worst["price_bucket"],
            net=worst["net_ev_cents_per_contract"],
            gross=worst["gross_ev_cents_per_contract"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
