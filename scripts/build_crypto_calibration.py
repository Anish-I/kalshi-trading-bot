#!/usr/bin/env python3
"""Build an offline calibration artifact for the crypto XGB+Momentum conjunction."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
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
from engine.market_archive import load_quotes_and_trades
from features.honest_features import compute_honest_features


DEFAULT_DATA_FILE = Path("D:/kalshi-data/binance_history/klines_90d.parquet")
DEFAULT_MODEL_PATH = Path("D:/kalshi-models/latest_model.json")
KALSHI_OUTPUT_PATH = Path("D:/kalshi-models/crypto_conjunction_calibration_kalshi.json")
DEFAULT_KALSHI_SERIES = [
    "KXBTC15M",
    "KXETH15M",
    "KXSOL15M",
    "KXXRP15M",
    "KXHYPE15M",
    "KXDOGE15M",
]

log = logging.getLogger("build_crypto_calibration")


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
    parser.add_argument(
        "--use-kalshi-archive",
        action="store_true",
        help="Build calibration from archived Kalshi quotes+trades instead of Binance replay.",
    )
    parser.add_argument(
        "--kalshi-series",
        type=str,
        default=",".join(DEFAULT_KALSHI_SERIES),
        help="Comma-separated list of Kalshi crypto series to include.",
    )
    parser.add_argument(
        "--kalshi-days",
        type=int,
        default=30,
        help="Lookback window in days when reading the Kalshi archive.",
    )
    return parser.parse_args()


def _fetch_settlement_results(tickers: list[str]) -> dict[str, str]:
    """Look up the canonical settlement ``result`` for each ticker via the Kalshi API.

    Returns ``{ticker: "yes"|"no"|""}``. Unsettled/unknown markets map to ``""``.
    This is authoritative ground truth from Kalshi (not a guess) — used only to
    determine wins in Kalshi-archive calibration mode.
    """
    try:
        from kalshi.client import KalshiClient
    except Exception:
        log.warning("KalshiClient unavailable; cannot fetch settlements")
        return {}
    client = KalshiClient()
    out: dict[str, str] = {}
    for t in tickers:
        try:
            m = client.get_market(t)
            out[t] = str(m.get("result", "") or "").lower()
        except Exception:
            log.warning("Failed to fetch settlement for %s", t, exc_info=False)
            out[t] = ""
    return out


def _build_from_kalshi_archive(args: argparse.Namespace) -> int:
    """Construct a calibration artifact from archived Kalshi quotes+trades.

    Win determination: each trade row has a ``ticker`` that references a Kalshi
    market. We fetch the market's canonical ``result`` field from the Kalshi API
    (``"yes"`` / ``"no"`` / ``""`` for unsettled). A YES-taker trade wins iff
    ``result == "yes"``; a NO-taker trade wins iff ``result == "no"``. Trades on
    unsettled markets are skipped.

    Bucketing: ``price_bucket = round(price_cents / bucket_size) * bucket_size``
    where ``bucket_size = args.bucket-size-equivalent`` (we reuse
    ``engine.crypto_calibration.bucket_price_cents`` via the same
    ``PRICE_BUCKET_WIDTH_CENTS = 5`` constant used by the binance path).

    Output schema is identical to the binance-mode artifact (``rows``,
    ``rows_by_key`` via loader, ``bucket_size_cents`` via ``metadata
    .bucket_width_cents``, ``version``) with added top-level metadata keys:
    ``data_source``, ``kalshi_archive_range``, ``kalshi_series``.
    """
    from engine.crypto_calibration import PRICE_BUCKET_WIDTH_CENTS, bucket_price_cents

    series_list = [s.strip() for s in args.kalshi_series.split(",") if s.strip()]
    end_d = date.today()
    start_d = end_d - timedelta(days=int(args.kalshi_days))

    log.info(
        "Kalshi-archive mode: series=%s range=%s..%s",
        series_list,
        start_d.isoformat(),
        end_d.isoformat(),
    )

    all_trades: list[pd.DataFrame] = []
    for series in series_list:
        _quotes, trades = load_quotes_and_trades(series, start_d, end_d)
        if trades is None or trades.empty:
            log.warning("no archived trades found for series %s", series)
            continue
        trades = trades.copy()
        trades["__series"] = series
        all_trades.append(trades)

    if not all_trades:
        log.warning(
            "No archived Kalshi trades found across any series — writing empty artifact."
        )
        artifact = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "artifact_type": "crypto_conjunction_calibration",
                "artifact_version": 1,
                "bucket_width_cents": PRICE_BUCKET_WIDTH_CENTS,
                "min_trades": int(args.min_trades),
                "ev_buffer_cents": float(args.ev_buffer_cents),
                "min_net_ev_cents": float(args.min_net_ev_cents),
                "total_events": 0,
                "tradable_bucket_count": 0,
                "artifact_id": "empty_kalshi_archive",
            },
            "summary": {
                "total_events": 0,
                "bucket_count": 0,
                "tradable_bucket_count": 0,
                "best_bucket": None,
                "worst_bucket": None,
            },
            "session_stats": {},
            "cells": [],
            "rows": [],
            "bucket_size_cents": PRICE_BUCKET_WIDTH_CENTS,
            "version": "empty_kalshi_archive",
            "data_source": "kalshi_archive",
            "kalshi_archive_range": f"{start_d.isoformat()} to {end_d.isoformat()}",
            "kalshi_series": series_list,
        }
        out_path = KALSHI_OUTPUT_PATH if args.output == Path(settings.CRYPTO_CALIBRATION_PATH) else args.output
        out = save_artifact(artifact, out_path)
        print(f"Wrote EMPTY calibration artifact (no archived trades): {out}")
        return 0

    df = pd.concat(all_trades, ignore_index=True)
    unique_tickers = sorted(df["ticker"].dropna().unique().tolist()) if "ticker" in df.columns else []
    log.info("Loaded %d trades across %d tickers", len(df), len(unique_tickers))

    settlements = _fetch_settlement_results(unique_tickers)
    df["__result"] = df["ticker"].map(settlements).fillna("")
    df = df[df["__result"].isin(["yes", "no"])].reset_index(drop=True)
    log.info("After settlement filter: %d trades on settled markets", len(df))

    if df.empty:
        log.warning("No trades on settled markets — writing empty artifact.")
        # fall-through to empty path
        return _build_from_kalshi_archive_empty(args, series_list, start_d, end_d)

    # Build CalibrationEvents: one per trade.
    # taker_side: "yes" trade at yes_price_cents => side="up" (bet yes), wins if result=="yes"
    #             "no" trade at no_price_cents => side="down" (bet no), wins if result=="no"
    events: list[CalibrationEvent] = []
    for row in df.itertuples(index=False):
        taker = str(getattr(row, "taker_side", "") or "").lower()
        if taker == "yes":
            side = "up"
            price = int(getattr(row, "yes_price_cents", 0) or 0)
        elif taker == "no":
            side = "down"
            price = int(getattr(row, "no_price_cents", 0) or 0)
        else:
            continue
        if price <= 0 or price >= 100:
            continue
        result = getattr(row, "_5", None)  # __result via itertuples — robust fallback below
        result = str(getattr(row, "__result", result) or "").lower()
        won = (side == "up" and result == "yes") or (side == "down" and result == "no")
        ts_raw = getattr(row, "created_time", None)
        try:
            ts = pd.Timestamp(ts_raw).tz_convert("UTC").to_pydatetime() if ts_raw else datetime.now(timezone.utc)
        except Exception:
            try:
                ts = pd.Timestamp(ts_raw, tz="UTC").to_pydatetime()
            except Exception:
                ts = datetime.now(timezone.utc)
        pnl = (100 - price) if won else -price
        events.append(
            CalibrationEvent(
                timestamp=ts.isoformat(),
                session_tag=session_tag_for_timestamp(ts),
                side=side,
                entry_price_cents=price,
                won=won,
                pnl_cents=int(pnl),
                xgb_vote=side.upper(),
                xgb_conf=0.5,
                momentum_vote=side.upper(),
                momentum_conf=0.5,
                label=2 if side == "up" else 0,
            )
        )

    if not events:
        log.warning("No valid calibration events after filtering.")
        return _build_from_kalshi_archive_empty(args, series_list, start_d, end_d)

    artifact = build_calibration_artifact(
        events,
        metadata={
            "source_mode": "kalshi_archive",
            "kalshi_series": series_list,
            "kalshi_archive_range": f"{start_d.isoformat()} to {end_d.isoformat()}",
            "settled_trades": len(df),
            "calibration_events": len(events),
        },
        min_trades=int(args.min_trades),
        ev_buffer_cents=float(args.ev_buffer_cents),
        min_net_ev_cents=float(args.min_net_ev_cents),
    )

    # Preserve schema expected by engine.crypto_decision.load_crypto_calibration:
    # it reads top-level "rows" (or "cells"), "bucket_size_cents", "version".
    artifact["rows"] = artifact["cells"]
    artifact["bucket_size_cents"] = int(artifact["metadata"]["bucket_width_cents"])
    artifact["version"] = artifact["metadata"].get("artifact_id", "")
    artifact["data_source"] = "kalshi_archive"
    artifact["kalshi_archive_range"] = f"{start_d.isoformat()} to {end_d.isoformat()}"
    artifact["kalshi_series"] = series_list

    out_path = KALSHI_OUTPUT_PATH if args.output == Path(settings.CRYPTO_CALIBRATION_PATH) else args.output
    out = save_artifact(artifact, out_path)
    summary = artifact["summary"]
    print(f"Wrote Kalshi-archive calibration artifact: {out}")
    print(f"Events: {len(events)}  Tradable buckets: {summary['tradable_bucket_count']}")
    return 0


def _build_from_kalshi_archive_empty(
    args: argparse.Namespace,
    series_list: list[str],
    start_d: date,
    end_d: date,
) -> int:
    from engine.crypto_calibration import PRICE_BUCKET_WIDTH_CENTS

    artifact = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "artifact_type": "crypto_conjunction_calibration",
            "artifact_version": 1,
            "bucket_width_cents": PRICE_BUCKET_WIDTH_CENTS,
            "min_trades": int(args.min_trades),
            "ev_buffer_cents": float(args.ev_buffer_cents),
            "min_net_ev_cents": float(args.min_net_ev_cents),
            "total_events": 0,
            "tradable_bucket_count": 0,
            "artifact_id": "empty_kalshi_archive",
        },
        "summary": {
            "total_events": 0,
            "bucket_count": 0,
            "tradable_bucket_count": 0,
            "best_bucket": None,
            "worst_bucket": None,
        },
        "session_stats": {},
        "cells": [],
        "rows": [],
        "bucket_size_cents": PRICE_BUCKET_WIDTH_CENTS,
        "version": "empty_kalshi_archive",
        "data_source": "kalshi_archive",
        "kalshi_archive_range": f"{start_d.isoformat()} to {end_d.isoformat()}",
        "kalshi_series": series_list,
    }
    out_path = KALSHI_OUTPUT_PATH if args.output == Path(settings.CRYPTO_CALIBRATION_PATH) else args.output
    out = save_artifact(artifact, out_path)
    print(f"Wrote EMPTY calibration artifact (no valid events): {out}")
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.use_kalshi_archive:
        return _build_from_kalshi_archive(args)

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
