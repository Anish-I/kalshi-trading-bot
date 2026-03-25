"""
Auto-retrain: waits for collector to accumulate enough data (6 hours),
kills it, then trains on the full feature set including microstructure.

Run: python scripts/auto_retrain.py
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("auto_retrain")

COLLECT_HOURS = 6
MIN_FEATURE_ROWS = 2000
DATA_DIR = Path(settings.DATA_DIR)
MODEL_DIR = Path(settings.MODEL_DIR)


def start_collector() -> subprocess.Popen:
    """Start collect_data.py as a subprocess."""
    log.info("Starting data collector...")
    proc = subprocess.Popen(
        [sys.executable, "scripts/collect_data.py"],
        stdout=open("collect_data_bg.log", "w"),
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    log.info("Collector started (PID %d)", proc.pid)
    return proc


def stop_collector(proc: subprocess.Popen) -> None:
    """Stop the collector gracefully."""
    log.info("Stopping collector (PID %d)...", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log.info("Collector stopped.")


def wait_for_data(hours: float) -> None:
    """Wait for collector to accumulate data, with progress updates."""
    total_seconds = int(hours * 3600)
    start = time.time()

    while time.time() - start < total_seconds:
        elapsed = time.time() - start
        remaining = total_seconds - elapsed
        hours_left = remaining / 3600

        # Check how much data we have
        features_dir = DATA_DIR / "features"
        row_count = 0
        if features_dir.exists():
            for f in features_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(f)
                    row_count += len(df)
                except Exception:
                    pass

        bars_dir = DATA_DIR / "bars_5s"
        bar_count = 0
        if bars_dir.exists():
            for f in bars_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(f)
                    bar_count += len(df)
                except Exception:
                    pass

        log.info(
            "Collecting... %.1fh remaining | %d feature rows, %d 5s bars",
            hours_left, row_count, bar_count,
        )

        # Early exit if we have enough data
        if row_count >= MIN_FEATURE_ROWS:
            log.info("Reached %d feature rows — enough data, proceeding to train.", row_count)
            return

        # Check every 10 minutes
        time.sleep(600)

    log.info("Collection period complete (%.1f hours).", hours)


def load_collected_features() -> pd.DataFrame:
    """Load all collected feature parquet files."""
    features_dir = DATA_DIR / "features"
    if not features_dir.exists():
        return pd.DataFrame()

    dfs = []
    for f in sorted(features_dir.glob("*.parquet")):
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def load_collected_bars() -> pd.DataFrame:
    """Load all collected 1m bar parquet files."""
    bars_dir = DATA_DIR / "bars_1m"
    if not bars_dir.exists():
        return pd.DataFrame()

    dfs = []
    for f in sorted(bars_dir.glob("*.parquet")):
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def train_on_live_features(features_df: pd.DataFrame) -> dict:
    """Train XGBoost on live-collected features with microstructure data."""
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report, accuracy_score

    # Identify feature columns (exclude metadata)
    exclude = {"timestamp", "btc_price", "label"}
    feature_cols = [c for c in features_df.columns if c not in exclude]

    X = features_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

    # We need to create labels from btc_price
    # Forward 15-minute return label (using btc_price column)
    if "btc_price" in features_df.columns:
        price = features_df["btc_price"].astype(float)
        # Features are computed every 5s, so 15 min = 180 rows ahead
        horizon = 180
        fwd_ret = np.log(price.shift(-horizon) / price)
        thr = 3.0 * 1e-4  # 3 bps
        labels = pd.Series(
            np.where(fwd_ret > thr, 1, np.where(fwd_ret < -thr, -1, 0)),
            index=features_df.index,
        )
        labels.iloc[-horizon:] = np.nan
    else:
        log.error("No btc_price column — cannot create labels")
        return {}

    # Drop NaN labels
    valid = ~labels.isna()
    X = X[valid].reset_index(drop=True)
    y = labels[valid].astype(int).reset_index(drop=True)
    y_mc = y + 1  # {-1,0,1} -> {0,1,2}

    log.info("Training set: %d samples, %d features", len(X), len(feature_cols))
    log.info("Label distribution:\n  UP: %d\n  FLAT: %d\n  DOWN: %d",
             (y == 1).sum(), (y == 0).sum(), (y == -1).sum())

    # Count non-NaN features to see microstructure contribution
    sample = X.iloc[0] if len(X) > 0 else pd.Series()
    non_nan = (sample != 0).sum()
    log.info("Non-zero features in first row: %d / %d", non_nan, len(feature_cols))

    if len(X) < 500:
        log.error("Not enough valid samples (%d). Need at least 500.", len(X))
        return {}

    # Walk-forward: train on first 80%, test on last 20%
    split = int(len(X) * 0.8)
    purge = 180  # 15 min purge gap

    X_train = X.iloc[:split]
    y_train = y_mc.iloc[:split]
    X_test = X.iloc[split + purge:]
    y_test = y_mc.iloc[split + purge:]

    if len(X_test) < 100:
        log.warning("Small test set (%d). Using 70/30 split.", len(X_test))
        split = int(len(X) * 0.7)
        X_train = X.iloc[:split]
        y_train = y_mc.iloc[:split]
        X_test = X.iloc[split + purge:]
        y_test = y_mc.iloc[split + purge:]

    log.info("Train: %d samples | Test: %d samples (purge=%d)", len(X_train), len(X_test), purge)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        tree_method="hist",
        device="cuda",
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)

    report = classification_report(
        y_test, preds,
        target_names=["down", "flat", "up"],
        digits=4,
    )
    log.info("Test accuracy: %.4f", acc)
    log.info("Classification report:\n%s", report)

    # Feature importance
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    log.info("Top 20 features:\n%s", importance.head(20).to_string())

    # Check if microstructure features rank high
    micro_features = [f for f in importance.head(20).index if any(
        kw in f for kw in ["mid_price", "spread", "microprice", "imbalance", "depth",
                           "taker", "cvd", "cancel", "retreat", "replenishment"]
    )]
    log.info("Microstructure features in top 20: %d (%s)", len(micro_features), micro_features)

    # Kalshi P&L simulation
    log.info("\n--- Kalshi P&L Simulation ---")
    for threshold in [0.45, 0.50, 0.55, 0.60, 0.65]:
        trades = 0
        wins = 0
        pnl = 0

        for i in range(len(preds)):
            pred = preds[i]
            true = y_test.iloc[i]
            conf = proba[i][pred]

            if pred == 1 or conf < threshold:
                continue

            trades += 1
            entry = int(conf * 100)
            correct = pred == true
            if correct:
                wins += 1
                pnl += 100 - entry
            else:
                pnl -= entry

        if trades > 0:
            log.info(
                "  Threshold=%d%%: trades=%d win=%.1f%% pnl=$%.2f avg=%.1fc",
                int(threshold * 100), trades, wins / trades * 100, pnl / 100, pnl / trades,
            )

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"xgb_live_{ts}.json"
    model.save_model(str(model_path))
    log.info("Model saved to %s", model_path)

    # Also train final model on ALL data for deployment
    log.info("Training deployment model on full dataset...")
    full_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        tree_method="hist",
        device="cuda",
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    full_model.fit(X, y_mc)
    deploy_path = MODEL_DIR / f"xgb_deploy_{ts}.json"
    full_model.save_model(str(deploy_path))
    log.info("Deployment model saved to %s", deploy_path)

    return {
        "test_accuracy": acc,
        "test_size": len(X_test),
        "train_size": len(X_train),
        "total_features": len(feature_cols),
        "micro_features_in_top20": len(micro_features),
        "model_path": str(deploy_path),
    }


def main():
    log.info("=" * 60)
    log.info("AUTO-RETRAIN PIPELINE")
    log.info("Collect %d hours → Kill collector → Train with microstructure", COLLECT_HOURS)
    log.info("=" * 60)

    # Check if collector is already running
    collector_proc = None
    try:
        # Check for existing collector
        result = subprocess.run(
            ["pgrep", "-f", "collect_data.py"],
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            existing_pids = result.stdout.strip().split("\n")
            log.info("Collector already running (PID %s). Using existing.", existing_pids[0])
        else:
            collector_proc = start_collector()
    except Exception:
        collector_proc = start_collector()

    # Wait for data
    log.info("Waiting %d hours for data collection...", COLLECT_HOURS)
    wait_for_data(COLLECT_HOURS)

    # Stop collector
    if collector_proc is not None:
        stop_collector(collector_proc)
    else:
        # Kill existing collector
        try:
            subprocess.run(["pkill", "-f", "collect_data.py"], capture_output=True)
            log.info("Killed existing collector process.")
        except Exception:
            pass

    # Give it a moment to flush
    time.sleep(3)

    # Load and train
    log.info("\n--- Loading collected data ---")
    features_df = load_collected_features()

    if features_df.empty:
        log.error("No feature data found! The collector may not have run long enough.")
        log.info("Feature data expected at: %s/features/", DATA_DIR)
        return

    log.info("Loaded %d feature rows", len(features_df))
    log.info("Columns: %d", len(features_df.columns))
    log.info("Time range: %s to %s",
             features_df["timestamp"].iloc[0],
             features_df["timestamp"].iloc[-1])

    # Train
    results = train_on_live_features(features_df)

    if results:
        log.info("\n" + "=" * 60)
        log.info("RETRAIN COMPLETE")
        log.info("Test accuracy: %.4f", results["test_accuracy"])
        log.info("Microstructure features in top 20: %d", results["micro_features_in_top20"])
        log.info("Model: %s", results["model_path"])
        log.info("=" * 60)

        if results["test_accuracy"] > 0.50:
            log.info("Accuracy > 50%% — model shows promise for paper trading!")
        else:
            log.info("Accuracy <= 50%% — more data or feature tuning needed.")
    else:
        log.error("Training failed. Check logs above.")


if __name__ == "__main__":
    main()
