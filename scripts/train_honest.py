#!/usr/bin/env python3
"""Train honest XGBoost model on Binance historical data with 32 real features."""
import sys
sys.path.insert(0, ".")

import argparse
import contextlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, log_loss

from features.honest_features import (
    HONEST_FEATURE_NAMES,
    compute_all_features,
    compute_honest_features,
)


@contextlib.contextmanager
def _macro_features_enabled(enable: bool):
    """Temporarily toggle MACRO_FEATURES_ENABLED on the settings module for
    the duration of this process without persisting to disk."""
    from config import settings as settings_module
    s = settings_module.settings
    prev = getattr(s, "MACRO_FEATURES_ENABLED", False)
    try:
        object.__setattr__(s, "MACRO_FEATURES_ENABLED", bool(enable))
    except Exception:
        setattr(s, "MACRO_FEATURES_ENABLED", bool(enable))
    try:
        yield
    finally:
        try:
            object.__setattr__(s, "MACRO_FEATURES_ENABLED", prev)
        except Exception:
            setattr(s, "MACRO_FEATURES_ENABLED", prev)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_FILE = Path("D:/kalshi-data/binance_history/klines_90d.parquet")
MODEL_DIR = Path("D:/kalshi-models")
FORWARD_BARS = 15          # 15-min forward return on 1m bars
THRESHOLD_BPS = 0.001      # 10 bps
N_FOLDS = 5
PURGE_BARS = 15


def make_labels(close: pd.Series) -> pd.Series:
    """Create 3-class labels: 0=DOWN, 1=FLAT, 2=UP."""
    fwd_ret = np.log(close.shift(-FORWARD_BARS) / close)
    labels = pd.Series(np.where(
        fwd_ret > THRESHOLD_BPS, 2,
        np.where(fwd_ret < -THRESHOLD_BPS, 0, 1)
    ), index=close.index)
    return labels


def walk_forward_split(n: int, n_folds: int, purge: int):
    """Yield (train_idx, test_idx) for walk-forward CV with purge gap."""
    fold_size = n // (n_folds + 1)
    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        test_start = train_end + purge
        test_end = min(test_start + fold_size, n)
        if test_start >= n or test_end <= test_start:
            continue
        yield np.arange(0, train_end), np.arange(test_start, test_end)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features",
        choices=["honest", "honest_plus_macro"],
        default="honest",
        help="Feature set to train on. 'honest' = 32 baseline features. "
             "'honest_plus_macro' = baseline + optional macro features "
             "(MACRO_FEATURES_ENABLED toggled in-process only).",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    feature_set = args.features
    use_macro = feature_set == "honest_plus_macro"

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    log.info(f"Loading data from {DATA_FILE}")
    if not DATA_FILE.exists():
        log.error(f"Data file not found: {DATA_FILE}")
        log.error("Run scripts/download_binance_history.py first.")
        sys.exit(1)

    raw = pd.read_parquet(DATA_FILE)
    log.info(f"Raw data: {len(raw):,} rows, {raw['timestamp'].min()} -> {raw['timestamp'].max()}")

    # ------------------------------------------------------------------
    # 2. Compute features
    # ------------------------------------------------------------------
    log.info(f"Computing features (feature_set={feature_set})...")
    with _macro_features_enabled(use_macro):
        if use_macro:
            df = compute_all_features(raw)
        else:
            df = compute_honest_features(raw)
    # Build feature list: honest baseline + any extra columns macro added.
    if use_macro:
        extra = [c for c in df.columns if c not in HONEST_FEATURE_NAMES
                 and c not in {"timestamp", "open", "high", "low", "close",
                               "volume", "buy_volume", "sell_volume", "trade_count",
                               "ts", "label"}]
        feature_names = list(HONEST_FEATURE_NAMES) + extra
    else:
        feature_names = list(HONEST_FEATURE_NAMES)
    log.info(f"After feature computation: {len(df):,} rows, {len(feature_names)} features")

    # ------------------------------------------------------------------
    # 3. Create labels
    # ------------------------------------------------------------------
    df["label"] = make_labels(df["close"])
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    # Drop last FORWARD_BARS rows (no valid label)
    df = df.iloc[:-FORWARD_BARS].reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # Drop rows with NaN in any feature (macro features may introduce NaN)
    df = df.dropna(subset=feature_names).reset_index(drop=True)

    X = df[feature_names].values
    y = df["label"].values
    n = len(y)

    log.info(f"Final dataset: {n:,} samples, {len(feature_names)} features")

    # ------------------------------------------------------------------
    # 4. Diagnostics
    # ------------------------------------------------------------------
    class_counts = pd.Series(y).value_counts().sort_index()
    class_names = {0: "DOWN", 1: "FLAT", 2: "UP"}
    print("\n" + "=" * 60)
    print("CLASS BALANCE")
    print("=" * 60)
    for cls_id, cnt in class_counts.items():
        print(f"  {class_names[cls_id]:>5}: {cnt:>7,}  ({100 * cnt / n:.1f}%)")
    print(f"  {'TOTAL':>5}: {n:>7,}")
    print(f"  Features: {len(feature_names)}")
    print("=" * 60)

    # Compute scale_pos_weight equivalent for multiclass
    # Use sample weights: weight_i = n / (n_classes * count_of_class_i)
    n_classes = 3
    sample_weights_map = {c: n / (n_classes * cnt) for c, cnt in class_counts.items()}

    # ------------------------------------------------------------------
    # 5. Walk-forward CV
    # ------------------------------------------------------------------
    print("\nWALK-FORWARD CROSS-VALIDATION (5 folds, 15-bar purge)")
    print("-" * 60)

    fold_metrics = []
    all_test_preds = []
    all_test_labels = []

    for fold_i, (train_idx, test_idx) in enumerate(walk_forward_split(n, N_FOLDS, PURGE_BARS)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Sample weights for imbalance
        w_train = np.array([sample_weights_map[yi] for yi in y_train])

        # Try GPU, fall back to CPU
        try:
            model = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                device="cuda",
                max_depth=5,
                learning_rate=0.03,
                n_estimators=300,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                verbosity=0,
                random_state=42,
            )
            model.fit(X_train, y_train, sample_weight=w_train)
        except Exception:
            log.info("CUDA not available, falling back to CPU")
            model = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                device="cpu",
                max_depth=5,
                learning_rate=0.03,
                n_estimators=300,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                verbosity=0,
                random_state=42,
            )
            model.fit(X_train, y_train, sample_weight=w_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob, labels=[0, 1, 2])

        fold_metrics.append({"fold": fold_i + 1, "accuracy": acc, "log_loss": ll,
                             "train_size": len(train_idx), "test_size": len(test_idx)})
        all_test_preds.extend(y_pred.tolist())
        all_test_labels.extend(y_test.tolist())

        print(f"  Fold {fold_i + 1}: acc={acc:.4f}  log_loss={ll:.4f}  "
              f"train={len(train_idx):,}  test={len(test_idx):,}")

    # ------------------------------------------------------------------
    # 6. Overall CV metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("OVERALL CV RESULTS")
    print("=" * 60)
    avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
    avg_ll = np.mean([m["log_loss"] for m in fold_metrics])
    print(f"  Mean accuracy:  {avg_acc:.4f}")
    print(f"  Mean log_loss:  {avg_ll:.4f}")
    print()

    all_test_preds_arr = np.array(all_test_preds)
    all_test_labels_arr = np.array(all_test_labels)
    print("Classification Report (all CV test folds combined):")
    print(classification_report(all_test_labels_arr, all_test_preds_arr,
                                target_names=["DOWN", "FLAT", "UP"]))

    pred_dist = pd.Series(all_test_preds_arr).value_counts().sort_index()
    print("Prediction distribution (CV):")
    for cls_id, cnt in pred_dist.items():
        total = len(all_test_preds_arr)
        print(f"  {class_names[cls_id]:>5}: {cnt:>7,}  ({100 * cnt / total:.1f}%)")

    # ------------------------------------------------------------------
    # 7. Train final model on all data
    # ------------------------------------------------------------------
    log.info("Training final model on all data...")
    w_all = np.array([sample_weights_map[yi] for yi in y])

    try:
        final_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            device="cuda",
            max_depth=5,
            learning_rate=0.03,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            verbosity=0,
            random_state=42,
        )
        final_model.fit(X, y, sample_weight=w_all)
    except Exception:
        final_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            device="cpu",
            max_depth=5,
            learning_rate=0.03,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            verbosity=0,
            random_state=42,
        )
        final_model.fit(X, y, sample_weight=w_all)

    # ------------------------------------------------------------------
    # 8. Save model
    # ------------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    name_stem = "xgb_honest_plus_macro" if use_macro else "xgb_honest"
    model_path = MODEL_DIR / f"{name_stem}_{timestamp}.json"
    schema_path = MODEL_DIR / f"{name_stem}_{timestamp}.schema.json"
    latest_model = MODEL_DIR / "latest_model.json"
    latest_schema = MODEL_DIR / "latest_model.schema.json"

    final_model.save_model(str(model_path))
    log.info(f"Model saved: {model_path}")

    schema = {
        "model_type": name_stem,
        "feature_set": feature_set,
        "timestamp": timestamp,
        "features": feature_names,
        "n_features": len(feature_names),
        "n_classes": n_classes,
        "class_map": class_names,
        "forward_bars": FORWARD_BARS,
        "threshold_bps": THRESHOLD_BPS,
        "training_samples": n,
        "cv_accuracy": round(avg_acc, 4),
        "cv_log_loss": round(avg_ll, 4),
        "data_file": str(DATA_FILE),
    }
    schema_path.write_text(json.dumps(schema, indent=2))
    log.info(f"Schema saved: {schema_path}")

    # Copy to latest
    shutil.copy2(model_path, latest_model)
    shutil.copy2(schema_path, latest_schema)
    log.info(f"Copied to {latest_model} and {latest_schema}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:    {model_path}")
    print(f"  Schema:   {schema_path}")
    print(f"  Latest:   {latest_model}")
    print(f"  Samples:  {n:,}")
    print(f"  Features: {len(feature_names)}  ({feature_set})")
    print(f"  CV acc:   {avg_acc:.4f}")
    print(f"  CV loss:  {avg_ll:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
