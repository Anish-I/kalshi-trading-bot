"""
Scheduled Retrain: Loads feature parquets, creates balanced labels,
trains XGBoost on CUDA, saves timestamped model + updates latest pointer.

Run every 12 hours via Windows Scheduled Task "KalshiRetrain".
"""
import sys
sys.path.insert(0, ".")

import os
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score

# ── Paths ────────────────────────────────────────────────────────────────
FEATURES_DIR = Path("D:/kalshi-data/features")
MODELS_DIR = Path("D:/kalshi-models")
LOG_FILE = Path("D:/kalshi-data/logs/retrain.log")
LATEST_MODEL = MODELS_DIR / "latest_model.json"

# ── Logging ──────────────────────────────────────────────────────────────
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE), mode="a"),
    ],
)
log = logging.getLogger("retrain")

# ── Hyperparameters ──────────────────────────────────────────────────────
LABEL_HORIZON = 180        # 180 x 5s bars = 15 minutes
LABEL_THRESHOLD = 3e-4     # 3 bps
PURGE_GAP = 180            # walk-forward purge gap in bars
TRAIN_FRAC = 0.80

XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "device": "cuda",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "verbosity": 1,
    "random_state": 42,
}

EXCLUDE_COLS = {"timestamp", "btc_price", "label"}


def load_features() -> pd.DataFrame:
    """Load and concatenate all parquet files from the features directory."""
    parquet_files = sorted(FEATURES_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {FEATURES_DIR}")

    frames = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        frames.append(df)
        log.info(f"Loaded {f.name}: {len(df):,} rows, {df.shape[1]} cols")

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Total rows after concat: {len(combined):,}")
    return combined


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create balanced 3-class labels from forward return."""
    if "btc_price" not in df.columns:
        raise ValueError("btc_price column required for label creation")

    # Forward return over LABEL_HORIZON bars
    fwd_price = df["btc_price"].shift(-LABEL_HORIZON)
    fwd_ret = (fwd_price - df["btc_price"]) / df["btc_price"]

    # 3-class label: -1 (down), 0 (flat), 1 (up)
    labels = np.where(
        fwd_ret > LABEL_THRESHOLD, 1,
        np.where(fwd_ret < -LABEL_THRESHOLD, -1, 0)
    ).astype(float)

    # Last LABEL_HORIZON rows have no valid forward return — mark as NaN
    labels[-LABEL_HORIZON:] = np.nan

    df["label"] = labels

    # Drop rows where we can't compute forward return (NaN from shift + trailing rows)
    valid = ~np.isnan(df["label"])
    df = df[valid].copy()
    # Shift label to 0-indexed for XGBoost: {-1 -> 0, 0 -> 1, 1 -> 2}
    df["label"] = df["label"].astype(int) + 1

    log.info(f"Label distribution:\n{df['label'].value_counts().sort_index().to_string()}")
    return df


def train_model(df: pd.DataFrame) -> tuple[xgb.XGBClassifier, list[str]]:
    """Walk-forward split with purge gap, train XGBoost on CUDA."""
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    log.info(f"Using {len(feature_cols)} features")

    X = df[feature_cols].values
    y = df["label"].values

    # Walk-forward split: first 80% train, last 20% test, with purge gap
    split_idx = int(len(df) * TRAIN_FRAC)
    test_start = split_idx + PURGE_GAP

    if test_start >= len(df):
        test_start = split_idx
        log.warning("Purge gap exceeds available data, using split without gap")

    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[test_start:], y[test_start:]

    log.info(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | Purge gap: {PURGE_GAP} bars")

    # Class weights for imbalanced data
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    weight_dict = {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weights = np.array([weight_dict[int(yi)] for yi in y_train])

    log.info(f"Class weights: {weight_dict}")

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    log.info(f"Test accuracy: {acc:.4f}")
    log.info(f"Classification report:\n{classification_report(y_test, y_pred, zero_division=0)}")

    return model, feature_cols


def save_model(model: xgb.XGBClassifier, feature_cols: list[str]) -> Path:
    """Save model with timestamp and update latest_model.json pointer."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"xgb_balanced_{timestamp}.json"

    model.save_model(str(model_path))
    log.info(f"Model saved: {model_path}")

    # Save feature schema alongside model
    schema_path = str(model_path).replace('.json', '.schema.json')
    with open(schema_path, 'w') as f:
        json.dump(feature_cols, f)
    log.info(f"Feature schema saved: {schema_path}")

    # Update latest_model.json — copy the model file so latest always works
    shutil.copy2(str(model_path), str(LATEST_MODEL))
    log.info(f"Updated {LATEST_MODEL} -> {model_path.name}")

    # Also save schema for latest_model
    latest_schema = str(LATEST_MODEL).replace('.json', '.schema.json')
    with open(latest_schema, 'w') as f:
        json.dump(feature_cols, f)

    return model_path


def main():
    log.info("=" * 60)
    log.info("SCHEDULED RETRAIN STARTED")
    log.info("=" * 60)

    try:
        df = load_features()
        df = create_labels(df)
        model, feature_cols = train_model(df)
        model_path = save_model(model, feature_cols)
        log.info(f"RETRAIN COMPLETE: {model_path}")
    except Exception as e:
        log.exception(f"RETRAIN FAILED: {e}")
        raise

    log.info("=" * 60)


if __name__ == "__main__":
    main()
