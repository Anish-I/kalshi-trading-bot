"""Train XGBoost model on collected/backfilled feature data."""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import TimeSeriesSplit

from config.settings import settings
from data.storage import DataStorage
from models.xgboost_model import XGBoostDirectionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def walk_forward_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict:
    """Walk-forward cross-validation for time-series data.

    Returns:
        Dict with mean metrics across folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBoostDirectionModel()
        model.train(X_train, y_train)

        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)

        y_test_mapped = y_test + 1  # {-1,0,1} -> {0,1,2}
        try:
            ll = log_loss(y_test_mapped, probas, labels=[0, 1, 2])
        except ValueError:
            ll = float("nan")

        accuracy = float(np.mean(preds == y_test.values))

        fold_metrics.append({
            "fold": fold,
            "log_loss": ll,
            "accuracy": accuracy,
            "train_size": len(X_train),
            "test_size": len(X_test),
        })

        logger.info(
            "Fold %d: accuracy=%.4f  log_loss=%.4f  train=%d  test=%d",
            fold, accuracy, ll, len(X_train), len(X_test),
        )

    metrics_df = pd.DataFrame(fold_metrics)
    return {
        "mean_accuracy": float(metrics_df["accuracy"].mean()),
        "std_accuracy": float(metrics_df["accuracy"].std()),
        "mean_log_loss": float(metrics_df["log_loss"].mean()),
        "folds": fold_metrics,
    }


def main() -> None:
    logger.info("=== Training pipeline starting ===")

    storage = DataStorage()

    # 1. Load features
    features_df = storage.load_features()
    if features_df.empty:
        logger.error("No feature data found in %s. Run backfill.py first.", settings.DATA_DIR)
        return

    if "label" not in features_df.columns:
        logger.error("Feature data missing 'label' column")
        return

    # Separate features and labels
    non_feature_cols = {"timestamp", "label"}
    feature_cols = [c for c in features_df.columns if c not in non_feature_cols]
    X = features_df[feature_cols].astype(float)
    y = features_df["label"].astype(int)

    # Replace inf with NaN, then fill
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    logger.info(
        "Dataset: %d samples, %d features, label distribution:\n%s",
        len(X), len(feature_cols), y.value_counts().sort_index().to_string(),
    )

    # 2. Walk-forward evaluation
    logger.info("--- Walk-forward evaluation (5 folds) ---")
    wf_metrics = walk_forward_evaluate(X, y, n_splits=5)
    logger.info(
        "Walk-forward results: accuracy=%.4f +/- %.4f  log_loss=%.4f",
        wf_metrics["mean_accuracy"],
        wf_metrics["std_accuracy"],
        wf_metrics["mean_log_loss"],
    )

    # 3. Train final model on all data
    logger.info("--- Training final model on full dataset ---")
    final_model = XGBoostDirectionModel()
    train_metrics = final_model.train(X, y)

    # Classification report
    preds = final_model.predict(X)
    report = classification_report(
        y, preds, target_names=["down (-1)", "flat (0)", "up (+1)"],
    )
    logger.info("Classification report (full dataset):\n%s", report)

    # Feature importance
    importance = final_model.get_feature_importance()
    logger.info("Top 20 features:\n%s", importance.head(20).to_string())

    # 4. Save model
    model_dir = Path(settings.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"xgb_{timestamp}.json"
    final_model.save(str(model_path))

    logger.info("=== Training complete === Model saved to %s", model_path)
    logger.info(
        "Summary: samples=%d  features=%d  wf_accuracy=%.4f  train_logloss=%.4f",
        len(X), len(feature_cols),
        wf_metrics["mean_accuracy"],
        train_metrics["log_loss"],
    )


if __name__ == "__main__":
    main()
