"""Walk-forward evaluation and PnL simulation for direction models."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    log_loss,
)

from models.xgboost_model import XGBoostDirectionModel

logger = logging.getLogger(__name__)


class WalkForwardEvaluator:
    """Time-series aware walk-forward cross-validation with purging."""

    def __init__(self, n_splits: int = 5, purge_gap_minutes: int = 15):
        self.n_splits = n_splits
        self.purge_gap_minutes = purge_gap_minutes

    def evaluate_xgb(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        timestamps: pd.Series,
    ) -> dict:
        """Walk-forward evaluation of XGBoost with purge gap.

        Args:
            X: Feature matrix.
            y: Labels in {-1, 0, 1}.
            timestamps: Corresponding timestamps for each sample.

        Returns:
            Dict with per-fold results and averages.
        """
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        purge_delta = pd.Timedelta(minutes=self.purge_gap_minutes)

        fold_results = []

        for fold in range(self.n_splits):
            train_end = fold_size * (fold + 1)
            test_start_raw = train_end
            test_end = min(train_end + fold_size, n)

            if test_end <= test_start_raw:
                continue

            test_start_ts = timestamps.iloc[test_start_raw]

            # Purge: remove training samples whose timestamp + purge_gap overlaps test start
            train_mask = timestamps.iloc[:train_end] + purge_delta <= test_start_ts
            X_train = X.iloc[:train_end][train_mask]
            y_train = y.iloc[:train_end][train_mask]

            X_test = X.iloc[test_start_raw:test_end]
            y_test = y.iloc[test_start_raw:test_end]

            if len(X_train) < 100 or len(X_test) < 10:
                logger.warning("Fold %d skipped: train=%d, test=%d", fold, len(X_train), len(X_test))
                continue

            model = XGBoostDirectionModel()
            model.train(X_train, y_train)

            preds = model.predict(X_test)
            probas = model.predict_proba(X_test)

            y_test_mapped = y_test + 1  # for log_loss

            fold_metrics = {
                "fold": fold,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "accuracy": accuracy_score(y_test, preds),
                "balanced_accuracy": balanced_accuracy_score(y_test, preds),
                "log_loss": log_loss(y_test_mapped, probas, labels=[0, 1, 2]),
            }
            fold_results.append(fold_metrics)
            logger.info("Fold %d — acc=%.4f, bal_acc=%.4f, logloss=%.4f",
                         fold, fold_metrics["accuracy"], fold_metrics["balanced_accuracy"],
                         fold_metrics["log_loss"])

        if not fold_results:
            return {"folds": [], "averages": {}}

        averages = {
            "accuracy": np.mean([f["accuracy"] for f in fold_results]),
            "balanced_accuracy": np.mean([f["balanced_accuracy"] for f in fold_results]),
            "log_loss": np.mean([f["log_loss"] for f in fold_results]),
        }
        logger.info("Walk-forward averages — acc=%.4f, bal_acc=%.4f, logloss=%.4f",
                     averages["accuracy"], averages["balanced_accuracy"], averages["log_loss"])

        return {"folds": fold_results, "averages": averages}

    @staticmethod
    def simulate_pnl(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probas: np.ndarray,
        fee_cents: int = 7,
    ) -> dict:
        """Simulate Kalshi PnL based on predictions.

        Only trades when model predicts up or down (not flat).
        Entry price = (1 - confidence) * 100 cents.

        Returns:
            Dict with total_pnl, sharpe_ratio, max_drawdown, win_rate, n_trades.
        """
        trade_mask = y_pred != 0
        if not np.any(trade_mask):
            return {"total_pnl": 0, "sharpe_ratio": 0.0, "max_drawdown": 0, "win_rate": 0.0, "n_trades": 0}

        pnls = []
        for i in np.where(trade_mask)[0]:
            pred = y_pred[i]
            true = y_true[i]
            # Confidence is the probability of the predicted class
            pred_class_idx = int(pred + 1)  # {-1,0,1} -> {0,1,2}
            confidence = float(probas[i, pred_class_idx])
            entry_price = (1.0 - confidence) * 100.0

            if pred == true:
                pnl = 100.0 - entry_price - fee_cents
            else:
                pnl = -(entry_price + fee_cents)
            pnls.append(pnl)

        pnls = np.array(pnls)
        total_pnl = float(pnls.sum())
        n_trades = len(pnls)
        win_rate = float(np.mean(pnls > 0)) if n_trades > 0 else 0.0

        # Sharpe ratio (annualized assuming ~1-min trades, ~525600 per year)
        if pnls.std() > 0:
            sharpe_ratio = float((pnls.mean() / pnls.std()) * np.sqrt(min(n_trades, 525600)))
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

        logger.info("PnL sim: total=%.2f, trades=%d, win_rate=%.2f%%, sharpe=%.2f, max_dd=%.2f",
                     total_pnl, n_trades, win_rate * 100, sharpe_ratio, max_drawdown)

        return {
            "total_pnl": total_pnl,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "n_trades": n_trades,
        }

    @staticmethod
    def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Return sklearn classification report string."""
        return classification_report(
            y_true,
            y_pred,
            target_names=["down", "flat", "up"],
            zero_division=0,
        )
