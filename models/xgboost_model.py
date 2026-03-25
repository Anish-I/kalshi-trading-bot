"""XGBoost-based direction classifier for Bitcoin price movement."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class XGBoostDirectionModel:
    """Multi-class XGBoost classifier predicting price direction {down, flat, up}."""

    DEFAULT_PARAMS = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 50,
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "mlogloss",
        "random_state": 42,
    }

    def __init__(self, params: dict | None = None):
        self.params = {**self.DEFAULT_PARAMS}
        if params:
            self.params.update(params)
        self.model: XGBClassifier | None = None
        self.feature_names: list[str] | None = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train the classifier.

        Args:
            X: Feature matrix.
            y: Labels in {-1, 0, 1}.

        Returns:
            Dict with training metrics.
        """
        y_mapped = y + 1  # {-1,0,1} -> {0,1,2}
        self.feature_names = list(X.columns)

        self.model = XGBClassifier(**self.params)
        self.model.fit(X, y_mapped)

        train_probas = self.model.predict_proba(X)
        train_logloss = log_loss(y_mapped, train_probas, labels=[0, 1, 2])

        logger.info("XGBoost training complete — log_loss=%.4f, samples=%d, features=%d",
                     train_logloss, len(X), len(self.feature_names))
        return {"log_loss": train_logloss, "n_samples": len(X), "n_features": len(self.feature_names)}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return (N, 3) array of [p_down, p_flat, p_up]."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class predictions mapped back to {-1, 0, 1}."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        preds = self.model.predict(X)
        return preds.astype(int) - 1  # {0,1,2} -> {-1,0,1}

    def get_feature_importance(self) -> pd.Series:
        """Return feature importance as a named Series sorted descending."""
        if self.model is None or self.feature_names is None:
            raise RuntimeError("Model not trained or loaded")
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)

    def save(self, path: str) -> None:
        """Save model to JSON and feature names alongside."""
        if self.model is None:
            raise RuntimeError("Model not trained — nothing to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        features_path = f"{path}.features.json"
        with open(features_path, "w") as f:
            json.dump(self.feature_names, f)
        logger.info("XGBoost model saved to %s", path)

    def load(self, path: str) -> None:
        """Load model from JSON and feature names."""
        self.model = XGBClassifier()
        self.model.load_model(path)
        features_path = f"{path}.features.json"
        with open(features_path) as f:
            self.feature_names = json.load(f)
        logger.info("XGBoost model loaded from %s (%d features)", path, len(self.feature_names))
