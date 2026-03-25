"""Ensemble combining XGBoost and Transformer predictions."""

import logging

import numpy as np
import pandas as pd

from models.xgboost_model import XGBoostDirectionModel
from models.transformer_model import TransformerDirectionModel

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Weighted ensemble of XGBoost and Transformer direction models."""

    def __init__(self, xgb_weight: float = 0.6, transformer_weight: float = 0.4):
        if not np.isclose(xgb_weight + transformer_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {xgb_weight + transformer_weight}")
        self.xgb_weight = xgb_weight
        self.transformer_weight = transformer_weight
        self.xgb_model: XGBoostDirectionModel | None = None
        self.transformer_model: TransformerDirectionModel | None = None

    def set_models(
        self,
        xgb_model: XGBoostDirectionModel,
        transformer_model: TransformerDirectionModel | None = None,
    ) -> None:
        """Assign models. Transformer is optional."""
        self.xgb_model = xgb_model
        self.transformer_model = transformer_model

    def predict_proba(
        self,
        xgb_features: pd.DataFrame,
        transformer_sequences: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return weighted-average probabilities (N, 3) or (3,) for single sample.

        Columns: [p_down, p_flat, p_up].
        """
        if self.xgb_model is None:
            raise RuntimeError("XGBoost model not set")

        xgb_probas = self.xgb_model.predict_proba(xgb_features)

        if self.transformer_model is not None and transformer_sequences is not None:
            tf_probas = self.transformer_model.predict_proba(transformer_sequences)
            combined = self.xgb_weight * xgb_probas + self.transformer_weight * tf_probas
        else:
            combined = xgb_probas

        # Squeeze to 1-D if single sample
        if combined.shape[0] == 1:
            combined = combined.squeeze(0)

        return combined

    def predict_direction(
        self,
        xgb_features: pd.DataFrame,
        transformer_sequences: np.ndarray | None = None,
        confidence_threshold: float = 0.65,
    ) -> tuple[int, float]:
        """Predict direction with confidence gating.

        Returns:
            (direction, confidence) where direction is {-1, 0, 1}.
            0 means abstain (confidence below threshold).
        """
        probas = self.predict_proba(xgb_features, transformer_sequences)
        # Ensure 1-D
        if probas.ndim > 1:
            probas = probas.squeeze(0)

        p_down, _p_flat, p_up = probas[0], probas[1], probas[2]
        direction_probs = [p_down, p_up]
        max_prob = float(max(direction_probs))

        if max_prob > confidence_threshold:
            direction = 1 if p_up > p_down else -1
            logger.info("Ensemble prediction: direction=%d, confidence=%.4f", direction, max_prob)
            return direction, max_prob

        logger.info("Ensemble abstaining: max_directional_prob=%.4f < threshold=%.2f", max_prob, confidence_threshold)
        return 0, max_prob
