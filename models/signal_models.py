"""
Signal models for multi-model voting system.

Each model independently scores UP/DOWN/FLAT with a confidence level.
The voting engine combines them — trades only fire when multiple agree.
"""

import math
import logging
from statistics import mean
from pathlib import Path

import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)


class XGBoostModel:
    """Model 1: Trained ML model on 83 microstructure features."""

    def __init__(self, model_path: str):
        self.booster = xgb.Booster()
        self.booster.load_model(model_path)
        # Load feature schema if available
        schema_path = model_path.replace('.json', '.schema.json')
        self.feature_cols: list[str] | None = None
        try:
            import json
            with open(schema_path) as f:
                raw = json.load(f)
            # Handle both formats: plain list or dict with "features" key
            if isinstance(raw, list):
                self.feature_cols = raw
            elif isinstance(raw, dict) and "features" in raw:
                self.feature_cols = raw["features"]
            if self.feature_cols:
                logger.info("Loaded feature schema (%d cols) from %s", len(self.feature_cols), schema_path)
        except Exception:
            logger.warning("No feature schema at %s, using dynamic columns", schema_path)
        logger.info("XGBoost loaded from %s", model_path)

    def score(self, features_df) -> tuple[str, float]:
        exclude = {"timestamp", "btc_price", "label"}
        if self.feature_cols:
            # Ensure exact column match with training schema
            for col in self.feature_cols:
                if col not in features_df.columns:
                    features_df = features_df.copy()
                    features_df[col] = 0.0
            cols = self.feature_cols
        else:
            cols = [c for c in features_df.columns if c not in exclude]
        X = features_df[cols].iloc[-1:].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        dmat = xgb.DMatrix(X)
        proba = self.booster.predict(dmat)[0]
        p_down, p_flat, p_up = float(proba[0]), float(proba[1]), float(proba[2])

        # Convert 3-class to binary: P(YES) = p_up / (p_up + p_down)
        denom = p_up + p_down
        if denom < 0.01:
            return "flat", 0.50
        p_binary_up = p_up / denom
        p_binary_down = p_down / denom

        # XGBoost is undertrained — demote to advisory role
        # Cap at 0.65 so it can't dominate the voting
        MAX_XGB_CONF = 0.65

        if p_binary_up > 0.55:
            return "up", min(p_binary_up, MAX_XGB_CONF)
        elif p_binary_down > 0.55:
            return "down", min(p_binary_down, MAX_XGB_CONF)
        return "flat", 0.50


class MomentumModel:
    """Model 2: Pure price action — catches trends.

    No ML, no training, no overfitting. Just price momentum signals.
    """

    def score(self, features: dict) -> tuple[str, float]:
        signals = 0

        # 5-min return (calibrated to p75 of actual BTC data)
        m5 = features.get("momentum_5m") or features.get("ret_5m", 0) or 0
        if m5 > 0.0007:
            signals += 1
        elif m5 < -0.0007:
            signals -= 1

        # 10-min return
        m10 = features.get("momentum_10m") or features.get("ret_10m", 0) or 0
        if m10 > 0.0012:
            signals += 1
        elif m10 < -0.0012:
            signals -= 1

        # Price vs VWAP
        vwap_dev = features.get("vwap_deviation", 0) or 0
        if vwap_dev > 0.002:
            signals += 1
        elif vwap_dev < -0.002:
            signals -= 1

        # Donchian position (>0.75 = near high, <0.25 = near low)
        donch = features.get("donchian_position", 0.5) or 0.5
        if donch > 0.75:
            signals += 1
        elif donch < 0.25:
            signals -= 1

        # EMA 9 slope
        ema_s = features.get("ema_9_slope", 0) or 0
        if ema_s > 0.00012:
            signals += 1
        elif ema_s < -0.00012:
            signals -= 1

        # Need 3/5 signals (was 4/5 which was too strict)
        if signals >= 3:
            return "up", min(0.80, 0.55 + signals * 0.05)
        if signals <= -3:
            return "down", min(0.80, 0.55 + abs(signals) * 0.05)
        return "flat", 0.50


class MeanReversionModel:
    """Model 3: Catches bounces when price is overextended.

    Looks for RSI/Stochastic/Bollinger extremes that suggest a reversal.
    """

    def score(self, features: dict) -> tuple[str, float]:
        signals = 0

        # RSI extremes (optimized: rsi_lo=25, rsi_hi=80)
        rsi = features.get("rsi_14", 50) or 50
        if rsi < 25:
            signals += 2
        elif rsi < 35:
            signals += 1
        elif rsi > 80:
            signals -= 2
        elif rsi > 70:
            signals -= 1

        # Donchian position as proxy for BB (optimized: donch_lo=0.25)
        donch = features.get("donchian_position", 0.5) or 0.5
        if donch < 0.25:
            signals += 2
        elif donch < 0.35:
            signals += 1
        elif donch > 0.75:
            signals -= 2
        elif donch > 0.65:
            signals -= 1

        # Stochastic %K (optimized: stoch_lo=10)
        stoch = features.get("stoch_k", 50) or 50
        if stoch < 10:
            signals += 1
        elif stoch > 90:
            signals -= 1

        # Volume spike amplifies signal
        vol_ratio = features.get("volume_sma_ratio", 1.0) or 1.0
        if vol_ratio > 2.0 and abs(signals) >= 2:
            signals = int(signals * 1.5)

        if signals >= 3:
            return "up", min(0.80, 0.55 + min(abs(signals), 5) * 0.04)
        if signals <= -3:
            return "down", min(0.80, 0.55 + min(abs(signals), 5) * 0.04)
        return "flat", 0.50


class KalshiConsensusModel:
    """Model 4: What Kalshi's own market participants think.

    Uses the orderbook depth imbalance, price velocity, and market mid
    to read the collective signal from real-money traders.
    """

    def score(self, kalshi_imbalance: float, velocity: float, market_mid: float) -> tuple[str, float]:
        signals = 0

        # Orderbook depth imbalance
        if kalshi_imbalance > 0.3:
            signals += 2
        elif kalshi_imbalance > 0.1:
            signals += 1
        elif kalshi_imbalance < -0.3:
            signals -= 2
        elif kalshi_imbalance < -0.1:
            signals -= 1

        # Price velocity (cents/min)
        if velocity > 0.05:
            signals += 1
        elif velocity < -0.05:
            signals -= 1
        if velocity > 0.15:
            signals += 1
        elif velocity < -0.15:
            signals -= 1

        # Market mid price lean
        if market_mid > 0.60:
            signals += 1
        elif market_mid < 0.40:
            signals -= 1

        if signals >= 2:
            return "up", min(0.80, 0.55 + min(abs(signals), 4) * 0.05)
        if signals <= -2:
            return "down", min(0.80, 0.55 + min(abs(signals), 4) * 0.05)
        return "flat", 0.50


def vote(
    results: list[tuple[str, float]],
    weights: list[float] | None = None,
) -> tuple[str, float, float]:
    """Weighted voting: models with higher adaptive weights count more.

    Args:
        results: List of (direction, confidence) from each model.
        weights: Per-model weights (default 1.0 each). A weight of 2.0
                 means that model's vote counts double.

    Returns:
        (direction, clamped_confidence, weighted_score)
        - weighted_score >= 2.0 with no stronger opposition → trade
        - Otherwise → flat, no trade
        - Confidence always clamped to [0, 0.95]
    """
    if weights is None:
        weights = [1.0] * len(results)

    up_score = sum(w for (d, _), w in zip(results, weights) if d == "up")
    down_score = sum(w for (d, _), w in zip(results, weights) if d == "down")

    if up_score >= 2.0 and up_score > down_score:
        # Weighted average of UP-voting models' confidences
        total_w = sum(w for (d, _), w in zip(results, weights) if d == "up")
        avg_conf = sum(c * w for (d, c), w in zip(results, weights) if d == "up") / total_w
        return "up", min(avg_conf, 0.95), round(up_score, 2)

    if down_score >= 2.0 and down_score > up_score:
        total_w = sum(w for (d, _), w in zip(results, weights) if d == "down")
        avg_conf = sum(c * w for (d, c), w in zip(results, weights) if d == "down") / total_w
        return "down", min(avg_conf, 0.95), round(down_score, 2)

    return "flat", 0.50, 0.0
