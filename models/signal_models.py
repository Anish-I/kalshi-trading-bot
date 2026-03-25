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
        logger.info("XGBoost loaded from %s", model_path)

    def score(self, features_df) -> tuple[str, float]:
        exclude = {"timestamp", "btc_price", "label"}
        cols = [c for c in features_df.columns if c not in exclude]
        X = features_df[cols].iloc[-1:].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        dmat = xgb.DMatrix(X)
        proba = self.booster.predict(dmat)[0]
        p_down, p_flat, p_up = float(proba[0]), float(proba[1]), float(proba[2])

        if p_up > p_down and p_up > p_flat:
            return "up", p_up
        elif p_down > p_up and p_down > p_flat:
            return "down", p_down
        return "flat", p_flat


class MomentumModel:
    """Model 2: Pure price action — catches trends.

    No ML, no training, no overfitting. Just price momentum signals.
    """

    def score(self, features: dict) -> tuple[str, float]:
        signals = 0

        # 5-min return (optimized: wider threshold)
        m5 = features.get("momentum_5m", 0) or 0
        if m5 > 0.002:
            signals += 1
        elif m5 < -0.002:
            signals -= 1

        # 10-min return (optimized: wider)
        m10 = features.get("momentum_10m", 0) or 0
        if m10 > 0.003:
            signals += 1
        elif m10 < -0.003:
            signals -= 1

        # Price vs VWAP
        vwap_dev = features.get("vwap_deviation", 0) or 0
        if vwap_dev > 0.001:
            signals += 1
        elif vwap_dev < -0.001:
            signals -= 1

        # Donchian position (optimized: need more extreme)
        donch = features.get("donchian_position", 0.5) or 0.5
        if donch > 0.8:
            signals += 1
        elif donch < 0.2:
            signals -= 1

        # EMA 9 slope
        ema_s = features.get("ema_9_slope", 0) or 0
        if ema_s > 0.0001:
            signals += 1
        elif ema_s < -0.0001:
            signals -= 1

        # Need 4/5 signals for higher selectivity (was 3/5)
        if signals >= 4:
            return "up", min(0.80, 0.55 + signals * 0.05)
        if signals <= -4:
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


def vote(results: list[tuple[str, float]]) -> tuple[str, float, int]:
    """Combine model votes into a single trading decision.

    Args:
        results: List of (direction, confidence) from each model.

    Returns:
        (direction, avg_confidence, agreement_count)
        - 3-4 agree → strong signal, full confidence
        - 2 agree (0 oppose) → moderate signal, reduced confidence
        - <2 agree or conflicting → flat, no trade
    """
    up_votes = [(d, c) for d, c in results if d == "up"]
    down_votes = [(d, c) for d, c in results if d == "down"]
    n_up = len(up_votes)
    n_down = len(down_votes)

    if n_up >= 3:
        avg_conf = mean(c for _, c in up_votes)
        return "up", avg_conf, n_up

    if n_down >= 3:
        avg_conf = mean(c for _, c in down_votes)
        return "down", avg_conf, n_down

    if n_up == 2 and n_down == 0:
        avg_conf = mean(c for _, c in up_votes) * 0.9
        return "up", avg_conf, n_up

    if n_down == 2 and n_up == 0:
        avg_conf = mean(c for _, c in down_votes) * 0.9
        return "down", avg_conf, n_down

    return "flat", 0.50, 0
