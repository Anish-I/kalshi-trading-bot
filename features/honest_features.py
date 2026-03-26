#!/usr/bin/env python3
"""
Vectorized feature computation on 1-minute candle DataFrames.

Only features we can honestly compute from OHLCV + buy/sell volume.
No orderbook, no L2, no derivatives data.

TOTAL: 32 features.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature name registry
# ---------------------------------------------------------------------------
HONEST_FEATURE_NAMES = [
    # Returns (6)
    "ret_1m", "ret_5m", "ret_10m", "ret_15m", "ret_30m", "ret_60m",
    # EMAs (5)
    "ema_9", "ema_21", "ema_50", "ema_9_slope", "ema_cross",
    # Volatility (4)
    "volatility_5m", "volatility_15m", "volatility_60m", "vol_ratio",
    # RSI + Stochastic (3)
    "rsi_14", "stoch_k", "stoch_d",
    # VWAP + ATR (3)
    "vwap_deviation", "atr_14", "atr_pct",
    # Donchian (2)
    "donchian_position", "donchian_width",
    # Volume (3)
    "volume_sma_ratio", "taker_imbalance", "buy_sell_ratio",
    # Calendar (6)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_us_session", "is_asia_session",
]

assert len(HONEST_FEATURE_NAMES) == 32, f"Expected 32 features, got {len(HONEST_FEATURE_NAMES)}"


def compute_honest_features(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute 32 honest features from 1-minute OHLCV + buy/sell volume bars.

    Parameters
    ----------
    bars_1m : pd.DataFrame
        Must contain columns: timestamp, open, high, low, close, volume,
        buy_volume, sell_volume, trade_count

    Returns
    -------
    pd.DataFrame
        Original columns plus 32 feature columns. NaN warmup rows dropped.
    """
    df = bars_1m.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ------------------------------------------------------------------
    # Returns (6)
    # ------------------------------------------------------------------
    for n, name in [(1, "ret_1m"), (5, "ret_5m"), (10, "ret_10m"),
                    (15, "ret_15m"), (30, "ret_30m"), (60, "ret_60m")]:
        df[name] = close.pct_change(n)

    # ------------------------------------------------------------------
    # EMAs (5)
    # ------------------------------------------------------------------
    df["ema_9"] = close.ewm(span=9, adjust=False).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["ema_9_slope"] = df["ema_9"].pct_change()
    df["ema_cross"] = (df["ema_9"] - df["ema_21"]) / close

    # ------------------------------------------------------------------
    # Volatility (4)
    # ------------------------------------------------------------------
    ret_1m = df["ret_1m"]
    df["volatility_5m"] = ret_1m.rolling(5).std()
    df["volatility_15m"] = ret_1m.rolling(15).std()
    df["volatility_60m"] = ret_1m.rolling(60).std()
    df["vol_ratio"] = df["volatility_5m"] / df["volatility_60m"]
    df["vol_ratio"] = df["vol_ratio"].replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # RSI (standard 14-period)
    # ------------------------------------------------------------------
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = pd.Series(100 - (100 / (1 + rs)), index=close.index)
    # Handle zero-loss regimes: RSI=100 when avg_loss=0 and avg_gain>0, else 50
    rsi = rsi.where(avg_loss > 0, pd.Series(np.where(avg_gain > 0, 100.0, 50.0), index=close.index))
    df["rsi_14"] = rsi

    # ------------------------------------------------------------------
    # Stochastic %K / %D (14-period)
    # ------------------------------------------------------------------
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low_14) / (high_14 - low_14).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ------------------------------------------------------------------
    # VWAP deviation (cumulative per-day VWAP)
    # ------------------------------------------------------------------
    ts = pd.to_datetime(df["timestamp"], utc=True)
    day_group = ts.dt.date
    cum_vol_price = (close * volume).groupby(day_group).cumsum()
    cum_vol = volume.groupby(day_group).cumsum()
    vwap = cum_vol_price / cum_vol.replace(0, np.nan)
    df["vwap_deviation"] = (close - vwap) / vwap.replace(0, np.nan)

    # ------------------------------------------------------------------
    # ATR (14-period rolling)
    # ------------------------------------------------------------------
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / close

    # ------------------------------------------------------------------
    # Donchian (20-period)
    # ------------------------------------------------------------------
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    denom = (high_20 - low_20).replace(0, np.nan)
    df["donchian_position"] = (close - low_20) / denom
    df["donchian_width"] = (high_20 - low_20) / close

    # ------------------------------------------------------------------
    # Volume features (3)
    # ------------------------------------------------------------------
    vol_sma_20 = volume.rolling(20).mean()
    df["volume_sma_ratio"] = volume / vol_sma_20.replace(0, np.nan)

    buy_vol = df["buy_volume"]
    sell_vol = df["sell_volume"]
    df["taker_imbalance"] = (buy_vol - sell_vol) / volume.replace(0, np.nan)
    df["buy_sell_ratio"] = buy_vol / sell_vol.replace(0, np.nan)

    # ------------------------------------------------------------------
    # Calendar features (6)
    # ------------------------------------------------------------------
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek.astype(float)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # US session: 14:30 - 21:00 UTC
    minutes_utc = ts.dt.hour * 60 + ts.dt.minute
    df["is_us_session"] = ((minutes_utc >= 14 * 60 + 30) & (minutes_utc < 21 * 60)).astype(int)

    # Asia session: 00:00 - 06:30 UTC
    df["is_asia_session"] = ((minutes_utc >= 0) & (minutes_utc < 6 * 60 + 30)).astype(int)

    # ------------------------------------------------------------------
    # Drop warmup NaN rows
    # ------------------------------------------------------------------
    df = df.dropna(subset=HONEST_FEATURE_NAMES).reset_index(drop=True)

    return df
