import math

import pandas as pd


class TechnicalFeatures:
    def __init__(self):
        pass

    def compute(self, bars_1m: pd.DataFrame) -> dict[str, float]:
        nan = float("nan")
        nan_dict = {
            "ema_9": nan,
            "ema_21": nan,
            "ema_9_slope": nan,
            "ema_21_slope": nan,
            "ema_cross": nan,
            "vwap": nan,
            "vwap_deviation": nan,
            "atr_14": nan,
            "rsi_14": nan,
            "stoch_k": nan,
            "stoch_d": nan,
            "donchian_high_20": nan,
            "donchian_low_20": nan,
            "donchian_position": nan,
            "volume_sma_ratio": nan,
            "momentum_5m": nan,
            "momentum_10m": nan,
        }

        if bars_1m is None or len(bars_1m) < 30:
            return nan_dict

        close = bars_1m["close"].astype(float)
        high = bars_1m["high"].astype(float)
        low = bars_1m["low"].astype(float)
        volume = bars_1m["volume"].astype(float)

        result: dict[str, float] = {}

        # --- EMA ---
        ema_9 = close.ewm(span=9, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()

        result["ema_9"] = float(ema_9.iloc[-1])
        result["ema_21"] = float(ema_21.iloc[-1])

        if len(ema_9) >= 2 and ema_9.iloc[-2] != 0:
            result["ema_9_slope"] = (ema_9.iloc[-1] - ema_9.iloc[-2]) / ema_9.iloc[-2]
        else:
            result["ema_9_slope"] = nan

        if len(ema_21) >= 2 and ema_21.iloc[-2] != 0:
            result["ema_21_slope"] = (ema_21.iloc[-1] - ema_21.iloc[-2]) / ema_21.iloc[-2]
        else:
            result["ema_21_slope"] = nan

        result["ema_cross"] = float(ema_9.iloc[-1] - ema_21.iloc[-1])

        # --- VWAP ---
        cum_vol = volume.cumsum()
        cum_pv = (close * volume).cumsum()
        vwap = cum_pv / cum_vol.replace(0, nan)

        result["vwap"] = float(vwap.iloc[-1]) if not math.isnan(vwap.iloc[-1]) else nan

        if not math.isnan(vwap.iloc[-1]) and vwap.iloc[-1] != 0:
            result["vwap_deviation"] = (close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]
        else:
            result["vwap_deviation"] = nan

        # --- ATR (14) ---
        tr_high_low = high - low
        tr_high_prev = (high - close.shift(1)).abs()
        tr_low_prev = (low - close.shift(1)).abs()
        tr = pd.concat([tr_high_low, tr_high_prev, tr_low_prev], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14).mean()
        result["atr_14"] = float(atr_14.iloc[-1]) if not pd.isna(atr_14.iloc[-1]) else nan

        # --- RSI (14) ---
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        last_avg_gain = avg_gain.iloc[-1]
        last_avg_loss = avg_loss.iloc[-1]
        if pd.isna(last_avg_gain) or pd.isna(last_avg_loss):
            result["rsi_14"] = nan
        elif last_avg_loss == 0:
            result["rsi_14"] = 100.0 if last_avg_gain > 0 else 50.0
        else:
            rs = last_avg_gain / last_avg_loss
            result["rsi_14"] = float(100.0 - 100.0 / (1.0 + rs))

        # --- Stochastic (14, 3) ---
        low_14 = low.rolling(window=14).min()
        high_14 = high.rolling(window=14).max()
        denom = high_14 - low_14
        stoch_k = 100.0 * (close - low_14) / denom.replace(0, nan)
        stoch_d = stoch_k.rolling(window=3).mean()

        result["stoch_k"] = float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else nan
        result["stoch_d"] = float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else nan

        # --- Donchian Channel (20) ---
        donchian_high = high.rolling(window=20).max()
        donchian_low = low.rolling(window=20).min()
        result["donchian_high_20"] = float(donchian_high.iloc[-1]) if not pd.isna(donchian_high.iloc[-1]) else nan
        result["donchian_low_20"] = float(donchian_low.iloc[-1]) if not pd.isna(donchian_low.iloc[-1]) else nan

        dh = donchian_high.iloc[-1]
        dl = donchian_low.iloc[-1]
        if not pd.isna(dh) and not pd.isna(dl) and (dh - dl) != 0:
            result["donchian_position"] = (close.iloc[-1] - dl) / (dh - dl)
        else:
            result["donchian_position"] = nan

        # --- Volume SMA ratio ---
        vol_sma_20 = volume.rolling(window=20).mean()
        if not pd.isna(vol_sma_20.iloc[-1]) and vol_sma_20.iloc[-1] != 0:
            result["volume_sma_ratio"] = float(volume.iloc[-1]) / float(vol_sma_20.iloc[-1])
        else:
            result["volume_sma_ratio"] = nan

        # --- Momentum ---
        if len(close) >= 5 and close.iloc[-5] != 0:
            result["momentum_5m"] = close.iloc[-1] / close.iloc[-5] - 1.0
        else:
            result["momentum_5m"] = nan

        if len(close) >= 10 and close.iloc[-10] != 0:
            result["momentum_10m"] = close.iloc[-1] / close.iloc[-10] - 1.0
        else:
            result["momentum_10m"] = nan

        return result
