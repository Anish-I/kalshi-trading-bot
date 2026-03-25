import math
from collections import deque
from datetime import datetime, timezone

import pandas as pd


class MultiTimeframeFeatures:
    def __init__(self):
        self.bars_15m: deque[dict] = deque(maxlen=32)
        self.bars_1h: deque[dict] = deque(maxlen=24)
        self.bars_4h: deque[dict] = deque(maxlen=12)

        self._current_15m: dict | None = None
        self._current_1h: dict | None = None
        self._current_4h: dict | None = None

    def update(self, bar_1m: dict) -> None:
        ts = bar_1m["timestamp"]
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts, tz=timezone.utc)
        elif isinstance(ts, str):
            ts = pd.Timestamp(ts).to_pydatetime().replace(tzinfo=timezone.utc)
        elif isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

        self._aggregate(bar_1m, ts, 15, self.bars_15m, "_current_15m")
        self._aggregate(bar_1m, ts, 60, self.bars_1h, "_current_1h")
        self._aggregate(bar_1m, ts, 240, self.bars_4h, "_current_4h")

    def _aggregate(
        self,
        bar_1m: dict,
        ts: datetime,
        period_minutes: int,
        completed: deque,
        current_attr: str,
    ) -> None:
        # Compute the boundary slot for this timestamp
        epoch_min = int(ts.timestamp()) // 60
        slot = epoch_min // period_minutes

        current = getattr(self, current_attr)

        if current is None:
            # Start a new bar
            setattr(
                self,
                current_attr,
                {
                    "slot": slot,
                    "timestamp": ts,
                    "open": bar_1m["open"],
                    "high": bar_1m["high"],
                    "low": bar_1m["low"],
                    "close": bar_1m["close"],
                    "volume": bar_1m["volume"],
                },
            )
            return

        if slot != current["slot"]:
            # Current bar completed; store it and start a new one
            completed.append(current)
            setattr(
                self,
                current_attr,
                {
                    "slot": slot,
                    "timestamp": ts,
                    "open": bar_1m["open"],
                    "high": bar_1m["high"],
                    "low": bar_1m["low"],
                    "close": bar_1m["close"],
                    "volume": bar_1m["volume"],
                },
            )
        else:
            # Same slot — extend the current bar
            current["high"] = max(current["high"], bar_1m["high"])
            current["low"] = min(current["low"], bar_1m["low"])
            current["close"] = bar_1m["close"]
            current["volume"] = current["volume"] + bar_1m["volume"]

    def compute(self) -> dict[str, float]:
        nan = float("nan")
        result: dict[str, float] = {}

        slopes: dict[str, float] = {}

        for label, bars_deque in [("15m", self.bars_15m), ("1h", self.bars_1h), ("4h", self.bars_4h)]:
            if len(bars_deque) < 2:
                result[f"rsi_{label}"] = nan
                result[f"ema_slope_{label}"] = nan
                result[f"bb_position_{label}"] = nan
                slopes[label] = nan
                continue

            df = pd.DataFrame(list(bars_deque))
            close = df["close"].astype(float)

            # RSI
            result[f"rsi_{label}"] = self._rsi(close, min(14, len(close) - 1))

            # EMA slope
            ema_slope = self._ema_slope(close, min(9, len(close)))
            result[f"ema_slope_{label}"] = ema_slope
            slopes[label] = ema_slope

            # Bollinger Band position
            result[f"bb_position_{label}"] = self._bb_position(close, min(20, len(close)))

        # Cross-timeframe divergence: 15m vs 1h
        s15 = slopes.get("15m", nan)
        s1h = slopes.get("1h", nan)
        if math.isnan(s15) or math.isnan(s1h):
            result["cross_tf_divergence_15m_1h"] = nan
        else:
            sign_15 = _sign(s15)
            sign_1h = _sign(s1h)
            if sign_15 > 0 and sign_1h < 0:
                result["cross_tf_divergence_15m_1h"] = 1.0
            elif sign_15 < 0 and sign_1h > 0:
                result["cross_tf_divergence_15m_1h"] = -1.0
            else:
                result["cross_tf_divergence_15m_1h"] = 0.0

        # Higher timeframe trend
        s4h = slopes.get("4h", nan)
        if math.isnan(s1h) or math.isnan(s4h):
            result["higher_tf_trend"] = nan
        else:
            result["higher_tf_trend"] = float(_sign(s1h) + _sign(s4h))

        return result

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> float:
        if len(close) < period + 1:
            return float("nan")
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        last_loss = avg_loss.iloc[-1]
        if last_loss == 0:
            return 100.0
        rs = avg_gain.iloc[-1] / last_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    @staticmethod
    def _ema_slope(close: pd.Series, span: int) -> float:
        if len(close) < 2:
            return float("nan")
        ema = close.ewm(span=span, adjust=False).mean()
        if ema.iloc[-2] == 0:
            return float("nan")
        return float((ema.iloc[-1] - ema.iloc[-2]) / ema.iloc[-2])

    @staticmethod
    def _bb_position(close: pd.Series, window: int) -> float:
        if len(close) < window:
            return float("nan")
        sma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        upper = sma + 2.0 * std
        lower = sma - 2.0 * std
        denom = upper.iloc[-1] - lower.iloc[-1]
        if pd.isna(denom) or denom == 0:
            return float("nan")
        return float((close.iloc[-1] - lower.iloc[-1]) / denom)


def _sign(x: float) -> int:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0
