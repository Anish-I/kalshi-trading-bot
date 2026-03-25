"""
Aggregates raw trades into 5-second and 1-minute OHLCV bars.

Tracks buy/sell volume separately for order flow analysis.
"""

import logging
from collections import deque
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BarAggregator:
    """Builds time-based bars from a stream of individual trades."""

    def __init__(self, bar_interval_seconds: int = 5):
        self.bar_interval = bar_interval_seconds

        self.current_bar: Optional[dict] = None
        self.bars_5s: deque = deque(maxlen=2880)   # 4 hours of 5s bars
        self.bars_1m: deque = deque(maxlen=240)     # 4 hours of 1m bars
        self.current_1m_bar: Optional[dict] = None

    def _floor_to_interval(self, timestamp_ms: int, interval_seconds: int) -> int:
        """Floor a millisecond timestamp to the nearest interval boundary (in ms)."""
        interval_ms = interval_seconds * 1000
        return (timestamp_ms // interval_ms) * interval_ms

    def add_trade(
        self,
        price: float,
        qty: float,
        is_buyer_maker: bool,
        timestamp_ms: int,
    ) -> Optional[dict]:
        """
        Add a trade to the current 5-second bar.

        Args:
            price: Trade price.
            qty: Trade quantity.
            is_buyer_maker: True if the buyer is the maker (i.e. a sell taker trade).
            timestamp_ms: Trade timestamp in milliseconds.

        Returns:
            A finalized 5s bar dict if a bar boundary was crossed, else None.
        """
        bar_ts = self._floor_to_interval(timestamp_ms, self.bar_interval)

        # Buy taker volume: when is_buyer_maker is False, the buyer is the taker
        buy_vol = qty if not is_buyer_maker else 0.0
        sell_vol = qty if is_buyer_maker else 0.0

        if self.current_bar is None:
            # Start the very first bar
            self.current_bar = {
                "timestamp": bar_ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": qty,
                "buy_volume": buy_vol,
                "sell_volume": sell_vol,
                "trade_count": 1,
            }
            return None

        if bar_ts == self.current_bar["timestamp"]:
            # Same bar — update OHLCV
            self.current_bar["high"] = max(self.current_bar["high"], price)
            self.current_bar["low"] = min(self.current_bar["low"], price)
            self.current_bar["close"] = price
            self.current_bar["volume"] += qty
            self.current_bar["buy_volume"] += buy_vol
            self.current_bar["sell_volume"] += sell_vol
            self.current_bar["trade_count"] += 1
            return None

        # New bar boundary crossed — finalize the current bar
        completed_bar = self.current_bar.copy()
        self.bars_5s.append(completed_bar)
        logger.debug(
            "5s bar completed: ts=%d close=%.2f vol=%.6f trades=%d",
            completed_bar["timestamp"],
            completed_bar["close"],
            completed_bar["volume"],
            completed_bar["trade_count"],
        )

        # Check if we also completed a 1m bar
        completed_1m = self._check_1m_bar(completed_bar)
        if completed_1m is not None:
            self.bars_1m.append(completed_1m)
            logger.debug(
                "1m bar completed: ts=%d close=%.2f vol=%.6f",
                completed_1m["timestamp"],
                completed_1m["close"],
                completed_1m["volume"],
            )

        # Start new bar
        self.current_bar = {
            "timestamp": bar_ts,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": qty,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "trade_count": 1,
        }

        return completed_bar

    def _check_1m_bar(self, completed_5s_bar: dict) -> Optional[dict]:
        """
        Check if a 1-minute boundary was crossed and aggregate into a 1m bar.

        Args:
            completed_5s_bar: The just-completed 5-second bar.

        Returns:
            A completed 1m bar dict if a minute boundary was crossed, else None.
        """
        one_minute_ms = 60_000
        bar_minute = self._floor_to_interval(completed_5s_bar["timestamp"], 60)

        if self.current_1m_bar is None:
            # First 1m bar — start tracking
            self.current_1m_bar = {
                "timestamp": bar_minute,
                "open": completed_5s_bar["open"],
                "high": completed_5s_bar["high"],
                "low": completed_5s_bar["low"],
                "close": completed_5s_bar["close"],
                "volume": completed_5s_bar["volume"],
                "buy_volume": completed_5s_bar["buy_volume"],
                "sell_volume": completed_5s_bar["sell_volume"],
                "trade_count": completed_5s_bar["trade_count"],
            }
            return None

        if bar_minute == self.current_1m_bar["timestamp"]:
            # Same minute — accumulate
            self.current_1m_bar["high"] = max(self.current_1m_bar["high"], completed_5s_bar["high"])
            self.current_1m_bar["low"] = min(self.current_1m_bar["low"], completed_5s_bar["low"])
            self.current_1m_bar["close"] = completed_5s_bar["close"]
            self.current_1m_bar["volume"] += completed_5s_bar["volume"]
            self.current_1m_bar["buy_volume"] += completed_5s_bar["buy_volume"]
            self.current_1m_bar["sell_volume"] += completed_5s_bar["sell_volume"]
            self.current_1m_bar["trade_count"] += completed_5s_bar["trade_count"]
            return None

        # New minute — finalize the current 1m bar and start a new one
        completed_1m = self.current_1m_bar.copy()

        self.current_1m_bar = {
            "timestamp": bar_minute,
            "open": completed_5s_bar["open"],
            "high": completed_5s_bar["high"],
            "low": completed_5s_bar["low"],
            "close": completed_5s_bar["close"],
            "volume": completed_5s_bar["volume"],
            "buy_volume": completed_5s_bar["buy_volume"],
            "sell_volume": completed_5s_bar["sell_volume"],
            "trade_count": completed_5s_bar["trade_count"],
        }

        return completed_1m

    def get_bars_5s_df(self) -> pd.DataFrame:
        """Convert the 5-second bars deque to a DataFrame."""
        if not self.bars_5s:
            return pd.DataFrame(columns=[
                "timestamp", "open", "high", "low", "close",
                "volume", "buy_volume", "sell_volume", "trade_count",
            ])
        df = pd.DataFrame(list(self.bars_5s))
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def get_bars_1m_df(self) -> pd.DataFrame:
        """Convert the 1-minute bars deque to a DataFrame."""
        if not self.bars_1m:
            return pd.DataFrame(columns=[
                "timestamp", "open", "high", "low", "close",
                "volume", "buy_volume", "sell_volume", "trade_count",
            ])
        df = pd.DataFrame(list(self.bars_1m))
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df
