"""
REST client for historical Binance data using ccxt.

Provides methods to fetch klines, funding rates, and recent trades
as pandas DataFrames for backtesting and feature engineering.
"""

import logging
from datetime import datetime
from typing import Optional

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceREST:
    """Synchronous Binance REST client backed by ccxt."""

    def __init__(self):
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
        })

    def fetch_klines(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        since: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV klines from Binance spot.

        Args:
            symbol: Trading pair in ccxt format (e.g. "BTC/USDT").
            timeframe: Candle interval (e.g. "1m", "5m", "1h").
            since: Start time as Unix timestamp in milliseconds.
            limit: Maximum number of candles to fetch.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=limit
            )
        except ccxt.BaseError:
            logger.exception("Failed to fetch klines for %s", symbol)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        if not ohlcv:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def fetch_funding_rate(self, symbol: str = "BTC/USDT:USDT") -> list[dict]:
        """
        Fetch recent funding rates from Binance futures.

        Args:
            symbol: Perpetual futures symbol in ccxt format.

        Returns:
            List of funding rate dicts with keys: timestamp, symbol, fundingRate.
        """
        try:
            # Use the futures-capable exchange
            futures_exchange = ccxt.binance({
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })
            rates = futures_exchange.fetch_funding_rate(symbol)
            return [rates] if isinstance(rates, dict) else rates
        except ccxt.BaseError:
            logger.exception("Failed to fetch funding rate for %s", symbol)
            return []

    def fetch_trades(
        self,
        symbol: str = "BTC/USDT",
        since: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch recent trades from Binance spot.

        Args:
            symbol: Trading pair in ccxt format.
            since: Start time as Unix timestamp in milliseconds.
            limit: Maximum number of trades to fetch.

        Returns:
            DataFrame with columns: timestamp, price, qty, is_buyer_maker.
        """
        try:
            trades = self.exchange.fetch_trades(symbol, since=since, limit=limit)
        except ccxt.BaseError:
            logger.exception("Failed to fetch trades for %s", symbol)
            return pd.DataFrame(columns=["timestamp", "price", "qty", "is_buyer_maker"])

        if not trades:
            return pd.DataFrame(columns=["timestamp", "price", "qty", "is_buyer_maker"])

        records = []
        for t in trades:
            records.append({
                "timestamp": pd.to_datetime(t["timestamp"], unit="ms", utc=True),
                "price": float(t["price"]),
                "qty": float(t["amount"]),
                "is_buyer_maker": t["side"] == "sell",  # sell-side trade = buyer is maker
            })

        return pd.DataFrame(records)
