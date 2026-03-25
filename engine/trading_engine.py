"""Core trading engine -- orchestrates data collection, inference, and execution."""

import asyncio
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config.settings import settings
from data.binance_ws import BinanceCollector
from data.bar_aggregator import BarAggregator
from data.storage import DataStorage
from features.pipeline import FeaturePipeline
from kalshi.client import KalshiClient
from kalshi.market_discovery import KXBTCMarketTracker
from models.xgboost_model import XGBoostDirectionModel
from models.ensemble import EnsembleModel

from engine.risk import RiskManager
from engine.position_manager import PositionManager
from engine.scheduler import MarketScheduler

logger = logging.getLogger(__name__)


class TradingEngine:
    """Full trading loop: data -> features -> model -> order -> settle."""

    def __init__(self):
        # Kalshi
        self.kalshi = KalshiClient()
        self.tracker = KXBTCMarketTracker(self.kalshi)
        self.scheduler = MarketScheduler(self.tracker)

        # Data
        self.bar_aggregator = BarAggregator(bar_interval_seconds=5)
        self.storage = DataStorage()
        self.latest_orderbook: dict = {"bids": [], "asks": []}

        # Features
        self.pipeline = FeaturePipeline()

        # Model
        self.ensemble = EnsembleModel(xgb_weight=1.0, transformer_weight=0.0)
        self._load_model()

        # Risk & positions
        self.risk = RiskManager(
            max_contracts=settings.MAX_POSITION_CONTRACTS,
            daily_loss_limit_cents=settings.DAILY_LOSS_LIMIT_CENTS,
            consecutive_loss_halt=settings.CONSECUTIVE_LOSS_HALT,
        )
        self.positions = PositionManager()

        # Binance collector (callbacks wired up later in run())
        self.collector = BinanceCollector(
            symbol=settings.BINANCE_SYMBOL,
            on_trade=self._on_trade,
            on_depth=self._on_depth,
            on_kline=self._on_kline,
        )

        self._running = False
        self._cycle_count = 0
        self._last_status_log = 0.0

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the latest XGBoost model from MODEL_DIR."""
        from pathlib import Path

        model_dir = Path(settings.MODEL_DIR)
        model_files = sorted(model_dir.glob("xgb_*.json")) if model_dir.exists() else []

        if not model_files:
            logger.warning("No trained model found in %s -- predictions disabled", model_dir)
            return

        latest = model_files[-1]
        xgb = XGBoostDirectionModel()
        xgb.load(str(latest))
        self.ensemble.set_models(xgb)
        logger.info("Loaded model: %s", latest.name)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the trading engine."""
        self._running = True
        logger.info("Trading engine starting")

        # Start the Binance collector in the background
        collector_task = asyncio.create_task(self.collector.start())

        try:
            while self._running:
                await self._inference_cycle()
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("Trading engine cancelled")
        finally:
            await self.stop()
            collector_task.cancel()
            try:
                await collector_task
            except asyncio.CancelledError:
                pass

    async def _inference_cycle(self) -> None:
        """Run one inference + execution cycle."""
        self._cycle_count += 1

        # 1. Update scheduler
        await self.scheduler.update()
        phase = self.scheduler.get_phase()

        # 2. Check for settled positions
        await self._check_settlements()

        # 3. Maybe trade
        if self.scheduler.should_trade():
            await self._try_trade()

        # 4. Periodic status log (every 60s ~ 12 cycles)
        now = asyncio.get_event_loop().time()
        if now - self._last_status_log > 60:
            self._log_status(phase)
            self._last_status_log = now

        # 5. Periodic data save (every 60 cycles ~ 5 min)
        if self._cycle_count % 60 == 0:
            self._save_data()

    async def _try_trade(self) -> None:
        """Compute features, run model, and place an order if confident."""
        market = self.scheduler.current_market
        if market is None:
            return

        ticker = market.get("ticker", "")

        # Build feature vector
        bars_5s_df = self.bar_aggregator.get_bars_5s_df()
        bars_1m_df = self.bar_aggregator.get_bars_1m_df()

        if len(bars_5s_df) < 10:
            logger.debug("Not enough 5s bars (%d), skipping", len(bars_5s_df))
            return

        try:
            features = self.pipeline.compute(
                self.latest_orderbook, bars_5s_df, bars_1m_df,
            )
        except Exception:
            logger.exception("Feature computation failed")
            return

        if not features:
            return

        # Model prediction
        try:
            feature_df = pd.DataFrame([features])
            if self.ensemble.xgb_model is not None:
                # Align columns to model's expected features
                expected = self.ensemble.xgb_model.feature_names
                if expected:
                    for col in expected:
                        if col not in feature_df.columns:
                            feature_df[col] = 0.0
                    feature_df = feature_df[expected]

            direction, confidence = self.ensemble.predict_direction(
                feature_df, confidence_threshold=settings.CONFIDENCE_THRESHOLD,
            )
        except Exception:
            logger.exception("Model prediction failed")
            return

        if direction == 0:
            logger.debug("Model abstained (confidence=%.4f)", confidence)
            return

        # Risk check
        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            logger.warning("Risk blocked trade: %s", reason)
            return

        contracts = 1
        ok, size_reason = self.risk.check_order_size(contracts)
        if not ok:
            logger.warning("Order size blocked: %s", size_reason)
            return

        # Determine side and price
        side = "yes" if direction == 1 else "no"
        limit_price = int(round(confidence * 100))
        limit_price = max(1, min(99, limit_price))

        # Place order
        try:
            logger.info(
                "Placing order: %s %s x%d @ %dc (conf=%.4f)",
                ticker, side, contracts, limit_price, confidence,
            )
            order_resp = self.kalshi.place_order(
                ticker=ticker,
                side=side,
                action="buy",
                count=contracts,
                order_type="limit",
                **({f"{side}_price": limit_price}),
            )
            logger.info("Order response: %s", order_resp)

            # Track position
            self.positions.open_position(ticker, side, contracts, limit_price)
            self.scheduler.mark_traded(ticker)

        except Exception:
            logger.exception("Failed to place order on %s", ticker)

    async def _check_settlements(self) -> None:
        """Check if any open position's market has settled."""
        tickers_to_close = []

        for ticker, pos in list(self.positions.positions.items()):
            try:
                market_data = self.kalshi.get_market(ticker)
                status = market_data.get("status", "")
                result = market_data.get("result", "")

                if status in ("settled", "finalized") and result:
                    settled_yes = result.lower() == "yes"
                    tickers_to_close.append((ticker, settled_yes))
            except Exception:
                logger.debug("Could not check settlement for %s", ticker)

        for ticker, settled_yes in tickers_to_close:
            pnl = self.positions.close_position(ticker, settled_yes)
            self.risk.record_trade(pnl, {"ticker": ticker, "settled_yes": settled_yes})

    def _log_status(self, phase: str) -> None:
        """Log a summary of current engine state."""
        market_ticker = (
            self.scheduler.current_market.get("ticker", "?")
            if self.scheduler.current_market else "none"
        )
        logger.info(
            "STATUS  market=%s  phase=%s  open_positions=%d  daily_pnl=%+dc  "
            "consec_losses=%d  bars_5s=%d  bars_1m=%d  cycle=%d",
            market_ticker,
            phase,
            len(self.positions.positions),
            self.risk.daily_pnl_cents,
            self.risk.consecutive_losses,
            len(self.bar_aggregator.bars_5s),
            len(self.bar_aggregator.bars_1m),
            self._cycle_count,
        )

    def _save_data(self) -> None:
        """Persist collected bars to parquet."""
        try:
            bars_5s = self.bar_aggregator.get_bars_5s_df()
            if not bars_5s.empty:
                self.storage.save_bars(bars_5s, "bars_5s")

            bars_1m = self.bar_aggregator.get_bars_1m_df()
            if not bars_1m.empty:
                self.storage.save_bars(bars_1m, "bars_1m")
        except Exception:
            logger.exception("Failed to save bar data")

    # ------------------------------------------------------------------
    # WebSocket callbacks
    # ------------------------------------------------------------------

    async def _on_trade(self, trade: dict) -> None:
        """Process a Binance trade tick through the bar aggregator."""
        self.bar_aggregator.add_trade(
            price=trade["price"],
            qty=trade["qty"],
            is_buyer_maker=trade["is_buyer_maker"],
            timestamp_ms=trade["timestamp_ms"],
        )

    async def _on_depth(self, orderbook: dict) -> None:
        """Store the latest orderbook snapshot and update the feature pipeline."""
        self.latest_orderbook = orderbook
        self.pipeline.update_orderbook(orderbook)

    async def _on_kline(self, kline: dict) -> None:
        """Update the feature pipeline with a closed 1-minute bar."""
        self.pipeline.update_1m_bar(kline)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Gracefully stop the engine."""
        self._running = False
        await self.collector.stop()
        self._save_data()

        trade_log = self.positions.get_trade_log_df()
        if not trade_log.empty:
            logger.info(
                "Final trade log:\n%s",
                trade_log.to_string(index=False),
            )
        logger.info(
            "Engine stopped. Realized P&L: %+dc across %d trades",
            self.positions.realized_pnl_cents,
            len(self.positions.trade_log),
        )
