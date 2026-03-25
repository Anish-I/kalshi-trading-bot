"""Paper trading mode -- real market data, simulated order execution."""

import asyncio
import logging
import signal
import sys

sys.path.insert(0, ".")

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("paper_trade.log"),
    ],
)
logger = logging.getLogger("paper_trade")


class PaperTradingEngine:
    """Same logic as TradingEngine but logs orders instead of placing them."""

    def __init__(self):
        self.kalshi = KalshiClient()
        self.tracker = KXBTCMarketTracker(self.kalshi)
        self.scheduler = MarketScheduler(self.tracker)

        self.bar_aggregator = BarAggregator(bar_interval_seconds=5)
        self.storage = DataStorage()
        self.latest_orderbook: dict = {"bids": [], "asks": []}

        self.pipeline = FeaturePipeline()

        self.ensemble = EnsembleModel(xgb_weight=1.0, transformer_weight=0.0)
        self._load_model()

        self.risk = RiskManager(
            max_contracts=settings.MAX_POSITION_CONTRACTS,
            daily_loss_limit_cents=settings.DAILY_LOSS_LIMIT_CENTS,
            consecutive_loss_halt=settings.CONSECUTIVE_LOSS_HALT,
        )
        self.positions = PositionManager()

        self.collector = BinanceCollector(
            symbol=settings.BINANCE_SYMBOL,
            on_trade=self._on_trade,
            on_depth=self._on_depth,
            on_kline=self._on_kline,
        )

        self._running = False
        self._cycle_count = 0
        self._last_status_log = 0.0

        # Paper trading stats
        self.paper_orders: list[dict] = []

    def _load_model(self) -> None:
        from pathlib import Path

        model_dir = Path(settings.MODEL_DIR)
        model_files = sorted(model_dir.glob("xgb_*.json")) if model_dir.exists() else []

        if not model_files:
            logger.warning("No trained model found -- predictions disabled")
            return

        latest = model_files[-1]
        xgb = XGBoostDirectionModel()
        xgb.load(str(latest))
        self.ensemble.set_models(xgb)
        logger.info("Loaded model: %s", latest.name)

    async def run(self) -> None:
        self._running = True
        logger.info("=== Paper trading engine starting ===")

        collector_task = asyncio.create_task(self.collector.start())

        try:
            while self._running:
                await self._inference_cycle()
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()
            collector_task.cancel()
            try:
                await collector_task
            except asyncio.CancelledError:
                pass

    async def _inference_cycle(self) -> None:
        self._cycle_count += 1

        await self.scheduler.update()
        phase = self.scheduler.get_phase()

        await self._check_settlements()

        if self.scheduler.should_trade():
            await self._try_paper_trade()

        now = asyncio.get_event_loop().time()
        if now - self._last_status_log > 60:
            self._log_status(phase)
            self._last_status_log = now

    async def _try_paper_trade(self) -> None:
        """Simulate a trade -- compute everything but don't hit the API."""
        market = self.scheduler.current_market
        if market is None:
            return

        ticker = market.get("ticker", "")

        bars_5s_df = self.bar_aggregator.get_bars_5s_df()
        bars_1m_df = self.bar_aggregator.get_bars_1m_df()

        if len(bars_5s_df) < 10:
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

        try:
            feature_df = pd.DataFrame([features])
            if self.ensemble.xgb_model is not None:
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
            return

        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            logger.warning("Risk blocked paper trade: %s", reason)
            return

        side = "yes" if direction == 1 else "no"
        limit_price = max(1, min(99, int(round(confidence * 100))))

        # PAPER: log instead of placing
        paper_order = {
            "ticker": ticker,
            "side": side,
            "contracts": 1,
            "limit_price": limit_price,
            "confidence": confidence,
            "direction": direction,
        }
        self.paper_orders.append(paper_order)
        logger.info(
            "[PAPER] Would place: %s %s x1 @ %dc (conf=%.4f)",
            ticker, side, limit_price, confidence,
        )

        self.positions.open_position(ticker, side, 1, limit_price)
        self.scheduler.mark_traded(ticker)

    async def _check_settlements(self) -> None:
        for ticker, pos in list(self.positions.positions.items()):
            try:
                market_data = self.kalshi.get_market(ticker)
                status = market_data.get("status", "")
                result = market_data.get("result", "")

                if status in ("settled", "finalized") and result:
                    settled_yes = result.lower() == "yes"
                    pnl = self.positions.close_position(ticker, settled_yes)
                    self.risk.record_trade(pnl, {
                        "ticker": ticker,
                        "settled_yes": settled_yes,
                        "paper": True,
                    })
                    logger.info("[PAPER] Settlement: %s -> %s  pnl=%+dc", ticker, result, pnl)
            except Exception:
                logger.debug("Could not check settlement for %s", ticker)

    def _log_status(self, phase: str) -> None:
        market_ticker = (
            self.scheduler.current_market.get("ticker", "?")
            if self.scheduler.current_market else "none"
        )
        logger.info(
            "[PAPER] STATUS  market=%s  phase=%s  positions=%d  daily_pnl=%+dc  "
            "orders=%d  consec_losses=%d",
            market_ticker, phase,
            len(self.positions.positions),
            self.risk.daily_pnl_cents,
            len(self.paper_orders),
            self.risk.consecutive_losses,
        )

    async def _on_trade(self, trade: dict) -> None:
        self.bar_aggregator.add_trade(
            price=trade["price"],
            qty=trade["qty"],
            is_buyer_maker=trade["is_buyer_maker"],
            timestamp_ms=trade["timestamp_ms"],
        )

    async def _on_depth(self, orderbook: dict) -> None:
        self.latest_orderbook = orderbook
        self.pipeline.update_orderbook(orderbook)

    async def _on_kline(self, kline: dict) -> None:
        self.pipeline.update_1m_bar(kline)

    async def stop(self) -> None:
        self._running = False
        await self.collector.stop()

        trade_log = self.positions.get_trade_log_df()
        if not trade_log.empty:
            logger.info("[PAPER] Trade log:\n%s", trade_log.to_string(index=False))

        logger.info(
            "[PAPER] Session complete: %d orders, realized P&L=%+dc, %d trades settled",
            len(self.paper_orders),
            self.positions.realized_pnl_cents,
            len(self.positions.trade_log),
        )


async def main() -> None:
    engine = PaperTradingEngine()

    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    engine_task = asyncio.create_task(engine.run())

    try:
        await stop_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        await engine.stop()
        engine_task.cancel()
        try:
            await engine_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Paper trading interrupted")
