"""Production entry point for the Kalshi Bitcoin trading bot."""

import asyncio
import logging
import signal
import sys
from pathlib import Path

sys.path.insert(0, ".")

from config.settings import settings
from engine.trading_engine import TradingEngine

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

log_dir = Path(settings.DATA_DIR) / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "live_trade.log"),
    ],
)
logger = logging.getLogger("live_trade")


async def main() -> None:
    logger.info("=== Kalshi Live Trading Bot starting ===")
    logger.info("Config: max_contracts=%d  daily_loss_limit=%dc  confidence=%.2f",
                settings.MAX_POSITION_CONTRACTS,
                settings.DAILY_LOSS_LIMIT_CENTS,
                settings.CONFIDENCE_THRESHOLD)

    engine = TradingEngine()

    stop_event = asyncio.Event()

    def _request_shutdown() -> None:
        logger.info("Shutdown signal received -- stopping gracefully")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except NotImplementedError:
            # Windows fallback
            pass

    engine_task = asyncio.create_task(engine.run())

    try:
        await stop_event.wait()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    finally:
        logger.info("Stopping engine...")
        await engine.stop()
        engine_task.cancel()
        try:
            await engine_task
        except asyncio.CancelledError:
            pass

    logger.info("=== Kalshi Live Trading Bot stopped ===")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Live trading interrupted")
