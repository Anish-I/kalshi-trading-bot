#!/usr/bin/env python3
"""Kalshi Weather Trading Bot — scans for mispriced weather markets and trades.

Usage:
    python scripts/weather_trade.py              # continuous loop
    python scripts/weather_trade.py --once       # single scan + trade cycle
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

from weather.trader import WeatherTrader
from config.settings import settings

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

log_dir = Path(settings.DATA_DIR) / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "weather_trade.log"),
    ],
)
logger = logging.getLogger("weather_trade")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    logger.info("Shutdown signal received (signal %d) — finishing current cycle …", signum)
    _shutdown_requested = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_cycle(trader: WeatherTrader) -> None:
    """Execute one full scan → settle → trade → status cycle."""
    cycle_start = datetime.now(timezone.utc).isoformat()
    logger.info("=== Cycle start: %s ===", cycle_start)

    # 1. Check settlements on existing positions
    try:
        settled = trader.check_settlements()
        if settled:
            logger.info("Settled %d positions this cycle", len(settled))
    except Exception:
        logger.error("Settlement check failed", exc_info=True)

    # 2. Scan for opportunities
    try:
        opportunities = trader.run_scan()
    except Exception:
        logger.error("Market scan failed", exc_info=True)
        opportunities = []

    # 3. Execute trades if opportunities exist
    if opportunities:
        try:
            executed = trader.execute_trades(opportunities)
            if executed:
                logger.info("Placed %d new trades", len(executed))
        except Exception:
            logger.error("Trade execution failed", exc_info=True)

    # 4. Print status
    try:
        status = trader.get_status()
        logger.info(
            "STATUS  balance=$%s  open=%d  daily_pnl=%+dc  trades_today=%d  "
            "can_trade=%s  risk=%s",
            f"{status['balance_usd']:.2f}" if status["balance_usd"] is not None else "N/A",
            status["open_position_count"],
            status["daily_pnl_cents"],
            status["trades_today"],
            status["can_trade"],
            status["risk_status"],
        )
    except Exception:
        logger.error("Status report failed", exc_info=True)

    logger.info("=== Cycle complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi Weather Trading Bot")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan-trade cycle then exit",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Override scan interval in minutes (default from settings)",
    )
    args = parser.parse_args()

    interval_minutes = args.interval or settings.WEATHER_SCAN_INTERVAL_MINUTES
    interval_seconds = interval_minutes * 60

    logger.info("=" * 60)
    logger.info("  Kalshi Weather Trading Bot")
    logger.info("  Scan interval: %d minutes", interval_minutes)
    logger.info("  Edge threshold: %.0f%%", settings.WEATHER_EDGE_THRESHOLD * 100)
    logger.info("  Max contracts per trade: %d", settings.WEATHER_MAX_CONTRACTS)
    logger.info("=" * 60)

    try:
        trader = WeatherTrader()
    except Exception:
        logger.critical("Failed to initialise WeatherTrader", exc_info=True)
        sys.exit(1)

    if args.once:
        run_cycle(trader)
        return

    # Continuous loop
    while not _shutdown_requested:
        try:
            run_cycle(trader)
        except Exception:
            logger.error("Unhandled error in trading cycle", exc_info=True)

        if _shutdown_requested:
            break

        logger.info("Sleeping %d minutes until next scan …", interval_minutes)
        # Sleep in small increments to respond to shutdown signals quickly
        for _ in range(interval_seconds):
            if _shutdown_requested:
                break
            time.sleep(1)

    # Final status
    logger.info("Shutdown complete. Final status:")
    try:
        status = trader.get_status()
        logger.info("  Balance: $%s", status.get("balance_usd", "N/A"))
        logger.info("  Open positions: %d", status["open_position_count"])
        logger.info("  Daily P&L: %+dc", status["daily_pnl_cents"])
        logger.info("  Total realised P&L: %+dc", status["realized_pnl_cents"])
    except Exception:
        logger.error("Could not fetch final status", exc_info=True)


if __name__ == "__main__":
    main()
