"""
Crypto watcher: monitors BTC 15-min markets, bets small (1-2 contracts)
only when strong edge detected. Runs alongside data collector.

Run: python scripts/crypto_watch.py
"""
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import httpx

from kalshi.client import KalshiClient
from kalshi.market_discovery import KXBTCMarketTracker
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("crypto_watch.log"),
    ],
)
log = logging.getLogger("crypto_watch")

# Config
MAX_CONTRACTS = 2
MIN_EDGE = 0.03  # 3% edge minimum
MIN_SIGNALS = 3  # need 3/5 signals aligned
MAX_ENTRY_PRICE = 0.52  # don't pay more than 52c
SCAN_INTERVAL = 30  # check every 30 seconds
DAILY_LOSS_LIMIT = 500  # 5 dollars in cents
traded_tickers = set()
daily_pnl = 0
trade_log = []


def get_btc_price():
    try:
        resp = httpx.get("https://api.exchange.coinbase.com/products/BTC-USD/ticker", timeout=5)
        return float(resp.json()["price"])
    except Exception:
        return None


def load_latest_features():
    feat_dir = Path(settings.DATA_DIR) / "features"
    if not feat_dir.exists():
        return None
    files = sorted(feat_dir.glob("*.parquet"))
    if not files:
        return None
    try:
        df = pd.read_parquet(files[-1])
        if len(df) == 0:
            return None
        return df.iloc[-1]
    except Exception:
        return None


def score_signals(features):
    """Score bullish/bearish signals from features. Returns (bull, bear, reasons)."""
    bull = bear = 0
    reasons = []

    rsi = features.get("rsi_14", 50)
    imb = features.get("top_imbalance", 0)
    m5 = features.get("momentum_5m", 0)
    m10 = features.get("momentum_10m", 0)
    ema_s = features.get("ema_9_slope", 0)

    if rsi > 55:
        bull += 1; reasons.append("RSI>55")
    elif rsi < 45:
        bear += 1; reasons.append("RSI<45")

    if imb > 0.3:
        bull += 1; reasons.append("OB+")
    elif imb < -0.3:
        bear += 1; reasons.append("OB-")

    if m5 > 0.0003:
        bull += 1; reasons.append("Mom5+")
    elif m5 < -0.0003:
        bear += 1; reasons.append("Mom5-")

    if m10 > 0.0005:
        bull += 1; reasons.append("Mom10+")
    elif m10 < -0.0005:
        bear += 1; reasons.append("Mom10-")

    if ema_s > 0.00005:
        bull += 1; reasons.append("EMA+")
    elif ema_s < -0.00005:
        bear += 1; reasons.append("EMA-")

    return bull, bear, reasons


def try_trade(c, market, features):
    """Attempt a trade if strong edge exists. Returns True if traded."""
    global daily_pnl, traded_tickers

    ticker = market["ticker"]
    if ticker in traded_tickers:
        return False

    if daily_pnl <= -DAILY_LOSS_LIMIT:
        log.warning("Daily loss limit hit ($%.2f). Sitting out.", daily_pnl / 100)
        return False

    remaining = KXBTCMarketTracker(c).get_market_time_remaining(market)
    if remaining < 120 or remaining > 780:
        return False  # Only trade 2-13 min before close

    yes_ask = float(market.get("yes_ask_dollars", 0))
    no_ask = float(market.get("no_ask_dollars", 0))

    bull, bear, reasons = score_signals(features)

    # Need strong signal alignment
    if bull >= MIN_SIGNALS and bear == 0 and yes_ask <= MAX_ENTRY_PRICE:
        side = "yes"
        price = yes_ask
        conf = 0.52 + bull * 0.04
        edge = conf - price
    elif bear >= MIN_SIGNALS and bull == 0 and no_ask <= MAX_ENTRY_PRICE:
        side = "no"
        price = no_ask
        conf = 0.52 + bear * 0.04
        edge = conf - price
    else:
        return False

    if edge < MIN_EDGE:
        return False

    price_cents = int(price * 100)
    contracts = 1 if edge < 0.08 else MAX_CONTRACTS

    log.info(
        "TRADING %s %s @ %dc x%d | bull=%d bear=%d edge=%.0f%% | %s",
        side.upper(), ticker, price_cents, contracts, bull, bear, edge * 100, reasons,
    )

    try:
        if side == "yes":
            order = c.place_order(ticker, side="yes", action="buy", count=contracts,
                                  order_type="limit", yes_price=price_cents)
        else:
            order = c.place_order(ticker, side="no", action="buy", count=contracts,
                                  order_type="limit", no_price=price_cents)

        status = order.get("order", order).get("status", "?")
        log.info("  Order %s | Balance: $%.2f", status, c.get_balance())

        traded_tickers.add(ticker)
        trade_log.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "side": side,
            "price": price_cents,
            "contracts": contracts,
            "edge": edge,
            "signals": reasons,
            "status": status,
        })
        return True
    except Exception as e:
        log.error("  Order failed: %s", e)
        return False


def check_settlements(c):
    """Check if any past trades have settled."""
    global daily_pnl
    try:
        positions = c.get_positions()
        settled_positions = positions if isinstance(positions, list) else positions.get("positions", [])
        # This is simplified — full implementation would track and close positions
    except Exception:
        pass


def main():
    log.info("=" * 50)
    log.info("CRYPTO WATCH — Small bets, strong edges only")
    log.info("Max %d contracts, min %d%% edge, %d/5 signals needed",
             MAX_CONTRACTS, int(MIN_EDGE * 100), MIN_SIGNALS)
    log.info("=" * 50)

    c = KalshiClient()
    tracker = KXBTCMarketTracker(c)
    log.info("Balance: $%.2f", c.get_balance())

    scan_count = 0
    trade_count = 0
    last_status = time.time()

    while True:
        try:
            scan_count += 1
            market = tracker.get_next_market()

            if market:
                remaining = tracker.get_market_time_remaining(market)
                features = load_latest_features()

                if features is not None and 120 < remaining < 780:
                    btc = get_btc_price()
                    bull, bear, reasons = score_signals(features)

                    if try_trade(c, market, features):
                        trade_count += 1

                    # Status update every 5 min
                    if time.time() - last_status > 300:
                        log.info(
                            "Status: scans=%d trades=%d | %s %ds left | bull=%d bear=%d | BTC=$%s",
                            scan_count, trade_count, market["ticker"],
                            int(remaining), bull, bear,
                            f"{btc:,.2f}" if btc else "?",
                        )
                        last_status = time.time()

            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception:
            log.exception("Scan error")
            time.sleep(10)

    log.info("Shutting down. Trades made: %d", trade_count)
    if trade_log:
        df = pd.DataFrame(trade_log)
        path = Path(settings.DATA_DIR) / "crypto_trades.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        log.info("Trade log saved to %s", path)


if __name__ == "__main__":
    main()
