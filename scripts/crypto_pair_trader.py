"""
Crypto Pair Trader — buy both YES and NO for guaranteed profit.

Modes:
  --sim    (default): evaluate opportunities, log virtual trades
  --paper: place real limit orders but with pair-cap safety
  --live:  full execution (only after paper gate passes)

Run: python scripts/crypto_pair_trader.py --sim
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

from engine.process_lock import ProcessLock
from engine.alerts import alert_trade_placed
from engine.pair_pricing import evaluate_pair_opportunity, PAIR_FEE_CENTS
from engine.pair_state import PairTracker
from engine.pair_risk import PairRiskManager

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["sim", "paper", "live"], default="sim")
parser.add_argument("--interval", type=int, default=30, help="Scan interval seconds")
parser.add_argument("--pair-cap", type=int, default=96, help="Max pair cost in cents")
args = parser.parse_args()

_lock = ProcessLock("crypto_pair_trader")
_lock.kill_existing()
if not _lock.acquire():
    print("Another pair trader is running. Exiting.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"crypto_pair_{args.mode}.log"),
    ],
)
log = logging.getLogger("pair_trader")

from kalshi.client import KalshiClient
from kalshi.market_discovery import KXBTCMarketTracker

STATE_FILE = Path("D:/kalshi-data/pair_trader_state.json")
PAIR_LOG = Path("D:/kalshi-data/logs/pair_trades.log")
PAIR_LOG.parent.mkdir(parents=True, exist_ok=True)

log.info("=" * 60)
log.info("  Crypto Pair Trader — %s mode", args.mode.upper())
log.info("  Pair cap: %dc | Scan: %ds", args.pair_cap, args.interval)
log.info("=" * 60)

client = KalshiClient()
tracker_mkt = KXBTCMarketTracker(client)
pair_tracker = PairTracker()
risk = PairRiskManager(pair_cap_cents=args.pair_cap, budget_cents=2500)

scan_count = 0
traded_tickers = set()

while True:
    try:
        scan_count += 1
        market = tracker_mkt.get_next_market()
        if not market:
            time.sleep(args.interval)
            continue

        ticker = market["ticker"]
        remaining = tracker_mkt.get_market_time_remaining(market)

        # Skip if already traded or too close to expiry
        if ticker in traded_tickers:
            time.sleep(args.interval)
            continue
        if remaining < 60 or remaining > 780:
            time.sleep(args.interval)
            continue

        # Fetch live orderbook
        try:
            ob = client.get_orderbook(ticker, depth=5)
        except Exception:
            time.sleep(args.interval)
            continue

        # Evaluate pair opportunity
        opp = evaluate_pair_opportunity(ob, pair_cap_cents=args.pair_cap)

        # Write state for dashboard
        state = {
            "time": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "scan": scan_count,
            "ticker": ticker,
            "remaining_s": int(remaining),
            "yes_ask": opp["yes_ask_cents"],
            "no_ask": opp["no_ask_cents"],
            "pair_cost": opp["pair_cost_cents"],
            "gross_profit": opp["gross_profit_cents"],
            "net_profit": opp["net_profit_cents"],
            "tradeable": opp["tradeable"],
            "max_pairs": opp["max_pairs"],
            "active_pairs": len(pair_tracker.active),
            "stats": pair_tracker.get_stats(),
            "risk": risk.summary(),
        }
        try:
            STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
        except Exception:
            pass

        if not opp["tradeable"]:
            if scan_count % 10 == 0:
                log.info("Scan #%d %s: cost=%dc gross=%dc net=%.1fc — no opportunity",
                         scan_count, ticker, opp["pair_cost_cents"],
                         opp["gross_profit_cents"], opp["net_profit_cents"])
            time.sleep(args.interval)
            continue

        # Risk check
        can_open, reason = risk.can_open_pair(opp["pair_cost_cents"])
        if not can_open:
            log.info("Risk blocked: %s", reason)
            time.sleep(args.interval)
            continue

        # --- TRADE ---
        ya = opp["yes_ask_cents"]
        na = opp["no_ask_cents"]
        net = opp["net_profit_cents"]

        if args.mode == "sim":
            # Virtual trade
            pair = pair_tracker.start_pair(ticker, ya, na, market.get("close_time", ""))
            pair.record_yes_fill(ya, 1)
            pair.record_no_fill(na, 1)
            pair_tracker.complete_pair(ticker)
            risk.record_entry(ya + na)
            risk.record_pair_complete(ya + na)
            traded_tickers.add(ticker)

            log.info(">>> PAIR SIM: %s YES@%dc + NO@%dc = %dc cost, +%.1fc net profit",
                     ticker, ya, na, ya + na, net)

            with open(PAIR_LOG, "a") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()} SIM_PAIR "
                        f"{ticker} YES@{ya}c NO@{na}c cost={ya+na}c "
                        f"gross={100-ya-na}c net={net:.1f}c\n")

            alert_trade_placed(ticker, "PAIR", ya + na, 1, net, strategy="crypto_pair:sim")

        elif args.mode == "paper":
            # Record as paper trade (don't place real orders)
            pair = pair_tracker.start_pair(ticker, ya, na, market.get("close_time", ""))
            pair.record_yes_fill(ya, 1)
            pair.record_no_fill(na, 1)
            pair_tracker.complete_pair(ticker)
            risk.record_entry(ya + na)
            risk.record_pair_complete(ya + na)
            traded_tickers.add(ticker)

            log.info(">>> PAIR PAPER: %s YES@%dc + NO@%dc = %dc, +%.1fc net",
                     ticker, ya, na, ya + na, net)
            alert_trade_placed(ticker, "PAIR", ya + na, 1, net, strategy="crypto_pair:paper")

        elif args.mode == "live":
            # Real orders — place both legs
            log.warning("LIVE MODE: Would place YES@%dc + NO@%dc on %s — NOT IMPLEMENTED YET",
                        ya, na, ticker)
            # TODO: implement atomic pair placement with order_group_id
            # TODO: orphan leg handling with WebSocket monitoring

        time.sleep(args.interval)

    except KeyboardInterrupt:
        log.info("Pair trader stopped. Stats: %s", pair_tracker.get_stats())
        break
    except Exception:
        log.error("Pair trader error", exc_info=True)
        time.sleep(30)
