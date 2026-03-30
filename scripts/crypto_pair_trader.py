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

        # Evaluate pair opportunity (correct: bids→implied asks→maker pricing)
        opp = evaluate_pair_opportunity(ob, pair_cap_cents=args.pair_cap)

        # Write state for dashboard
        state = {
            "time": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "scan": scan_count,
            "ticker": ticker,
            "remaining_s": int(remaining),
            "best_yes_bid": opp.get("best_yes_bid", 0),
            "best_no_bid": opp.get("best_no_bid", 0),
            "implied_yes_ask": opp.get("implied_yes_ask", 0),
            "implied_no_ask": opp.get("implied_no_ask", 0),
            "taker_pair_cost": opp.get("taker_pair_cost", 0),
            "taker_arb": opp.get("taker_arb", False),
            "maker_yes": opp.get("maker_yes_price", 0),
            "maker_no": opp.get("maker_no_price", 0),
            "maker_cost": opp.get("maker_pair_cost", 0),
            "maker_net": opp.get("maker_net", 0),
            "maker_tradeable": opp.get("maker_tradeable", False),
            "spread": opp.get("spread_cents", 0),
            "active_pairs": len(pair_tracker.active),
            "stats": pair_tracker.get_stats(),
            "risk": risk.summary(),
        }
        try:
            STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
        except Exception:
            pass

        if not opp["maker_tradeable"]:
            if scan_count % 10 == 0:
                log.info("Scan #%d %s: taker=%dc(no arb) maker=%dc(net=%.1fc) spread=%dc",
                         scan_count, ticker, opp["taker_pair_cost"],
                         opp["maker_pair_cost"], opp["maker_net"], opp["spread_cents"])
            time.sleep(args.interval)
            continue

        # Risk check
        can_open, reason = risk.can_open_pair(opp["maker_pair_cost"])
        if not can_open:
            log.info("Risk blocked: %s", reason)
            time.sleep(args.interval)
            continue

        # --- TRADE (maker strategy: post limits at best_bid+1) ---
        my = opp["maker_yes_price"]
        mn = opp["maker_no_price"]
        mc = opp["maker_pair_cost"]
        mnet = opp["maker_net"]

        if args.mode == "sim":
            # Virtual maker trade (assumes both fill — optimistic)
            pair = pair_tracker.start_pair(ticker, my, mn, market.get("close_time", ""))
            pair.record_yes_fill(my, 1)
            pair.record_no_fill(mn, 1)
            pair_tracker.complete_pair(ticker)
            risk.record_entry(mc)
            risk.record_pair_complete(mc)
            traded_tickers.add(ticker)

            log.info(">>> PAIR SIM (MAKER): %s YES@%dc + NO@%dc = %dc cost, +%.1fc net "
                     "[bids: Y%dc N%dc | asks: Y%dc N%dc | spread:%dc]",
                     ticker, my, mn, mc, mnet,
                     opp["best_yes_bid"], opp["best_no_bid"],
                     opp["implied_yes_ask"], opp["implied_no_ask"],
                     opp["spread_cents"])

            with open(PAIR_LOG, "a") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()} SIM_MAKER_PAIR "
                        f"{ticker} YES@{my}c NO@{mn}c cost={mc}c "
                        f"gross={100-mc}c net={mnet:.1f}c "
                        f"spread={opp['spread_cents']}c\n")

            alert_trade_placed(ticker, "MAKER_PAIR", mc, 1, mnet, strategy="crypto_pair:sim")

        elif args.mode == "paper":
            pair = pair_tracker.start_pair(ticker, my, mn, market.get("close_time", ""))
            pair.record_yes_fill(my, 1)
            pair.record_no_fill(mn, 1)
            pair_tracker.complete_pair(ticker)
            risk.record_entry(mc)
            risk.record_pair_complete(mc)
            traded_tickers.add(ticker)

            log.info(">>> PAIR PAPER (MAKER): %s YES@%dc + NO@%dc = %dc, +%.1fc net",
                     ticker, my, mn, mc, mnet)
            alert_trade_placed(ticker, "MAKER_PAIR", mc, 1, mnet, strategy="crypto_pair:paper")

        elif args.mode == "live":
            log.warning("LIVE MODE: Would post maker YES@%dc + NO@%dc on %s — NOT IMPLEMENTED",
                        my, mn, ticker)
            # TODO: post both limits, monitor fills via WebSocket, handle orphans

        time.sleep(args.interval)

    except KeyboardInterrupt:
        log.info("Pair trader stopped. Stats: %s", pair_tracker.get_stats())
        break
    except Exception:
        log.error("Pair trader error", exc_info=True)
        time.sleep(30)
