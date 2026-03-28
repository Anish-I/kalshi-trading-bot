"""
Fed Rate Trader — scans KXFED markets and trades based on rate model.

Modes:
  --shadow  (default): log what we'd trade, no orders
  --paper:  record simulated trades in ledger
  --live:   real 1-contract orders (only after shadow gate passes)

Run: python scripts/run_fed_trader.py --shadow
"""

import argparse
import logging
import sys
import time

sys.path.insert(0, ".")

from engine.process_lock import ProcessLock

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["shadow", "paper", "live"], default="shadow")
parser.add_argument("--interval", type=int, default=300, help="Scan interval in seconds")
args = parser.parse_args()

_lock = ProcessLock("fed_trader")
_lock.kill_existing()
if not _lock.acquire():
    print("Another fed trader is running. Exiting.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"fed_trader_{args.mode}.log"),
    ],
)
log = logging.getLogger("fed_trader")

from fed.trader import FedTrader

log.info("=" * 60)
log.info("  Fed Rate Trader — %s mode", args.mode.upper())
log.info("  Scan interval: %ds", args.interval)
log.info("=" * 60)

trader = FedTrader(mode=args.mode)
scan_count = 0

while True:
    try:
        scan_count += 1
        state = trader.scan()

        action = state.get("action", "?")
        n_opps = state.get("opportunities", 0)
        signal = state.get("signal", {})

        log.info(
            "Scan #%d | rate=%.2f%% | fomc=%s (%s days) | %d opps | action=%s",
            scan_count,
            signal.get("current_rate", 0),
            signal.get("next_fomc", "?"),
            signal.get("days_to_fomc", "?"),
            n_opps,
            action,
        )

        time.sleep(args.interval)

    except KeyboardInterrupt:
        log.info("Fed trader stopped after %d scans", scan_count)
        break
    except Exception:
        log.error("Fed trader error, continuing...", exc_info=True)
        time.sleep(60)
