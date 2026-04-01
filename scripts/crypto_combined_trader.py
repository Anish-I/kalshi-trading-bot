"""
Crypto Combined Trader: ML conjunction + Pair acquisition on same markets.

Two strategies on every BTC 15m market:
1. ML CONJUNCTION: when XGB+MOM agree, take directional bet (3 contracts max)
2. PAIR MAKER: when spread > 5c, post both YES+NO limits (1 contract each)

Both run in the same loop, same market, non-conflicting.

Run: python scripts/crypto_combined_trader.py --mode sim
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
from engine.alerts import alert_trade_placed, alert_settlement
from engine.pair_pricing import evaluate_pair_opportunity
from engine.pair_state import PairTracker
from engine.pair_risk import PairRiskManager

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["sim", "live"], default="sim")
parser.add_argument("--pair-cap", type=int, default=98, help="Max maker pair cost in cents")
args = parser.parse_args()

_lock = ProcessLock("crypto_combined")
_lock.kill_existing()
if not _lock.acquire():
    print("Another combined trader is running. Exiting.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"crypto_combined_{args.mode}.log"),
    ],
)
log = logging.getLogger("combined_trader")

from kalshi.client import KalshiClient
from kalshi.market_discovery import KXBTCMarketTracker
from config.settings import settings
from models.signal_models import XGBoostModel, MomentumModel
from features.honest_features import compute_honest_features
from data.storage import DataStorage

# ML setup
MODEL_PATH = Path("D:/kalshi-models/latest_model.json")
SCHEMA_PATH = Path("D:/kalshi-models/latest_model.schema.json")

ML_MAX_CONTRACTS = 3
ML_MAX_ENTRY_PRICE = 0.45
ML_MIN_EDGE = 0.03
PAIR_MIN_NET = 0.5  # minimum 0.5c net profit for pair (more aggressive)
SCAN_INTERVAL = 15  # scan every 15s to catch spread windows faster
DAILY_LOSS_LIMIT = 500  # 5 dollars

STATE_FILE = Path(settings.DATA_DIR) / "combined_trader_state.json"
TRADE_LOG = Path(settings.DATA_DIR) / "logs" / "combined_trades.log"
TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)

log.info("=" * 60)
log.info("  CRYPTO COMBINED TRADER — %s mode", args.mode.upper())
log.info("  ML: conjunction only, %d contracts max", ML_MAX_CONTRACTS)
log.info("  Pair: maker at %dc cap, 1 contract", args.pair_cap)
log.info("=" * 60)

client = KalshiClient()
tracker = KXBTCMarketTracker(client)
pair_tracker = PairTracker()
pair_risk = PairRiskManager(pair_cap_cents=args.pair_cap, budget_cents=1500)

# Load ML models
xgb_model = XGBoostModel(str(MODEL_PATH))
momentum_model = MomentumModel()

traded_tickers_ml = set()
traded_tickers_pair = set()
daily_pnl = 0
scan_count = 0
ml_trades = 0
pair_trades = 0
ml_wins = 0
ml_losses = 0


_storage = DataStorage(settings.DATA_DIR)


def load_latest_features():
    """Load latest 1m bars and compute features."""
    try:
        from datetime import date, timedelta
        today = date.today()
        start = (today - timedelta(days=2)).strftime("%Y%m%d")
        end = today.strftime("%Y%m%d")
        bars = _storage.load_bars("bars_1m", start, end)
        if bars is None or len(bars) < 60:
            return None
        return compute_honest_features(bars)
    except Exception:
        return None


def get_btc_price():
    try:
        from datetime import date, timedelta
        start = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
        end = date.today().strftime("%Y%m%d")
        bars = _storage.load_bars("bars_1m", start, end)
        if bars is not None and len(bars) > 0:
            return float(bars.iloc[-1]["close"])
    except Exception:
        pass
    return 0.0


while True:
    try:
        scan_count += 1
        now = datetime.now(timezone.utc)

        market = tracker.get_next_market()
        if not market:
            time.sleep(SCAN_INTERVAL)
            continue

        ticker = market["ticker"]
        remaining = tracker.get_market_time_remaining(market)

        if remaining < 60 or remaining > 780:
            time.sleep(SCAN_INTERVAL)
            continue

        if daily_pnl <= -DAILY_LOSS_LIMIT:
            if scan_count % 20 == 0:
                log.info("Daily loss limit hit: %+dc", daily_pnl)
            time.sleep(SCAN_INTERVAL)
            continue

        btc = get_btc_price()

        # ============================================================
        # STRATEGY 1: ML CONJUNCTION (research/log only — no live orders)
        # ============================================================
        ml_action = "no_signal"
        ml_side = None

        if ticker not in traded_tickers_ml and remaining > 120:
            features_df = load_latest_features()
            if features_df is not None:
                xgb_vote, xgb_conf = xgb_model.score(features_df)
                last_row = features_df.iloc[-1].to_dict()
                mom_vote, mom_conf = momentum_model.score(last_row)

                if xgb_vote == mom_vote and xgb_vote != "flat":
                    direction = xgb_vote
                    yes_ask = float(market.get("yes_ask_dollars", 0) or 0)
                    no_ask = float(market.get("no_ask_dollars", 0) or 0)

                    # Get real prices from orderbook
                    try:
                        ob = client.get_orderbook(ticker, depth=3)
                        fp = ob.get("orderbook_fp", {})
                        yb = fp.get("yes_dollars", [])
                        nb = fp.get("no_dollars", [])
                        if yb:
                            yes_ask = 1.0 - float(nb[-1][0]) if nb else 0
                            no_ask = 1.0 - float(yb[-1][0]) if yb else 0
                    except Exception:
                        pass

                    if direction == "up" and 0 < yes_ask <= ML_MAX_ENTRY_PRICE:
                        ml_side = "yes"
                        entry = yes_ask
                    elif direction == "down" and 0 < no_ask <= ML_MAX_ENTRY_PRICE:
                        ml_side = "no"
                        entry = no_ask

                    if ml_side:
                        entry_cents = int(round(entry * 100))
                        edge = 0.485 - entry  # empirical P(win) for conjunction
                        if edge > ML_MIN_EDGE:
                            ml_action = "trading"

                            # ML always runs in SIM (research only, no live orders)
                            if True:
                                traded_tickers_ml.add(ticker)
                                log.info(">>> ML SIM: %s %s @%dc x%d | XGB=%s MOM=%s | BTC=$%s",
                                         ml_side.upper(), ticker, entry_cents, ML_MAX_CONTRACTS,
                                         xgb_vote[0].upper(), mom_vote[0].upper(),
                                         f"{btc:,.0f}" if btc else "?")
                                with open(TRADE_LOG, "a") as f:
                                    f.write(f"{now.isoformat()} ML_SIM {ml_side.upper()} {ticker} "
                                            f"@{entry_cents}c x{ML_MAX_CONTRACTS} "
                                            f"XGB={xgb_vote[0].upper()} MOM={mom_vote[0].upper()}\n")
                            else:
                                try:
                                    yes_price = entry_cents if ml_side == "yes" else None
                                    no_price = entry_cents if ml_side == "no" else None
                                    resp = client.place_order(
                                        ticker=ticker, side=ml_side, action="buy",
                                        count=ML_MAX_CONTRACTS, order_type="limit",
                                        yes_price=yes_price, no_price=no_price,
                                    )
                                    order_data = resp.get("order", resp)
                                    traded_tickers_ml.add(ticker)
                                    log.info(">>> ML LIVE: %s %s @%dc x%d | order=%s status=%s",
                                             ml_side.upper(), ticker, entry_cents, ML_MAX_CONTRACTS,
                                             order_data.get("order_id", "?"), order_data.get("status", "?"))
                                    with open(TRADE_LOG, "a") as f:
                                        f.write(f"{now.isoformat()} ML_LIVE {ml_side.upper()} {ticker} "
                                                f"@{entry_cents}c x{ML_MAX_CONTRACTS} "
                                                f"order={order_data.get('order_id', '?')}\n")
                                except Exception:
                                    log.error("ML order failed", exc_info=True)
                                    continue  # don't count failed orders

                            alert_trade_placed(ticker, ml_side, entry_cents, ML_MAX_CONTRACTS,
                                               edge * 100, strategy=f"combined_ml:{args.mode}")
                            ml_trades += 1

        # ============================================================
        # STRATEGY 2: PAIR MAKER
        # ============================================================
        pair_action = "no_spread"

        if ticker not in traded_tickers_pair and ticker not in traded_tickers_ml:
            try:
                ob = client.get_orderbook(ticker, depth=5)
            except Exception:
                ob = {}

            if ob:
                opp = evaluate_pair_opportunity(ob, pair_cap_cents=args.pair_cap)

                if opp["maker_tradeable"]:
                    can_open, reason = pair_risk.can_open_pair(opp["maker_pair_cost"])
                    if can_open:
                        my = opp["maker_yes_price"]
                        mn = opp["maker_no_price"]
                        mc = opp["maker_pair_cost"]
                        mnet = opp["maker_net"]
                        traded_tickers_pair.add(ticker)
                        pair_action = "trading"

                        if args.mode == "sim":
                            pair = pair_tracker.start_pair(ticker, my, mn)
                            pair.record_yes_fill(my, 1)
                            pair.record_no_fill(mn, 1)
                            pair_tracker.complete_pair(ticker)
                            pair_risk.record_entry(mc)
                            pair_risk.record_pair_complete(mc)

                            log.info(">>> PAIR SIM: %s YES@%dc NO@%dc = %dc, +%.1fc net [spread %dc]",
                                     ticker, my, mn, mc, mnet, opp["spread_cents"])
                            with open(TRADE_LOG, "a") as f:
                                f.write(f"{now.isoformat()} PAIR_SIM {ticker} "
                                        f"YES@{my}c NO@{mn}c cost={mc}c net={mnet:.1f}c\n")
                        else:
                            # Live: place both limit orders
                            try:
                                yes_resp = client.place_order(
                                    ticker=ticker, side="yes", action="buy",
                                    count=1, order_type="limit", yes_price=my,
                                )
                                no_resp = client.place_order(
                                    ticker=ticker, side="no", action="buy",
                                    count=1, order_type="limit", no_price=mn,
                                )
                                log.info(">>> PAIR LIVE: %s YES@%dc NO@%dc | orders placed",
                                         ticker, my, mn)
                                with open(TRADE_LOG, "a") as f:
                                    f.write(f"{now.isoformat()} PAIR_LIVE {ticker} "
                                            f"YES@{my}c NO@{mn}c cost={mc}c\n")
                            except Exception:
                                log.error("Pair order failed", exc_info=True)

                        alert_trade_placed(ticker, "PAIR", mc, 1, mnet,
                                           strategy=f"combined_pair:{args.mode}")
                        pair_trades += 1

        # ============================================================
        # STATE
        # ============================================================
        if scan_count % 10 == 0 or ml_action == "trading" or pair_action == "trading":
            state = {
                "time": now.isoformat(),
                "mode": args.mode,
                "scan": scan_count,
                "ticker": ticker,
                "remaining_s": int(remaining),
                "btc_price": btc,
                "ml_action": ml_action,
                "pair_action": pair_action,
                "ml_trades": ml_trades,
                "pair_trades": pair_trades,
                "daily_pnl": daily_pnl,
                "pair_stats": pair_tracker.get_stats(),
                "pair_risk": pair_risk.summary(),
            }
            try:
                STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
            except Exception:
                pass

        time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        log.info("Combined trader stopped. ML: %d trades, Pair: %d trades", ml_trades, pair_trades)
        break
    except Exception:
        log.error("Combined trader error", exc_info=True)
        time.sleep(30)
