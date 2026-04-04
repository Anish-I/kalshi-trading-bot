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
from engine.pair_pricing import evaluate_pair_opportunity, extract_book_from_orderbook
from engine.pair_state import PairTracker
from engine.pair_risk import PairRiskManager

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["sim", "live"], default="sim")
parser.add_argument("--pair-cap", type=int, default=95, help="Max maker pair cost in cents")
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

ML_MAX_CONTRACTS = 5
ML_MAX_ENTRY_PRICE = 0.40
ML_MIN_EDGE = 0.04
PAIR_MIN_NET = 5.0  # minimum 5c net profit per pair (raised from 2.5c to absorb orphan risk)
SCAN_INTERVAL = 15  # scan every 15s to catch spread windows faster

# Only trade pairs on series with wide enough spreads (avg > 4c).
# BTC/ETH/SOL/XRP have 1-3c avg spreads → 80%+ orphan rate, net negative.
PAIR_ENABLED_SERIES = {"KXHYPE15M", "KXDOGE15M", "KXSOL15M", "KXXRP15M"}

# Per-series orphan timeouts (seconds) — raised to 60s to allow fills on thin books
ORPHAN_TIMEOUT = {
    "KXBTC15M": 60,
    "KXETH15M": 60,
    "KXSOL15M": 60,
    "KXXRP15M": 60,
    "KXHYPE15M": 60,
    "KXDOGE15M": 60,
}
DEFAULT_ORPHAN_TIMEOUT = 60

# Auto-scaling: increase contracts as completed pairs accumulate without orphans
PAIR_SCALE_TIERS = [
    (0, 1),    # 0-9 completed pairs: 1 contract (prove it works first)
    (10, 5),   # 10-29 completed: scale to 5
    (30, 7),   # 30-49 completed: scale to 7
    (50, 10),  # 50+ completed: full 10 contracts
]


def auto_pair_size(completed: int, orphans: int) -> int:
    """Scale pair size based on track record. Drops back if orphan rate is bad."""
    if completed > 0 and orphans / completed > 0.1:
        return 1  # orphan rate > 10% → minimum size

    for threshold, size in reversed(PAIR_SCALE_TIERS):
        if completed >= threshold:
            return size
    return 3

# Elastic pair cap: adjusts based on recent spread history
PAIR_CAP_FLOOR = 95
PAIR_CAP_CEILING = 97
PAIR_CAP_WINDOW = 20

# Per-series elastic state
_series_spreads: dict[str, list[int]] = {}
_series_completed: dict[str, int] = {}
_series_orphans: dict[str, int] = {}

# 2-scan confirmation: only trade if spread qualified on previous scan too
_series_spread_qualified: dict[str, bool] = {}


def elastic_pair_cap(series: str) -> int:
    """Per-series elastic cap based on that series' spread history."""
    spreads = _series_spreads.get(series, [])
    if len(spreads) < 5:
        return args.pair_cap

    avg = sum(spreads) / len(spreads)
    if avg > 6:
        return PAIR_CAP_FLOOR
    elif avg > 3:
        return 97
    else:
        return PAIR_CAP_CEILING


def track_series_spread(series: str, spread: int) -> None:
    if series not in _series_spreads:
        _series_spreads[series] = []
    _series_spreads[series].append(spread)
    if len(_series_spreads[series]) > PAIR_CAP_WINDOW:
        _series_spreads[series].pop(0)


def series_pair_size(series: str) -> int:
    """Per-series auto-scaling based on that series' track record."""
    completed = _series_completed.get(series, 0)
    orphans = _series_orphans.get(series, 0)
    return auto_pair_size(completed, orphans)


def record_series_complete(series: str) -> None:
    _series_completed[series] = _series_completed.get(series, 0) + 1


def record_series_orphan(series: str) -> None:
    _series_orphans[series] = _series_orphans.get(series, 0) + 1
MAX_DRAWDOWN = 2900  # $29 max loss from zero before halting

# Trailing stop: tightens as confirmed profits grow
# Below TRAILING_ACTIVATION: just use MAX_DRAWDOWN from zero
# Above TRAILING_ACTIVATION: lock in TRAILING_LOCK_PCT of peak
TRAILING_ACTIVATION = 500   # $5 — trailing only kicks in after meaningful profit
TRAILING_LOCK_PCT = 0.5     # keep at least 50% of peak profit
ORPHAN_RISK_DISCOUNT = 0.5  # orphans are open positions, not confirmed losses


class TrailingStopLoss:
    """Trailing stop that protects gains as they grow.

    Two phases:
    1. Warmup (peak < activation): allow full max_drawdown from zero
    2. Trailing (peak >= activation): halt if P&L drops below peak * lock_pct

    Orphan legs are NOT confirmed losses — they're open positions with ~50/50
    settlement odds. We discount their risk accordingly.
    """

    def __init__(self, max_drawdown: int, activation: int, lock_pct: float,
                 orphan_discount: float = 0.5, orphan_ttl_s: float = 900.0):
        self.max_drawdown = max_drawdown
        self.activation = activation      # trailing activates above this peak
        self.lock_pct = lock_pct          # fraction of peak to protect
        self.orphan_discount = orphan_discount
        self.orphan_ttl_s = orphan_ttl_s  # orphan risk expires after market settles
        self.confirmed_pnl = 0            # settled P&L from completed pairs
        self._orphan_entries: list[tuple[float, int]] = []  # (timestamp, risk_cents)
        self.peak_pnl = 0                 # highest estimated_pnl seen

    @property
    def orphan_risk(self) -> int:
        """Active orphan risk — expired entries (settled markets) are dropped."""
        now = time.monotonic()
        self._orphan_entries = [(t, c) for t, c in self._orphan_entries
                                if now - t < self.orphan_ttl_s]
        return sum(c for _, c in self._orphan_entries)

    @property
    def estimated_pnl(self) -> int:
        """Best estimate of current P&L (confirmed minus active orphan risk)."""
        return self.confirmed_pnl - self.orphan_risk

    def record_profit(self, cents: int) -> None:
        """Record confirmed profit from a completed pair."""
        self.confirmed_pnl += cents
        if self.estimated_pnl > self.peak_pnl:
            self.peak_pnl = self.estimated_pnl

    def record_orphan(self, filled_cost_cents: int) -> None:
        """Record an orphan leg. Risk decays after market settles (~15min)."""
        risk = int(filled_cost_cents * self.orphan_discount)
        self._orphan_entries.append((time.monotonic(), risk))

    @property
    def stop_price(self) -> int:
        """P&L level at which trading halts."""
        if self.peak_pnl < self.activation:
            # Warmup: just use max drawdown from zero
            return -self.max_drawdown
        # Trailing: protect lock_pct of peak
        return int(self.peak_pnl * self.lock_pct)

    def should_halt(self) -> tuple[bool, str]:
        """Check if we should stop trading."""
        pnl = self.estimated_pnl
        # Hard floor: never lose more than max_drawdown from zero
        if pnl <= -self.max_drawdown:
            return True, (f"max drawdown: pnl={pnl}c <= -{self.max_drawdown}c "
                         f"(confirmed={self.confirmed_pnl}c orphan_risk={self.orphan_risk}c)")
        # Trailing stop: only active after meaningful gains
        if self.peak_pnl >= self.activation and pnl <= self.stop_price:
            return True, (f"trailing stop: pnl={pnl}c <= stop={self.stop_price}c "
                         f"(peak={self.peak_pnl}c, locking {self.lock_pct:.0%})")
        return False, "ok"

    def summary(self) -> dict:
        return {
            "confirmed_pnl": self.confirmed_pnl,
            "orphan_risk": self.orphan_risk,
            "estimated_pnl": self.estimated_pnl,
            "peak_pnl": self.peak_pnl,
            "stop_price": self.stop_price,
            "trailing_active": self.peak_pnl >= self.activation,
        }

STATE_FILE = Path(settings.DATA_DIR) / "combined_trader_state.json"
TRADE_LOG = Path(settings.DATA_DIR) / "logs" / "combined_trades.log"
TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)

CRYPTO_SERIES = ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXHYPE15M", "KXDOGE15M"]

log.info("=" * 60)
log.info("  CRYPTO COMBINED TRADER — %s mode", args.mode.upper())
log.info("  Series: %s", ", ".join(CRYPTO_SERIES))
log.info("  ML: conjunction only, %d contracts max (research)", ML_MAX_CONTRACTS)
log.info("  Pair: elastic cap, auto-scale, 24/7")
log.info("=" * 60)

client = KalshiClient()
pair_tracker = PairTracker()
pair_risk = PairRiskManager(pair_cap_cents=args.pair_cap, budget_cents=5000)

# Load ML models
xgb_model = XGBoostModel(str(MODEL_PATH))
momentum_model = MomentumModel()

traded_tickers_ml = set()
traded_tickers_pair = set()
trailing_stop = TrailingStopLoss(MAX_DRAWDOWN, TRAILING_ACTIVATION, TRAILING_LOCK_PCT, ORPHAN_RISK_DISCOUNT)
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

        # Scan ALL crypto series for pair opportunities
        all_markets = []
        for series in CRYPTO_SERIES:
            try:
                data = client._request("GET", "/markets",
                                       params={"series_ticker": series, "status": "open", "limit": 5})
                for m in data.get("markets", []):
                    close_time = m.get("close_time", "")
                    if close_time:
                        m["_series"] = series
                        all_markets.append(m)
            except Exception:
                pass

        if not all_markets:
            time.sleep(SCAN_INTERVAL)
            continue

        # Process EACH market for pair opportunities
        from kalshi.market_discovery import KXBTCMarketTracker
        _tracker = KXBTCMarketTracker(client)

        halt, halt_reason = trailing_stop.should_halt()
        if halt:
            if scan_count % 20 == 0:
                log.info("Trading halted: %s", halt_reason)
            time.sleep(SCAN_INTERVAL)
            continue

        btc = get_btc_price()
        last_ticker = None
        last_remaining = 0
        last_ml_action = "no_signal"
        last_pair_action = "no_spread"
        should_write_state = scan_count % 10 == 0

        for market in all_markets:
            ticker = market["ticker"]
            remaining = _tracker.get_market_time_remaining(market)
            series = market.get("_series", "")
            ml_action = "no_signal"
            pair_action = "no_spread"

            last_ticker = ticker
            last_remaining = int(remaining)
            last_ml_action = ml_action
            last_pair_action = pair_action

            # Re-check trailing stop inside loop (an orphan earlier in this scan could breach it)
            halt, _ = trailing_stop.should_halt()
            if halt:
                break

            if remaining < 60:
                last_ml_action = "skipped_short_window"
                last_pair_action = "skipped_short_window"
                continue

            # ============================================================
            # STRATEGY 1: ML CONJUNCTION (research/log only, BTC only)
            # ============================================================
            ml_side = None

            if series == "KXBTC15M" and ticker not in traded_tickers_ml and remaining > 120:
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

                                # ML: SIM in sim mode, LIVE in live mode
                                if args.mode == "sim":
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
            if ticker not in traded_tickers_pair and ticker not in traded_tickers_ml:
                try:
                    ob = client.get_orderbook(ticker, depth=5)
                except Exception:
                    ob = {}

                if ob:
                    # Track spread per-series for elastic cap
                    book = extract_book_from_orderbook(ob)
                    if book["spread_cents"] > 0:
                        track_series_spread(series, book["spread_cents"])

                    # Skip pair trading on tight-spread series (BTC/ETH/SOL/XRP)
                    if series not in PAIR_ENABLED_SERIES:
                        continue

                    current_cap = elastic_pair_cap(series)
                    pair_risk.pair_cap_cents = current_cap
                    opp = evaluate_pair_opportunity(ob, pair_cap_cents=current_cap)

                    if opp["maker_tradeable"]:
                        # Execute immediately on first qualifying scan (removed 2-scan
                        # confirmation — it doubled latency and killed fill rates)

                        # Check budget with per-series scaled size
                        pair_size_check = series_pair_size(series)
                        can_open, reason = pair_risk.can_open_pair(opp["maker_pair_cost"] * pair_size_check)
                        if can_open:
                            my = opp["maker_yes_price"]
                            mn = opp["maker_no_price"]
                            mc = opp["maker_pair_cost"]
                            mnet = opp["maker_net"]
                            traded_tickers_pair.add(ticker)
                            pair_action = "trading"

                            # Auto-scale per-series
                            pair_size = series_pair_size(series)

                            if args.mode == "sim":
                                pair = pair_tracker.start_pair(ticker, my, mn)
                                pair.record_yes_fill(my, pair_size)
                                pair.record_no_fill(mn, pair_size)
                                pair_tracker.complete_pair(ticker)
                                pair_risk.record_entry(mc * pair_size)
                                pair_risk.record_pair_complete(mc * pair_size)
                                trailing_stop.record_profit(int(mnet * pair_size))
                                record_series_complete(series)

                                log.info(">>> PAIR SIM: %s [%s] YES@%dc NO@%dc = %dc x%d, +%.1fc net [spread %dc cap %dc]",
                                         ticker, series, my, mn, mc, pair_size, mnet * pair_size, opp["spread_cents"], current_cap)
                                with open(TRADE_LOG, "a") as f:
                                    f.write(f"{now.isoformat()} PAIR_SIM {ticker} "
                                            f"YES@{my}c NO@{mn}c cost={mc}c x{pair_size} net={mnet*pair_size:.1f}c\n")
                            else:
                                # Live: submit both resting orders, then poll REST for fill/cancel outcome.
                                def _order_payload(response: dict | None) -> dict:
                                    if not isinstance(response, dict):
                                        return {}
                                    order = response.get("order")
                                    return order if isinstance(order, dict) else response

                                def _to_int(value: object, fallback: int = 0) -> int:
                                    if value in (None, ""):
                                        return fallback
                                    try:
                                        return int(value)
                                    except (TypeError, ValueError):
                                        try:
                                            return int(float(str(value)))
                                        except (TypeError, ValueError):
                                            return fallback

                                def _fill_count(order: dict) -> int:
                                    count = _to_int(order.get("fill_count"))
                                    if count == 0:
                                        count = _to_int(order.get("fill_count_fp"))
                                    if count == 0 and str(order.get("status", "")).lower() == "executed":
                                        return pair_size
                                    return count

                                def _remaining_count(order: dict) -> int:
                                    if "remaining_count" in order or "remaining_count_fp" in order:
                                        remaining = _to_int(order.get("remaining_count"))
                                        if remaining == 0:
                                            remaining = _to_int(order.get("remaining_count_fp"))
                                        return remaining
                                    return max(pair_size - _fill_count(order), 0)

                                def _is_filled(order: dict) -> bool:
                                    status = str(order.get("status", "")).lower()
                                    if status == "executed":
                                        return True
                                    fill_count = _fill_count(order)
                                    remaining_count = _remaining_count(order)
                                    return fill_count >= pair_size or (fill_count > 0 and remaining_count == 0)

                                def _order_price(order: dict, side: str, fallback: int) -> int:
                                    price_key = "yes_price" if side == "yes" else "no_price"
                                    return _to_int(order.get(price_key), fallback)

                                def _sync_leg(order: dict, side: str, fallback: int) -> None:
                                    if not _is_filled(order):
                                        return
                                    status = str(order.get("status", "")).lower()
                                    qty = _fill_count(order)
                                    if qty == 0 and status == "executed":
                                        qty = pair_size
                                    qty = max(qty, 1)
                                    price = _order_price(order, side, fallback)
                                    if side == "yes":
                                        pair.record_yes_fill(price, qty)
                                    else:
                                        pair.record_no_fill(price, qty)

                                def _complete_live_pair() -> None:
                                    completed = pair_tracker.complete_pair(ticker)
                                    if completed is None:
                                        return
                                    pair_risk.record_pair_complete(mc * pair_size)
                                    trailing_stop.record_profit(int(mnet * pair_size))
                                    record_series_complete(series)
                                    completed_at = datetime.now(timezone.utc).isoformat()
                                    log.info(
                                        ">>> PAIR LIVE COMPLETE: %s [%s] YES@%dc NO@%dc = %dc x%d, +%.1fc net",
                                        ticker,
                                        series,
                                        completed.yes_filled_price or my,
                                        completed.no_filled_price or mn,
                                        completed.pair_cost_cents,
                                        pair_size,
                                        mnet * pair_size,
                                    )
                                    with open(TRADE_LOG, "a") as f:
                                        f.write(
                                            f"{completed_at} PAIR_LIVE_COMPLETE {ticker} "
                                            f"YES@{completed.yes_filled_price or my}c "
                                            f"NO@{completed.no_filled_price or mn}c "
                                            f"cost={completed.pair_cost_cents}c x{pair_size} "
                                            f"net={mnet*pair_size:.1f}c "
                                            f"yes_order={yes_order_id} no_order={no_order_id}\n"
                                        )

                                yes_resp = None
                                no_resp = None
                                try:
                                    yes_resp = client.place_order(
                                        ticker=ticker, side="yes", action="buy",
                                        count=pair_size, order_type="limit", yes_price=my,
                                    )
                                    no_resp = client.place_order(
                                        ticker=ticker, side="no", action="buy",
                                        count=pair_size, order_type="limit", no_price=mn,
                                    )
                                except Exception:
                                    yes_order = _order_payload(yes_resp)
                                    yes_order_id = str(yes_order.get("order_id", "") or "")
                                    if yes_order_id:
                                        try:
                                            client.cancel_order(yes_order_id)
                                        except Exception:
                                            log.warning("Failed canceling submitted YES leg for %s", ticker, exc_info=True)
                                    traded_tickers_pair.discard(ticker)
                                    log.error("Pair order failed", exc_info=True)
                                    continue

                                yes_order = _order_payload(yes_resp)
                                no_order = _order_payload(no_resp)
                                yes_order_id = str(yes_order.get("order_id", "") or "")
                                no_order_id = str(no_order.get("order_id", "") or "")

                                if not yes_order_id or not no_order_id:
                                    for order_id, side_name in ((yes_order_id, "YES"), (no_order_id, "NO")):
                                        if not order_id:
                                            continue
                                        try:
                                            client.cancel_order(order_id)
                                        except Exception:
                                            log.warning("Failed canceling %s leg for %s after missing order id",
                                                        side_name, ticker, exc_info=True)
                                    traded_tickers_pair.discard(ticker)
                                    log.error("Pair order missing order ids: yes=%s no=%s", yes_order_id or "?", no_order_id or "?")
                                    continue

                                pair = pair_tracker.start_pair(ticker, my, mn, market.get("close_time", ""))
                                pair.yes_order_id = yes_order_id
                                pair.no_order_id = no_order_id
                                pair_risk.record_entry(mc * pair_size)

                                log.info(">>> PAIR LIVE: %s [%s] YES@%dc NO@%dc x%d | submitted yes=%s no=%s",
                                         ticker, series, my, mn, pair_size, yes_order_id, no_order_id)

                                poll_timeout = ORPHAN_TIMEOUT.get(series, DEFAULT_ORPHAN_TIMEOUT)
                                poll_deadline = time.monotonic() + poll_timeout
                                pair_completed = False

                                while time.monotonic() < poll_deadline:
                                    try:
                                        yes_order = client.get_order(yes_order_id)
                                        no_order = client.get_order(no_order_id)
                                    except Exception:
                                        log.warning("Pair order poll failed for %s", ticker, exc_info=True)
                                        time.sleep(1)
                                        continue

                                    if not pair.yes_filled:
                                        _sync_leg(yes_order, "yes", my)
                                    if not pair.no_filled:
                                        _sync_leg(no_order, "no", mn)

                                    if pair.yes_filled and pair.no_filled:
                                        _complete_live_pair()
                                        pair_completed = True
                                        break

                                    # Check for adverse book move
                                    try:
                                        live_ob = client.get_orderbook(ticker, depth=3)
                                        live_book = extract_book_from_orderbook(live_ob)
                                        if pair.yes_filled and not pair.no_filled:
                                            adverse = live_book["implied_no_ask"] - mn
                                            if adverse > 5:
                                                log.warning("Adverse move %dc on NO leg for %s, unwinding", adverse, ticker)
                                                break
                                        elif pair.no_filled and not pair.yes_filled:
                                            adverse = live_book["implied_yes_ask"] - my
                                            if adverse > 5:
                                                log.warning("Adverse move %dc on YES leg for %s, unwinding", adverse, ticker)
                                                break
                                    except Exception:
                                        pass

                                    time.sleep(1)

                                if not pair_completed:
                                    try:
                                        yes_order = client.get_order(yes_order_id)
                                        no_order = client.get_order(no_order_id)
                                        if not pair.yes_filled:
                                            _sync_leg(yes_order, "yes", my)
                                        if not pair.no_filled:
                                            _sync_leg(no_order, "no", mn)
                                    except Exception:
                                        log.warning("Final pair order poll failed for %s", ticker, exc_info=True)

                                if not pair_completed and pair.yes_filled and pair.no_filled:
                                    _complete_live_pair()
                                    pair_completed = True

                                if not pair_completed and pair.is_orphan:
                                    unfilled_side = "no" if pair.yes_filled else "yes"
                                    unfilled_order_id = no_order_id if unfilled_side == "no" else yes_order_id
                                    unfilled_price = mn if unfilled_side == "no" else my

                                    try:
                                        client.cancel_order(unfilled_order_id)
                                    except Exception:
                                        log.warning("Failed canceling %s orphan leg for %s",
                                                    unfilled_side.upper(), ticker, exc_info=True)

                                    try:
                                        latest_unfilled = client.get_order(unfilled_order_id)
                                        _sync_leg(latest_unfilled, unfilled_side, unfilled_price)
                                    except Exception:
                                        log.warning("Failed refreshing %s leg after cancel for %s",
                                                    unfilled_side.upper(), ticker, exc_info=True)

                                    if pair.yes_filled and pair.no_filled:
                                        _complete_live_pair()
                                        pair_completed = True
                                    else:
                                        resolved = pair_tracker.resolve_orphan(ticker, "timeout")
                                        if resolved is not None:
                                            loss_cents = 0
                                            if resolved.yes_filled:
                                                loss_cents = resolved.yes_filled_price * max(resolved.yes_qty, 1)
                                            elif resolved.no_filled:
                                                loss_cents = resolved.no_filled_price * max(resolved.no_qty, 1)
                                            pair_risk.record_orphan(loss_cents)
                                            trailing_stop.record_orphan(loss_cents)
                                            pair_risk.exposure_cents = max(0, pair_risk.exposure_cents - (mc * pair_size))
                                            record_series_orphan(series)
                                            orphaned_at = datetime.now(timezone.utc).isoformat()
                                            log.warning(
                                                ">>> PAIR LIVE ORPHAN: %s [%s] filled=%s canceled=%s loss<=%dc",
                                                ticker,
                                                series,
                                                "YES" if resolved.yes_filled else "NO",
                                                unfilled_side.upper(),
                                                loss_cents,
                                            )
                                            with open(TRADE_LOG, "a") as f:
                                                f.write(
                                                    f"{orphaned_at} PAIR_LIVE_ORPHAN {ticker} "
                                                    f"filled={'YES' if resolved.yes_filled else 'NO'} "
                                                    f"canceled={unfilled_side.upper()} "
                                                    f"loss<={loss_cents}c x{pair_size} "
                                                    f"yes_order={yes_order_id} no_order={no_order_id}\n"
                                                )

                                if not pair_completed and not pair.yes_filled and not pair.no_filled:
                                    for order_id, side_name in ((yes_order_id, "YES"), (no_order_id, "NO")):
                                        try:
                                            client.cancel_order(order_id)
                                        except Exception:
                                            log.warning("Failed canceling stale %s leg for %s",
                                                        side_name, ticker, exc_info=True)
                                    pair.transition("resolved", "timeout_no_fill")
                                    pair_tracker.complete_pair(ticker)
                                    pair_risk.exposure_cents = max(0, pair_risk.exposure_cents - (mc * pair_size))
                                    timed_out_at = datetime.now(timezone.utc).isoformat()
                                    log.info(">>> PAIR LIVE TIMEOUT: %s [%s] no fills in 30s, canceled both legs",
                                             ticker, series)
                                    with open(TRADE_LOG, "a") as f:
                                        f.write(
                                            f"{timed_out_at} PAIR_LIVE_TIMEOUT {ticker} "
                                            f"YES@{my}c NO@{mn}c cost={mc}c x{pair_size} "
                                            f"reason=no_fill_30s yes_order={yes_order_id} no_order={no_order_id}\n"
                                        )

                            alert_trade_placed(ticker, "PAIR", mc * pair_size, pair_size, mnet * pair_size,
                                               strategy=f"combined_pair:{args.mode}")
                            pair_trades += 1
                    else:
                        # Spread collapsed — reset qualification
                        _series_spread_qualified.pop(f"{series}:{ticker}", None)

            last_ml_action = ml_action
            last_pair_action = pair_action
            if ml_action == "trading" or pair_action == "trading":
                should_write_state = True

        # ============================================================
        # STATE
        # ============================================================
        if last_ticker and should_write_state:
            state = {
                "time": now.isoformat(),
                "mode": args.mode,
                "scan": scan_count,
                "ticker": last_ticker,
                "remaining_s": last_remaining,
                "btc_price": btc,
                "ml_action": last_ml_action,
                "pair_action": last_pair_action,
                "ml_trades": ml_trades,
                "pair_trades": pair_trades,
                "estimated_pnl": trailing_stop.estimated_pnl,
                "trailing_stop": trailing_stop.summary(),
                "pair_stats": pair_tracker.get_stats(),
                "pair_risk": pair_risk.summary(),
                "per_series": {
                    s: {
                        "pair_size": series_pair_size(s),
                        "elastic_cap": elastic_pair_cap(s),
                        "completed": _series_completed.get(s, 0),
                        "orphans": _series_orphans.get(s, 0),
                        "avg_spread": round(sum(_series_spreads.get(s, [])) / len(_series_spreads.get(s, [1])), 1) if _series_spreads.get(s) else 0,
                    } for s in CRYPTO_SERIES
                },
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
