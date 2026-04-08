"""
Crypto ML Trader v3: Multi-model voting system.

4 models vote independently. Trades only fire when 2+ agree.
  1. XGBoost (ML on 83 features)
  2. Momentum (pure price action)
  3. Mean Reversion (RSI/BB extremes)
  4. Kalshi Consensus (orderbook + velocity)

Run: python scripts/crypto_ml_trader.py
"""
import json
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

# === SINGLETON LOCK — sim and live get separate locks ===
from engine.process_lock import ProcessLock
import argparse as _ap
_pre_args = _ap.ArgumentParser(add_help=False)
_pre_args.add_argument("--simulate", action="store_true")
_pre_parsed, _ = _pre_args.parse_known_args()
_lock_name = "crypto_sim" if _pre_parsed.simulate else "crypto_live"
_lock = ProcessLock(_lock_name)
_lock.kill_existing()
if not _lock.acquire():
    print(f"FATAL: Another {_lock_name} is already running. Exiting.")
    sys.exit(1)

import numpy as np
import pandas as pd
import httpx
import xgboost as xgb

from kalshi.client import KalshiClient
from kalshi.market_discovery import KXBTCMarketTracker
from config.settings import settings
from models.signal_models import (
    XGBoostModel, MomentumModel, MeanReversionModel,
    KalshiConsensusModel, vote,
)
from models.trade_journal import TradeJournal
from engine.order_ledger import OrderLedger, OrderRecord
from engine.collector_health import check_collector_freshness
from engine.crypto_decision import (
    calibration_summary_view,
    evaluate_calibrated_trade,
    gross_ev_cents_per_contract,
    load_crypto_calibration,
    price_bucket_label,
)
from engine.session_tags import get_session_tag, is_live_session
from engine.alerts import alert_trade_placed, alert_settlement, alert_risk_halt
from engine.pre_trade_gate import PreTradeGate, GateContext
from engine.risk import RiskManager
from engine.family_limits import FamilyLimits

_log_file = "crypto_sim.log" if _pre_parsed.simulate else "crypto_live.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_file),
    ],
)
log = logging.getLogger("ml_trader")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Separate sim trade log — append-only, one line per trade for easy analysis
_sim_trade_log = Path(settings.DATA_DIR) / "sim_trades.log" if _pre_parsed.simulate else None

# Simulation mode — reuse the pre-parsed args from the lock section above
SIMULATE = _pre_parsed.simulate

# Config — looser thresholds in sim mode to generate more data
if SIMULATE:
    MAX_CONTRACTS = 10
    MAX_ENTRY_PRICE = 0.60  # sim can buy up to 60c (vs 45c live)
    SCAN_INTERVAL = 30
    DAILY_LOSS_LIMIT_CENTS = 99999  # no limit in sim
    MIN_BALANCE_FLOOR = 0.0  # no floor in sim
    SIM_MIN_AGREEMENT = 1  # lower: single-model signals fire for data generation
    SIM_MIN_EDGE = 0.01  # 1% edge minimum (vs 3% live)
else:
    MAX_CONTRACTS = 3
    MAX_ENTRY_PRICE = 0.45
    SCAN_INTERVAL = 30
    DAILY_LOSS_LIMIT_CENTS = 500
    MIN_BALANCE_FLOOR = 10.0
    SIM_MIN_AGREEMENT = 2.0
    SIM_MIN_EDGE = 0.03
STATE_FILE = Path(settings.DATA_DIR) / "ml_trader_state.json"

# Phase 1b: unified pre-trade gate.  Permissive defaults so the gate is purely
# additive — existing inline checks (edge, agreement, entry price cap) keep
# their authority; the gate adds a machine-readable reason code on top.
_GATE_RISK_MGR = RiskManager(
    max_contracts=10_000,
    daily_loss_limit_cents=10_000_000,
    consecutive_loss_halt=10_000,
)
_GATE_FAMILY_LIMITS = FamilyLimits()
_GATE_FAMILY_LIMITS._cooldown_s = 0.0  # disable cooldown — traders enforce their own pacing
_pre_trade_gate = PreTradeGate(
    risk_mgr=_GATE_RISK_MGR,
    family_limits=_GATE_FAMILY_LIMITS,
)
_gate_state: dict = {"last": None}


def get_latest_model_path():
    latest = Path("D:/kalshi-models/latest_model.json")
    if latest.exists():
        return str(latest)
    models = sorted(Path("D:/kalshi-models").glob("xgb_balanced_*.json"))
    if models:
        return str(models[-1])
    return "D:/kalshi-models/xgb_balanced_20260325_210315.json"


# State
traded_tickers: set = set()
settled_tickers: set = set()
resting_orders: dict[str, str] = {}  # ticker -> order_id
daily_pnl = 0
trade_count = 0
win_count = 0
loss_count = 0
last_reset_date: str = ""
price_history: dict[str, list[tuple[float, float]]] = {}


def get_btc_price():
    try:
        return float(httpx.get(
            "https://api.exchange.coinbase.com/products/BTC-USD/ticker", timeout=5
        ).json()["price"])
    except Exception:
        return None


def load_latest_features():
    """Load latest bars and compute honest 32-feature set."""
    from features.honest_features import compute_honest_features

    # Try 1m bars first (from collector)
    bars_dir = Path(settings.DATA_DIR) / "bars_1m"
    if bars_dir.exists():
        files = sorted(bars_dir.glob("*.parquet"))
        if files:
            try:
                df = pd.read_parquet(files[-1])
                if len(df) >= 65:  # need 60+ bars for feature warmup
                    featured = compute_honest_features(df)
                    if len(featured) >= 5:
                        return featured
            except Exception:
                pass

    # No fallback to legacy features — wait for honest bars warmup
    return None


def get_price_velocity(ticker: str, current_mid: float) -> float:
    now = time.time()
    if ticker not in price_history:
        price_history[ticker] = []
    price_history[ticker].append((now, current_mid))
    price_history[ticker] = price_history[ticker][-20:]
    entries = price_history[ticker]
    if len(entries) >= 5:
        oldest_t, oldest_p = entries[0]
        latest_t, latest_p = entries[-1]
        elapsed_min = (latest_t - oldest_t) / 60.0
        if elapsed_min >= 2.0:
            return (latest_p - oldest_p) / elapsed_min
    return 0.0


def get_kalshi_orderbook(c, ticker):
    try:
        ob = c.get_orderbook(ticker, depth=10)
        ob_data = ob.get("orderbook_fp", ob.get("orderbook", {}))
        yes_levels = ob_data.get("yes_dollars", [])
        no_levels = ob_data.get("no_dollars", [])
        yes_depth = sum(float(lvl[1]) for lvl in yes_levels)
        no_depth = sum(float(lvl[1]) for lvl in no_levels)
        total = yes_depth + no_depth
        if total > 0:
            return (yes_depth - no_depth) / total
    except Exception:
        pass
    return 0.0


def check_resting_orders(c, ledger=None):
    global trade_count, traded_tickers, resting_orders
    resolved = []
    for ticker, order_id in resting_orders.items():
        try:
            resp = c._request("GET", f"/portfolio/orders/{order_id}")
            order_data = resp.get("order", resp)
            status = order_data.get("status", "")
            if status == "executed":
                trade_count += 1
                ledger.update_status(ticker, "filled")
                ledger.save()
                resolved.append(ticker)
                log.info("Resting order FILLED: %s (%s)", ticker, order_id)
            elif status in ("canceled", "expired"):
                ledger.update_status(ticker, "cancelled")
                ledger.save()
                traded_tickers.discard(ticker)
                resolved.append(ticker)
                log.info("Resting order %s: %s (%s) — freed for retry",
                         status.upper(), ticker, order_id)
        except Exception as e:
            log.debug("Failed to check resting order %s: %s", order_id, e)
    for t in resolved:
        resting_orders.pop(t, None)


_journal: TradeJournal | None = None  # set in main()


def check_settlements(c, ledger=None):
    global daily_pnl, win_count, loss_count, settled_tickers

    if SIMULATE:
        # In simulation mode, check market results directly for our simulated tickers
        for ticker in list(traded_tickers - settled_tickers):
            try:
                mkt = c.get_market(ticker)
                m = mkt.get("market", mkt)
                if m.get("status") not in ("finalized", "settled"):
                    continue
                result_val = m.get("result", "")
                if not result_val:
                    continue

                # Find the journal entry for this ticker to get our simulated side
                sim_side = None
                sim_price = 50  # default
                if _journal:
                    for entry in reversed(_journal.entries):
                        if entry.get("ticker") == ticker and entry.get("action") == "simulated":
                            sim_side = entry.get("side")
                            sim_price = entry.get("entry_price", 50)
                            break

                if not sim_side:
                    settled_tickers.add(ticker)
                    continue

                won = (sim_side == "yes" and result_val == "yes") or \
                      (sim_side == "no" and result_val == "no")
                pnl = (100 - sim_price) if won else -sim_price

                daily_pnl += pnl
                if won:
                    win_count += 1
                else:
                    loss_count += 1
                settled_tickers.add(ticker)

                log.info("SIM SETTLED [%s] %s %s @ %dc pnl=%+dc total=%+dc",
                         "WIN" if won else "LOSS", ticker, sim_side.upper(), sim_price, pnl, daily_pnl)
                alert_settlement(ticker, "WIN" if won else "LOSS", pnl, strategy="crypto_sim")

                if ledger:
                    ledger.settle(ticker, result_val, pnl)
                    ledger.save()

                if _sim_trade_log:
                    with open(_sim_trade_log, "a") as f:
                        f.write(f"{datetime.now(timezone.utc).isoformat()} SETTLED {'WIN' if won else 'LOSS'} {sim_side.upper()} {ticker} @ {sim_price}c pnl={pnl:+d}c total={daily_pnl:+d}c W{win_count}/L{loss_count}\n")

                if _journal:
                    _journal.log_outcome(ticker, won, pnl)
                    _journal.save()
            except Exception:
                pass
        return

    # Live mode: check real settlements
    try:
        result = c._request("GET", "/portfolio/settlements", params={"limit": 10})
        for s in result.get("settlements", []):
            ticker = s.get("ticker", "")
            if ticker in settled_tickers or ticker not in traded_tickers:
                continue
            yes_ct = float(s.get("yes_count_fp", "0"))
            no_ct = float(s.get("no_count_fp", "0"))
            if not yes_ct and not no_ct:
                continue
            side = "YES" if yes_ct > 0 else "NO"
            # Read dollar amounts, convert to cents
            yes_cost_dollars = float(s.get("yes_total_cost_dollars", "0") or "0")
            no_cost_dollars = float(s.get("no_total_cost_dollars", "0") or "0")
            cost_cents = int((yes_cost_dollars if side == "YES" else no_cost_dollars) * 100)
            ct = yes_ct if side == "YES" else no_ct
            mkt_result = s.get("market_result", "")
            won = (side == "YES" and mkt_result == "yes") or (side == "NO" and mkt_result == "no")
            pnl = int(ct * 100 - cost_cents) if won else -cost_cents
            daily_pnl += pnl
            if won:
                win_count += 1
            else:
                loss_count += 1
            settled_tickers.add(ticker)
            log.info("SETTLED [%s] %s %s x%.0f pnl=%+dc total=%+dc",
                      "WIN" if won else "LOSS", ticker, side, ct, pnl, daily_pnl)
            if ledger:
                ledger.settle(ticker, mkt_result, pnl)
                ledger.save()
            # Feed outcome back to journal
            if _journal is not None:
                _journal.log_outcome(ticker, won, pnl)
                _journal.save()
    except Exception:
        pass


def write_state(state):
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, default=str))
    except Exception:
        pass


def build_signal_ref(decision_mode: str, calibration_version: str, side: str | None, price_bucket: str | None) -> str:
    if decision_mode == "calibrated_ev" and side and price_bucket:
        version = calibration_version or "unknown"
        return f"cal:{version}:{side}:{price_bucket}"
    return "legacy:xgb_momentum_conjunction"


def main():
    log.info("=" * 60)
    if SIMULATE:
        log.info("  CRYPTO ML TRADER v3 -- SIMULATION MODE (no real orders)")
    else:
        log.info("  CRYPTO ML TRADER v3 -- LIVE MODE")
    log.info("  Rule: XGBoost + Momentum conjunction")
    log.info("=" * 60)

    # Initialize models
    model_path = get_latest_model_path()
    xgb_model = XGBoostModel(model_path)
    momentum_model = MomentumModel()
    meanrev_model = MeanReversionModel()
    kalshi_model = KalshiConsensusModel()
    global _journal
    journal = TradeJournal(settings.DATA_DIR)
    _journal = journal

    c = KalshiClient()
    ledger = OrderLedger(settings.DATA_DIR)
    calibration = load_crypto_calibration(settings.CRYPTO_CALIBRATION_PATH)
    calibration_summary = calibration_summary_view(calibration)
    configured_decision_mode = (settings.CRYPTO_DECISION_MODE or "calibrated_ev").strip().lower()

    # SAFETY: In simulate mode, block place_order at the client level
    if SIMULATE:
        def _blocked_place_order(*args, **kwargs):
            raise RuntimeError("BLOCKED: place_order called in SIMULATE mode")
        c.place_order = _blocked_place_order
        log.info("SAFETY: place_order BLOCKED in simulation mode")

    tracker = KXBTCMarketTracker(c)
    log.info("Balance: $%.2f", c.get_balance())
    log.info("Journal: %d entries, weights=%s",
             len(journal.entries),
             {k: f"{v:.2f}" for k, v in journal.get_model_weights().items()})
    if calibration.get("exists"):
        log.info(
            "Calibration: %s | tradable=%s | generated=%s",
            calibration.get("path"),
            calibration_summary.get("tradable_buckets", 0),
            calibration_summary.get("generated_at", ""),
        )
    else:
        log.warning("Calibration artifact missing at %s", settings.CRYPTO_CALIBRATION_PATH)

    global traded_tickers, daily_pnl, trade_count, win_count, loss_count, last_reset_date, resting_orders
    scan_count = 0

    while True:
        try:
            scan_count += 1

            if scan_count == 1 or scan_count % 20 == 0:
                calibration = load_crypto_calibration(settings.CRYPTO_CALIBRATION_PATH)
                calibration_summary = calibration_summary_view(calibration)

            check_resting_orders(c, ledger)
            check_settlements(c, ledger)

            # Daily risk reset
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if today != last_reset_date:
                if last_reset_date:  # don't log on first iteration
                    log.info("Daily reset: W%d/L%d pnl=$%.2f", win_count, loss_count, daily_pnl / 100)
                daily_pnl = 0
                win_count = 0
                loss_count = 0
                trade_count = 0
                last_reset_date = today

            market = tracker.get_next_market()
            if not market:
                write_state({"time": datetime.now(timezone.utc).isoformat(), "status": "no_market"})
                time.sleep(SCAN_INTERVAL)
                continue

            remaining = tracker.get_market_time_remaining(market)
            ticker = market["ticker"]
            yes_ask = float(market.get("yes_ask_dollars", 0))
            no_ask = float(market.get("no_ask_dollars", 0))
            yes_bid = float(market.get("yes_bid_dollars", 0))
            btc = get_btc_price()
            market_mid = (yes_bid + yes_ask) / 2 if yes_ask > 0 else 0.5

            # --- Paired YES+NO arb check ---
            # If YES_ask + NO_ask < $1, buying both guarantees profit
            if yes_ask > 0 and no_ask > 0:
                pair_cost = yes_ask + no_ask
                pair_profit_cents = int((1.0 - pair_cost) * 100)
                if pair_profit_cents >= 3 and ticker not in traded_tickers:  # min 3c profit after fees
                    yes_cents = int(round(yes_ask * 100))
                    no_cents = int(round(no_ask * 100))
                    if SIMULATE:
                        log.info(">>> PAIR ARB SIM: %s YES@%dc + NO@%dc = %dc cost, +%dc guaranteed profit",
                                 ticker, yes_cents, no_cents, int(pair_cost * 100), pair_profit_cents)
                        if _sim_trade_log:
                            with open(_sim_trade_log, "a") as f:
                                f.write(f"{datetime.now(timezone.utc).isoformat()} PAIR_ARB {ticker} "
                                        f"YES@{yes_cents}c NO@{no_cents}c cost={int(pair_cost*100)}c "
                                        f"profit={pair_profit_cents}c BTC=${btc:,.0f}\n" if btc else "")
                        alert_trade_placed(ticker, "PAIR", int(pair_cost * 100), 1,
                                           pair_profit_cents, strategy="crypto_sim:pair_arb")
                        traded_tickers.add(ticker)
                    # Don't place real paired orders yet — needs atomic execution

            # --- Gather signals from all 4 models ---

            # Model 1: XGBoost
            features_df = load_latest_features()
            if features_df is not None:
                xgb_vote, xgb_conf = xgb_model.score(features_df)
                last_row = features_df.iloc[-1].to_dict()
            else:
                xgb_vote, xgb_conf = "flat", 0.50
                last_row = {}

            # Model 2: Momentum
            mom_vote, mom_conf = momentum_model.score(last_row)

            # Model 3: Mean Reversion
            mr_vote, mr_conf = meanrev_model.score(last_row)

            # Model 4: Kalshi Consensus
            kalshi_imb = get_kalshi_orderbook(c, ticker)
            velocity = get_price_velocity(ticker, market_mid)
            kalshi_vote, kalshi_conf = kalshi_model.score(kalshi_imb, velocity, market_mid)

            # --- XGB + Momentum conjunction (only tradeable signal) ---
            # MR and Kalshi kept for dashboard display only, not alpha
            legacy_probability = 0.485
            signal_rule = "xgb_mom_conjunction"
            if xgb_vote == mom_vote and xgb_vote != "flat":
                direction = xgb_vote
                confidence = legacy_probability
                agreement = 2
            elif SIMULATE and xgb_vote != "flat":
                # Single-model exploration: XGB only, for data generation
                direction = xgb_vote
                confidence = xgb_conf / 100.0
                agreement = 1
                signal_rule = "single_model_exploration"
            else:
                direction = "flat"
                confidence = 0.50
                agreement = 0

            # --- Determine action ---
            action = "monitoring"
            side = None
            entry = 0.0
            edge = 0.0
            price_bucket = None
            entry_price_cents = 0
            calibrated_p_win = None
            gross_ev_cents = None
            net_ev_cents = None
            bucket_trade_count = 0
            bucket_tradable = False

            # Guardrails
            try:
                current_bal = c.get_balance() if scan_count % 5 == 0 else None
            except Exception:
                current_bal = None

            collector = check_collector_freshness(settings.DATA_DIR, settings.COLLECTOR_STALE_SECONDS)
            session = get_session_tag()
            is_live = is_live_session(session, settings.CRYPTO_LIVE_SESSIONS)
            effective_simulate = SIMULATE or not is_live
            decision_mode = configured_decision_mode if effective_simulate else "legacy_conjunction"
            # Single-model exploration trades bypass calibration (no bucket data)
            if signal_rule == "single_model_exploration":
                decision_mode = "legacy_conjunction"

            if not collector["healthy"]:
                action = "collector_stale"
            elif ticker in traded_tickers:
                action = "already_traded"
            elif not SIMULATE and (remaining < 120 or remaining > 780):
                action = "outside_window"
            elif current_bal is not None and current_bal < MIN_BALANCE_FLOOR:
                action = "balance_floor"
            elif daily_pnl <= -DAILY_LOSS_LIMIT_CENTS:
                action = "loss_limit_hit"
            elif agreement < SIM_MIN_AGREEMENT:
                action = "no_consensus"
            else:
                if direction == "up" and 0 < yes_ask <= MAX_ENTRY_PRICE:
                    side = "yes"
                    entry = yes_ask
                elif direction == "down" and 0 < no_ask <= MAX_ENTRY_PRICE:
                    side = "no"
                    entry = no_ask

                if side:
                    entry_price_cents = int(round(entry * 100))
                    price_bucket = price_bucket_label(entry_price_cents)

                    if decision_mode == "calibrated_ev":
                        decision = evaluate_calibrated_trade(
                            calibration,
                            side,
                            entry_price_cents,
                            min_trades=settings.CRYPTO_CALIBRATION_MIN_TRADES,
                            ev_buffer_cents=settings.CRYPTO_EV_BUFFER_CENTS,
                            min_net_ev_cents=settings.CRYPTO_MIN_NET_EV_CENTS,
                        )
                        price_bucket = decision["price_bucket"]
                        if decision["status"] != "no_calibration":
                            calibrated_p_win = float(decision["calibrated_p_win"])
                            confidence = calibrated_p_win
                            bucket_trade_count = int(decision["bucket_trade_count"])
                            bucket_tradable = bool(decision["bucket_tradable"])
                            gross_ev_cents = float(decision["gross_ev_cents_per_contract"])
                            net_ev_cents = float(decision["net_ev_cents_per_contract"])
                            edge = calibrated_p_win - entry
                            action = decision["status"]
                        else:
                            action = "no_calibration"
                    else:
                        calibrated_p_win = legacy_probability
                        confidence = calibrated_p_win
                        gross_ev_cents = gross_ev_cents_per_contract(calibrated_p_win, entry_price_cents)
                        net_ev_cents = gross_ev_cents - float(settings.CRYPTO_EV_BUFFER_CENTS)
                        edge = calibrated_p_win - entry
                        if edge > SIM_MIN_EDGE:
                            action = "trading"
                        else:
                            action = "no_edge"
                else:
                    action = "no_edge"

            if action != "trading" and side:
                if decision_mode == "calibrated_ev" and action not in ("no_edge", "no_calibration"):
                    action = "no_edge"

            # --- Write state for dashboard ---
            state = {
                "time": datetime.now(timezone.utc).isoformat(),
                "scan": scan_count,
                "ticker": ticker,
                "remaining_s": int(remaining),
                "btc_price": btc,
                "models": {
                    "xgboost": {"vote": xgb_vote.upper(), "confidence": round(xgb_conf * 100, 1)},
                    "momentum": {"vote": mom_vote.upper(), "confidence": round(mom_conf * 100, 1)},
                    "mean_reversion": {"vote": mr_vote.upper(), "confidence": round(mr_conf * 100, 1)},
                    "kalshi_consensus": {"vote": kalshi_vote.upper(), "confidence": round(kalshi_conf * 100, 1)},
                },
                "prediction": direction.upper(),
                "agreement": agreement,
                "confidence": round(confidence * 100, 1),
                "collector_health": collector.get("healthy", True),
                "collector_age_s": collector.get("age_seconds"),
                "session_tag": session,
                "kalshi_imbalance": round(kalshi_imb, 3),
                "price_velocity": round(velocity * 100, 2),
                "yes_ask": yes_ask,
                "no_ask": no_ask,
                "edge": round(edge * 100, 1) if side else 0,
                "action": action,
                "trades_today": trade_count,
                "wins": win_count,
                "losses": loss_count,
                "rule": signal_rule,
                "decision_mode": decision_mode,
                "price_bucket": price_bucket,
                "entry_side": side,
                "entry_price_cents": entry_price_cents,
                "calibrated_p_win": round(calibrated_p_win, 4) if calibrated_p_win is not None else None,
                "gross_ev_cents_per_contract": round(gross_ev_cents, 3) if gross_ev_cents is not None else None,
                "net_ev_cents_per_contract": round(net_ev_cents, 3) if net_ev_cents is not None else None,
                "bucket_trade_count": bucket_trade_count,
                "bucket_tradable": bucket_tradable,
                "calibration_version": calibration.get("version", ""),
                "calibration_generated_at": calibration.get("generated_at", ""),
                "calibration_summary": calibration_summary,
                "daily_pnl_cents": daily_pnl,
                "balance": round(c.get_balance(), 2) if scan_count % 10 == 0 else None,
                "simulate": effective_simulate,
                "gate_last_decision": _gate_state["last"],
            }
            write_state(state)

            # --- Log ---
            votes_str = f"XGB={xgb_vote[0].upper()} MOM={mom_vote[0].upper()} MR={mr_vote[0].upper()} KAL={kalshi_vote[0].upper()}"
            log.info(
                "%s | %s %d/4 %.0f%% | %s | BTC=$%s mkt=%.0fc | edge=%.0f%% | %s | %s%s | W%d/L%d $%.2f",
                ticker[-8:], direction.upper(), agreement, confidence * 100,
                votes_str,
                f"{btc:,.0f}" if btc else "?",
                market_mid * 100,
                edge * 100 if side else 0,
                action,
                decision_mode,
                f" | EV={net_ev_cents:+.1f}c" if net_ev_cents is not None else "",
                win_count, loss_count, daily_pnl / 100,
            )

            # --- Journal every trade decision (not monitoring/outside_window) ---
            models_dict = {
                "xgboost": {"vote": xgb_vote.upper(), "confidence": round(xgb_conf * 100, 1)},
                "momentum": {"vote": mom_vote.upper(), "confidence": round(mom_conf * 100, 1)},
                "mean_reversion": {"vote": mr_vote.upper(), "confidence": round(mr_conf * 100, 1)},
                "kalshi_consensus": {"vote": kalshi_vote.upper(), "confidence": round(kalshi_conf * 100, 1)},
            }
            # Phase 1b: pre-trade gate (additive to existing inline checks).
            gate_decision = None
            gate_reason_code_kw: str | None = None
            gate_checks_json_kw: str | None = None
            if action == "trading" and side:
                try:
                    gate_ctx = GateContext(
                        ticker=ticker,
                        family="btc_15m",
                        side=side,
                        entry_cents=int(entry_price_cents or 0),
                        contracts=MAX_CONTRACTS,
                        yes_ask=float(yes_ask or 0),
                        no_ask=float(no_ask or 0),
                        model_prob=float(calibrated_p_win) if calibrated_p_win is not None else 0.5,
                        quote_age_s=0.0,
                        max_stale_s=60.0,
                        session_tag=session or "any",
                        strategy_tag="ml",
                        calibration_artifact=calibration if calibration.get("exists") else None,
                    )
                    gate_decision = _pre_trade_gate.evaluate(gate_ctx)
                except Exception:
                    log.exception("pre_trade_gate eval error — proceeding with inline checks only")
                    gate_decision = None
                if gate_decision is not None:
                    _gate_state["last"] = gate_decision.to_json()
                    gate_reason_code_kw = gate_decision.reason_code
                    gate_checks_json_kw = json.dumps(gate_decision.checks, default=str)
                    if not gate_decision.allowed:
                        log.info(
                            "GATE BLOCK [%s] %s: %s",
                            ticker, gate_decision.reason_code, gate_decision.reason_detail,
                        )
                        action = "gate_blocked"

            if action == "trading":
                contracts = MAX_CONTRACTS
                bet_dollars = contracts * entry_price_cents / 100
                journal.log_decision(
                    ticker=ticker, btc_price=btc or 0, models=models_dict,
                    vote_result=direction.upper(), agreement=agreement, action="pending",
                    side=side, entry_price=entry_price_cents, contracts=contracts,
                    bet_dollars=bet_dollars, edge=round(edge, 4) if side else None,
                    features_snapshot=last_row if last_row else None,
                    gate_reason_code=gate_reason_code_kw,
                    gate_checks_json=gate_checks_json_kw,
                    session_tag=session,
                    effective_simulate=effective_simulate,
                    rule_name=state["rule"],
                    entry_side=side,
                    entry_price_cents=entry_price_cents,
                    price_bucket=price_bucket,
                    calibrated_p_win=calibrated_p_win,
                    gross_ev_cents_per_contract=round(gross_ev_cents, 4) if gross_ev_cents is not None else None,
                    net_ev_cents_per_contract=round(net_ev_cents, 4) if net_ev_cents is not None else None,
                    bucket_trade_count=bucket_trade_count,
                    calibration_version=calibration.get("version", ""),
                )

            # --- Execute trade ---
            if action == "trading" and side:
                price_cents = entry_price_cents
                contracts = MAX_CONTRACTS
                bet_dollars = contracts * price_cents / 100  # actual notional
                signal_ref = build_signal_ref(decision_mode, calibration.get("version", ""), side, price_bucket)

                if effective_simulate:
                    # --- SIMULATION MODE: log but don't place ---
                    log.info(
                        ">>> SIM TRADE: %s %s @ %dc x%d ($%.2f) | %.0f/4 agree | edge=%.0f%% | %s",
                        side.upper(), ticker, price_cents, contracts, bet_dollars,
                        agreement, edge * 100, signal_ref,
                    )
                    # Write to dedicated sim trade log
                    if _sim_trade_log:
                        with open(_sim_trade_log, "a") as f:
                            f.write(f"{datetime.now(timezone.utc).isoformat()} TRADE {side.upper()} {ticker} @ {price_cents}c x{contracts} "
                                    f"edge={edge*100:.0f}% agree={agreement:.1f}/4 "
                                    f"bucket={price_bucket or 'n/a'} pwin={calibrated_p_win:.3f} netev={net_ev_cents:+.2f}c "
                                    f"XGB={xgb_vote[0].upper()} MOM={mom_vote[0].upper()} MR={mr_vote[0].upper()} KAL={kalshi_vote[0].upper()} "
                                    f"BTC=${btc:,.0f}\n" if btc else "")
                    traded_tickers.add(ticker)
                    trade_count += 1
                    alert_trade_placed(ticker, side, price_cents, contracts,
                                       edge * 100, strategy=f"crypto_sim:{signal_rule}")

                    ledger.add(OrderRecord(
                        strategy="crypto_sim",
                        market_type="btc_15m",
                        ticker=ticker,
                        status="simulated",
                        submitted_side=side,
                        submitted_price_cents=price_cents,
                        submitted_count=contracts,
                        session_tag=session,
                        signal_ref=signal_ref,
                    ))
                    ledger.save()

                    if journal.entries:
                        journal.entries[-1]["action"] = "simulated"
                        journal.entries[-1]["entry_price"] = price_cents
                        journal.entries[-1]["entry_price_cents"] = price_cents
                        journal.entries[-1]["contracts"] = contracts
                        journal.entries[-1]["bet_dollars"] = bet_dollars
                        journal.save()

                    # Track for settlement checking — we'll see if we would have won
                    # by checking the market result later
                else:
                    # --- LIVE MODE: place real order ---
                    log.info(
                        ">>> TRADE: %s %s @ %dc x%d ($%.2f) | %.0f/4 agree | edge=%.0f%% | %s",
                        side.upper(), ticker, price_cents, contracts, bet_dollars,
                        agreement, edge * 100, signal_ref,
                    )

                    try:
                        if side == "yes":
                            order = c.place_order(ticker, side="yes", action="buy",
                                                  count=contracts, order_type="limit",
                                                  yes_price=price_cents)
                        else:
                            order = c.place_order(ticker, side="no", action="buy",
                                                  count=contracts, order_type="limit",
                                                  no_price=price_cents)

                        status = order.get("order", order).get("status", "?")
                        log.info("    Order: %s", status)
                        if status == "executed":
                            traded_tickers.add(ticker)
                            trade_count += 1
                        elif status == "resting":
                            order_id = order.get("order", order).get("order_id", "")
                            if order_id:
                                resting_orders[ticker] = order_id
                            traded_tickers.add(ticker)

                        ledger.add(OrderRecord(
                            strategy="crypto",
                            market_type="btc_15m",
                            ticker=ticker,
                            order_id=order.get("order", order).get("order_id", ""),
                            status="filled" if status == "executed" else status,
                            submitted_side=side,
                            submitted_price_cents=price_cents,
                            submitted_count=contracts,
                            filled_price_cents=price_cents if status == "executed" else 0,
                            filled_count=contracts if status == "executed" else 0,
                            session_tag=session,
                            signal_ref=signal_ref,
                        ))
                        ledger.save()

                        if journal.entries:
                            journal.entries[-1]["action"] = "executed" if status == "executed" else status
                            journal.entries[-1]["entry_price"] = price_cents
                            journal.entries[-1]["entry_price_cents"] = price_cents
                            journal.entries[-1]["contracts"] = contracts
                            journal.entries[-1]["bet_dollars"] = bet_dollars
                            journal.save()

                        # Dashboard notification
                        notif = {
                            "type": "trade",
                            "time": datetime.now(timezone.utc).isoformat(),
                            "ticker": ticker, "side": side.upper(),
                            "price": price_cents, "contracts": contracts,
                            "bet": bet_dollars, "agreement": agreement,
                            "confidence": round(confidence * 100, 1),
                            "edge": round(edge * 100, 1),
                            "status": status, "btc_price": btc,
                        }
                        try:
                            notif_path = Path(settings.DATA_DIR) / "notifications.json"
                            notif_path.write_text(json.dumps(notif, default=str))
                        except Exception:
                            pass

                    except Exception as e:
                        log.error("    Order FAILED: %s", e)
                        if journal.entries:
                            journal.entries[-1]["action"] = "failed"
                            journal.save()

            # Save journal periodically
            if scan_count % 10 == 0:
                journal.save()

            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception:
            log.exception("Scan error")
            time.sleep(10)

    journal.save()
    stats = journal.get_stats()
    log.info("Shutdown. %d trades, W%d/L%d, pnl=$%.2f",
             trade_count, win_count, loss_count, daily_pnl / 100)
    log.info("Journal stats: %s", stats)


if __name__ == "__main__":
    main()
