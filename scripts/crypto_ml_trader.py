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
from engine.session_tags import get_session_tag, is_live_session

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
    SIM_MIN_AGREEMENT = 1.5  # lower agreement threshold for sim
    SIM_MIN_EDGE = 0.01  # 1% edge minimum (vs 3% live)
else:
    MAX_CONTRACTS = 30
    MAX_ENTRY_PRICE = 0.45
    SCAN_INTERVAL = 30
    DAILY_LOSS_LIMIT_CENTS = 1500
    MIN_BALANCE_FLOOR = 10.0
    SIM_MIN_AGREEMENT = 2.0
    SIM_MIN_EDGE = 0.03
STATE_FILE = Path(settings.DATA_DIR) / "ml_trader_state.json"


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


def main():
    log.info("=" * 60)
    if SIMULATE:
        log.info("  CRYPTO ML TRADER v3 -- SIMULATION MODE (no real orders)")
    else:
        log.info("  CRYPTO ML TRADER v3 -- LIVE MODE")
    log.info("  Models: XGBoost + Momentum + MeanReversion + KalshiConsensus")
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

    global traded_tickers, daily_pnl, trade_count, win_count, loss_count, last_reset_date, resting_orders
    scan_count = 0

    while True:
        try:
            scan_count += 1

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
            if xgb_vote == mom_vote and xgb_vote != "flat":
                direction = xgb_vote
                confidence = 0.485  # empirical P(win | XGB+MOM agree) from 90-day backtest
                agreement = 2
            else:
                direction = "flat"
                confidence = 0.50
                agreement = 0

            # --- Determine action ---
            action = "monitoring"
            side = None
            entry = 0.0
            edge = 0.0

            # Guardrails
            try:
                current_bal = c.get_balance() if scan_count % 5 == 0 else None
            except Exception:
                current_bal = None

            collector = check_collector_freshness(settings.DATA_DIR, settings.COLLECTOR_STALE_SECONDS)
            session = get_session_tag()
            is_live = is_live_session(session, settings.CRYPTO_LIVE_SESSIONS)
            effective_simulate = SIMULATE or not is_live

            if not collector["healthy"]:
                action = "collector_stale"
            elif ticker in traded_tickers:
                action = "already_traded"
            elif remaining < 120 or remaining > 780:
                action = "outside_window"
            elif current_bal is not None and current_bal < MIN_BALANCE_FLOOR:
                action = "balance_floor"
            elif daily_pnl <= -DAILY_LOSS_LIMIT_CENTS:
                action = "loss_limit_hit"
            elif agreement < 2:
                action = "no_consensus"
            elif direction == "up" and 0 < yes_ask <= MAX_ENTRY_PRICE:
                side = "yes"
                entry = yes_ask
                edge = confidence - yes_ask
                if edge > SIM_MIN_EDGE:
                    action = "trading"
            elif direction == "down" and 0 < no_ask <= MAX_ENTRY_PRICE:
                side = "no"
                entry = no_ask
                edge = confidence - no_ask
                if edge > SIM_MIN_EDGE:
                    action = "trading"

            if action != "trading" and side:
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
                "rule": "xgb+mom conjunction",
                "daily_pnl_cents": daily_pnl,
                "balance": round(c.get_balance(), 2) if scan_count % 10 == 0 else None,
                "simulate": effective_simulate,
            }
            write_state(state)

            # --- Log ---
            votes_str = f"XGB={xgb_vote[0].upper()} MOM={mom_vote[0].upper()} MR={mr_vote[0].upper()} KAL={kalshi_vote[0].upper()}"
            log.info(
                "%s | %s %d/4 %.0f%% | %s | BTC=$%s mkt=%.0fc | edge=%.0f%% | %s | W%d/L%d $%.2f",
                ticker[-8:], direction.upper(), agreement, confidence * 100,
                votes_str,
                f"{btc:,.0f}" if btc else "?",
                market_mid * 100,
                edge * 100 if side else 0,
                action, win_count, loss_count, daily_pnl / 100,
            )

            # --- Journal every trade decision (not monitoring/outside_window) ---
            models_dict = {
                "xgboost": {"vote": xgb_vote.upper(), "confidence": round(xgb_conf * 100, 1)},
                "momentum": {"vote": mom_vote.upper(), "confidence": round(mom_conf * 100, 1)},
                "mean_reversion": {"vote": mr_vote.upper(), "confidence": round(mr_conf * 100, 1)},
                "kalshi_consensus": {"vote": kalshi_vote.upper(), "confidence": round(kalshi_conf * 100, 1)},
            }
            if action == "trading":
                journal.log_decision(
                    ticker=ticker, btc_price=btc or 0, models=models_dict,
                    vote_result=direction.upper(), agreement=agreement, action="pending",
                    side=side, edge=round(edge, 4) if side else None,
                    features_snapshot=last_row if last_row else None,
                )

            # --- Execute trade ---
            if action == "trading" and side:
                price_cents = int(entry * 100)
                contracts = 10  # fixed 10 contracts per trade
                bet_dollars = contracts * price_cents / 100  # actual notional

                if effective_simulate:
                    # --- SIMULATION MODE: log but don't place ---
                    log.info(
                        ">>> SIM TRADE: %s %s @ %dc x%d ($%.1f) | %.0f/4 agree | edge=%.0f%%",
                        side.upper(), ticker, price_cents, contracts, bet_dollars,
                        agreement, edge * 100,
                    )
                    # Write to dedicated sim trade log
                    if _sim_trade_log:
                        with open(_sim_trade_log, "a") as f:
                            f.write(f"{datetime.now(timezone.utc).isoformat()} TRADE {side.upper()} {ticker} @ {price_cents}c x{contracts} "
                                    f"edge={edge*100:.0f}% agree={agreement:.1f}/4 "
                                    f"XGB={xgb_vote[0].upper()} MOM={mom_vote[0].upper()} MR={mr_vote[0].upper()} KAL={kalshi_vote[0].upper()} "
                                    f"BTC=${btc:,.0f}\n" if btc else "")
                    traded_tickers.add(ticker)
                    trade_count += 1

                    ledger.add(OrderRecord(
                        strategy="crypto_sim",
                        market_type="btc_15m",
                        ticker=ticker,
                        status="simulated",
                        submitted_side=side,
                        submitted_price_cents=price_cents,
                        submitted_count=contracts,
                        session_tag=session,
                    ))
                    ledger.save()

                    if journal.entries:
                        journal.entries[-1]["action"] = "simulated"
                        journal.entries[-1]["entry_price"] = price_cents
                        journal.entries[-1]["contracts"] = contracts
                        journal.entries[-1]["bet_dollars"] = bet_dollars
                        journal.save()

                    # Track for settlement checking — we'll see if we would have won
                    # by checking the market result later
                else:
                    # --- LIVE MODE: place real order ---
                    log.info(
                        ">>> TRADE: %s %s @ %dc x%d ($%.1f) | %.0f/4 agree | edge=%.0f%%",
                        side.upper(), ticker, price_cents, contracts, bet_dollars,
                        agreement, edge * 100,
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
                        ))
                        ledger.save()

                        if journal.entries:
                            journal.entries[-1]["action"] = "executed" if status == "executed" else status
                            journal.entries[-1]["entry_price"] = price_cents
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
