"""
Crypto ML Trader: Uses retrained XGBoost model with microstructure features
to trade BTC 15-min markets on Kalshi. Small bets, strong edges only.

Writes live state to D:/kalshi-data/ml_trader_state.json for dashboard.

Run: python scripts/crypto_ml_trader.py
"""
import json
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import httpx
import xgboost as xgb

from kalshi.client import KalshiClient
from kalshi.market_discovery import KXBTCMarketTracker
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("crypto_ml_trader.log"),
    ],
)
log = logging.getLogger("ml_trader")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Config
MAX_CONTRACTS = 30
CONFIDENCE_THRESHOLD = 0.58
MAX_ENTRY_PRICE = 0.62
SCAN_INTERVAL = 30
DAILY_LOSS_LIMIT_CENTS = 500
MODEL_PATH = "D:/kalshi-models/xgb_balanced_20260325_210315.json"
STATE_FILE = Path(settings.DATA_DIR) / "ml_trader_state.json"

traded_tickers: set = set()
settled_tickers: set = set()
daily_pnl = 0
trade_count = 0
win_count = 0
loss_count = 0


def load_model():
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    log.info("Loaded model from %s", MODEL_PATH)
    return booster


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
        return df if len(df) >= 5 else None
    except Exception:
        return None


def predict_direction(booster, features_df):
    exclude = {"timestamp", "btc_price", "label"}
    feature_cols = [c for c in features_df.columns if c not in exclude]
    X = features_df[feature_cols].iloc[-1:].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    dmat = xgb.DMatrix(X)
    proba = booster.predict(dmat)[0]
    p_down, p_flat, p_up = float(proba[0]), float(proba[1]), float(proba[2])

    if p_up > p_down and p_up > p_flat:
        return "up", p_up, (p_down, p_flat, p_up)
    elif p_down > p_up and p_down > p_flat:
        return "down", p_down, (p_down, p_flat, p_up)
    else:
        return "flat", p_flat, (p_down, p_flat, p_up)


def check_settlements(c):
    """Check for new settlements and update P&L."""
    global daily_pnl, win_count, loss_count, settled_tickers
    try:
        result = c._request("GET", "/portfolio/settlements", params={"limit": 10})
        for s in result.get("settlements", []):
            ticker = s.get("ticker", "")
            if ticker in settled_tickers or ticker not in traded_tickers:
                continue

            yes_ct = float(s.get("yes_count_fp", "0"))
            no_ct = float(s.get("no_count_fp", "0"))
            side = "YES" if yes_ct > 0 else "NO"
            cost = s.get("yes_total_cost", 0) if side == "YES" else s.get("no_total_cost", 0)
            ct = int(yes_ct if side == "YES" else no_ct)
            mkt_result = s.get("market_result", "")
            won = (side == "YES" and mkt_result == "yes") or (side == "NO" and mkt_result == "no")
            pnl = (ct * 100 - cost) if won else -cost

            daily_pnl += pnl
            if won:
                win_count += 1
            else:
                loss_count += 1
            settled_tickers.add(ticker)

            mark = "WIN" if won else "LOSS"
            log.info(
                "  SETTLED [%s] %s %s x%d pnl=%+dc total_pnl=%+dc",
                mark, ticker, side, ct, pnl, daily_pnl,
            )
    except Exception:
        pass


def write_state(state):
    """Write current state to JSON for dashboard."""
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, default=str))
    except Exception:
        pass


def main():
    log.info("=" * 55)
    log.info("  CRYPTO ML TRADER v2 — Live State + Settlement Tracking")
    log.info("  Model: %s", MODEL_PATH)
    log.info("=" * 55)

    model = load_model()
    c = KalshiClient()
    tracker = KXBTCMarketTracker(c)
    bal = c.get_balance()
    log.info("Balance: $%.2f", bal)

    global traded_tickers, daily_pnl, trade_count, win_count, loss_count
    scan_count = 0

    while True:
        try:
            scan_count += 1

            # Check settlements every cycle
            check_settlements(c)

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

            # --- Kalshi orderbook features ---
            kalshi_signal = 0  # -1 bearish, 0 neutral, +1 bullish
            kalshi_imbalance = 0.0
            try:
                ob = c.get_orderbook(ticker, depth=10)
                ob_data = ob.get("orderbook_fp", ob.get("orderbook", {}))
                yes_levels = ob_data.get("yes_dollars", [])
                no_levels = ob_data.get("no_dollars", [])

                yes_depth = sum(float(lvl[1]) for lvl in yes_levels)
                no_depth = sum(float(lvl[1]) for lvl in no_levels)
                total_depth = yes_depth + no_depth

                if total_depth > 0:
                    # Imbalance: positive = more YES bids = bullish sentiment
                    kalshi_imbalance = (yes_depth - no_depth) / total_depth

                    # Strong signal when one side dominates
                    if kalshi_imbalance > 0.3:
                        kalshi_signal = 1  # market leans UP
                    elif kalshi_imbalance < -0.3:
                        kalshi_signal = -1  # market leans DOWN
            except Exception:
                pass

            # --- Market price momentum (compare to 50/50 baseline) ---
            market_mid = (yes_bid + yes_ask) / 2 if yes_ask > 0 else 0.5
            market_lean = "up" if market_mid > 0.55 else "down" if market_mid < 0.45 else "neutral"

            # --- Load ML features + predict ---
            features_df = load_latest_features()
            direction, confidence, probs = "flat", 0.5, (0.33, 0.34, 0.33)
            if features_df is not None:
                direction, confidence, probs = predict_direction(model, features_df)

            # --- Combine ML model + Kalshi orderbook signal ---
            # Boost confidence when Kalshi orderbook agrees with model
            # Reduce confidence when they disagree
            combined_confidence = confidence
            if (direction == "up" and kalshi_signal > 0) or (direction == "down" and kalshi_signal < 0):
                combined_confidence = min(0.99, confidence + 0.05)  # agreement bonus
            elif (direction == "up" and kalshi_signal < 0) or (direction == "down" and kalshi_signal > 0):
                combined_confidence = max(0.40, confidence - 0.10)  # disagreement penalty

            # Determine action
            action = "monitoring"
            side = None
            entry = 0.0
            edge = 0.0

            if ticker in traded_tickers:
                action = "already_traded"
            elif remaining < 120 or remaining > 780:
                action = "outside_window"
            elif daily_pnl <= -DAILY_LOSS_LIMIT_CENTS:
                action = "loss_limit_hit"
            elif direction == "up" and combined_confidence >= CONFIDENCE_THRESHOLD and 0 < yes_ask <= MAX_ENTRY_PRICE:
                side = "yes"
                entry = yes_ask
                edge = combined_confidence - yes_ask
                if edge > 0.03:
                    action = "trading"
            elif direction == "down" and combined_confidence >= CONFIDENCE_THRESHOLD and 0 < no_ask <= MAX_ENTRY_PRICE:
                side = "no"
                entry = no_ask
                edge = combined_confidence - no_ask
                if edge > 0.03:
                    action = "trading"

            if action != "trading" and side:
                action = "no_edge"

            # Write state for dashboard
            state = {
                "time": datetime.now(timezone.utc).isoformat(),
                "scan": scan_count,
                "ticker": ticker,
                "remaining_s": int(remaining),
                "btc_price": btc,
                "prediction": direction.upper(),
                "confidence": round(confidence * 100, 1),
                "p_down": round(probs[0] * 100, 1),
                "p_flat": round(probs[1] * 100, 1),
                "p_up": round(probs[2] * 100, 1),
                "yes_ask": yes_ask,
                "no_ask": no_ask,
                "edge": round(edge * 100, 1) if side else 0,
                "kalshi_imbalance": round(kalshi_imbalance, 3),
                "kalshi_signal": kalshi_signal,
                "market_lean": market_lean,
                "combined_confidence": round(combined_confidence * 100, 1),
                "action": action,
                "trades_today": trade_count,
                "wins": win_count,
                "losses": loss_count,
                "daily_pnl_cents": daily_pnl,
                "balance": round(c.get_balance(), 2) if scan_count % 10 == 0 else None,
            }
            write_state(state)

            # Log every scan (concise)
            kalshi_arrow = "+" if kalshi_signal > 0 else "-" if kalshi_signal < 0 else "="
            log.info(
                "%s | %s %.0f%%→%.0f%% | BTC=$%s | mkt=%.0fc kalshi=%s%.0f%% | edge=%.0f%% | %s | W%d/L%d $%.2f",
                ticker[-8:], direction.upper(), confidence * 100, combined_confidence * 100,
                f"{btc:,.0f}" if btc else "?",
                market_mid * 100, kalshi_arrow, abs(kalshi_imbalance) * 100,
                edge * 100 if side else 0,
                action, win_count, loss_count, daily_pnl / 100,
            )

            # Execute trade
            if action == "trading" and side:
                price_cents = int(entry * 100)
                # Scale bet by confidence: $3 at 58%, up to $15 at 90%+
                bet_dollars = min(15, max(3, int((confidence - 0.50) * 50)))
                bet_cents = bet_dollars * 100
                contracts = max(1, min(MAX_CONTRACTS, bet_cents // price_cents))

                log.info(
                    ">>> TRADE: %s %s @ %dc x%d | conf=%.0f%% edge=%.0f%%",
                    side.upper(), ticker, price_cents, contracts,
                    confidence * 100, edge * 100,
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
                    traded_tickers.add(ticker)
                    trade_count += 1

                except Exception as e:
                    log.error("    Order FAILED: %s", e)

            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception:
            log.exception("Scan error")
            time.sleep(10)

    log.info("Shutdown. %d trades, %dW/%dL, pnl=$%.2f", trade_count, win_count, loss_count, daily_pnl / 100)


if __name__ == "__main__":
    main()
