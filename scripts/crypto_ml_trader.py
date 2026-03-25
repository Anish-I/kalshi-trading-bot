"""
Crypto ML Trader: Uses retrained XGBoost model with microstructure features
to trade BTC 15-min markets on Kalshi. Small bets, strong edges only.

Run: python scripts/crypto_ml_trader.py
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

# Config
MAX_CONTRACTS = 5
CONFIDENCE_THRESHOLD = 0.58  # model must be >58% confident in direction
MAX_ENTRY_PRICE = 0.62       # allow up to 62c if model is very confident
SCAN_INTERVAL = 30
DAILY_LOSS_LIMIT_CENTS = 500  # $5
MODEL_PATH = "D:/kalshi-models/xgb_deploy_20260325_060009.json"

traded_tickers: set = set()
daily_pnl = 0
trade_count = 0
win_count = 0
loss_count = 0


def load_model():
    """Load the retrained XGBoost model as Booster."""
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    log.info("Loaded XGBoost Booster from %s", MODEL_PATH)
    return booster


def get_btc_price():
    try:
        resp = httpx.get("https://api.exchange.coinbase.com/products/BTC-USD/ticker", timeout=5)
        return float(resp.json()["price"])
    except Exception:
        return None


def load_latest_features() -> pd.DataFrame | None:
    """Load latest feature row from parquet."""
    feat_dir = Path(settings.DATA_DIR) / "features"
    if not feat_dir.exists():
        return None
    files = sorted(feat_dir.glob("*.parquet"))
    if not files:
        return None
    try:
        df = pd.read_parquet(files[-1])
        if len(df) < 5:
            return None
        return df
    except Exception:
        return None


def predict_direction(booster: xgb.Booster, features_df: pd.DataFrame) -> tuple[str, float]:
    """Run model inference on latest features.

    Returns: (direction, confidence)
        direction: "up", "down", or "flat"
        confidence: probability of the predicted direction
    """
    exclude = {"timestamp", "btc_price", "label"}
    feature_cols = [c for c in features_df.columns if c not in exclude]

    X = features_df[feature_cols].iloc[-1:].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    dmat = xgb.DMatrix(X)

    # Predict probabilities: [p_down, p_flat, p_up] (classes 0,1,2)
    proba = booster.predict(dmat)[0]
    p_down, p_flat, p_up = float(proba[0]), float(proba[1]), float(proba[2])

    if p_up > p_down and p_up > p_flat:
        return "up", p_up
    elif p_down > p_up and p_down > p_flat:
        return "down", p_down
    else:
        return "flat", p_flat


def main():
    log.info("=" * 55)
    log.info("  CRYPTO ML TRADER — XGBoost with Microstructure")
    log.info("  Model: %s", MODEL_PATH)
    log.info("  Confidence threshold: %.0f%%", CONFIDENCE_THRESHOLD * 100)
    log.info("  Max contracts: %d | Max entry: %dc", MAX_CONTRACTS, int(MAX_ENTRY_PRICE * 100))
    log.info("=" * 55)

    model = load_model()
    c = KalshiClient()
    tracker = KXBTCMarketTracker(c)

    bal = c.get_balance()
    log.info("Balance: $%.2f", bal)

    global traded_tickers, daily_pnl, trade_count, win_count, loss_count
    scan_count = 0
    last_status = time.time()

    while True:
        try:
            scan_count += 1
            market = tracker.get_next_market()

            if not market:
                time.sleep(SCAN_INTERVAL)
                continue

            remaining = tracker.get_market_time_remaining(market)
            ticker = market["ticker"]

            # Only trade 2-13 min before close
            if remaining < 120 or remaining > 780:
                time.sleep(SCAN_INTERVAL)
                continue

            # Already traded this market
            if ticker in traded_tickers:
                time.sleep(SCAN_INTERVAL)
                continue

            # Daily loss limit
            if daily_pnl <= -DAILY_LOSS_LIMIT_CENTS:
                if time.time() - last_status > 300:
                    log.warning("Daily loss limit hit ($%.2f). Sitting out.", daily_pnl / 100)
                    last_status = time.time()
                time.sleep(SCAN_INTERVAL)
                continue

            # Load features
            features_df = load_latest_features()
            if features_df is None:
                time.sleep(SCAN_INTERVAL)
                continue

            # Model prediction
            direction, confidence = predict_direction(model, features_df)
            btc = get_btc_price()

            # Get market prices
            yes_ask = float(market.get("yes_ask_dollars", 0))
            no_ask = float(market.get("no_ask_dollars", 0))
            yes_bid = float(market.get("yes_bid_dollars", 0))
            no_bid = float(market.get("no_bid_dollars", 0))

            # Decide trade
            side = None
            entry = 0.0
            edge = 0.0

            if direction == "up" and confidence >= CONFIDENCE_THRESHOLD:
                if yes_ask <= MAX_ENTRY_PRICE and yes_ask > 0:
                    side = "yes"
                    entry = yes_ask
                    edge = confidence - yes_ask

            elif direction == "down" and confidence >= CONFIDENCE_THRESHOLD:
                if no_ask <= MAX_ENTRY_PRICE and no_ask > 0:
                    side = "no"
                    entry = no_ask
                    edge = confidence - no_ask

            # Status every 5 min
            if time.time() - last_status > 300:
                log.info(
                    "Scan #%d | %s %ds | %s %.0f%% | BTC=$%s | trades=%d W=%d L=%d pnl=$%.2f",
                    scan_count, ticker, int(remaining),
                    direction.upper(), confidence * 100,
                    f"{btc:,.2f}" if btc else "?",
                    trade_count, win_count, loss_count, daily_pnl / 100,
                )
                last_status = time.time()

            # Execute trade if edge exists
            if side and edge > 0.03:
                price_cents = int(entry * 100)
                # Target ~$3 per trade: contracts = 300 / price_cents
                contracts = max(1, min(MAX_CONTRACTS, 300 // price_cents))

                log.info(
                    ">>> TRADE: %s %s @ %dc x%d | model=%s %.1f%% | edge=%.0f%% | BTC=$%s",
                    side.upper(), ticker, price_cents, contracts,
                    direction, confidence * 100, edge * 100,
                    f"{btc:,.2f}" if btc else "?",
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
                    log.info("    Order: %s | Balance: $%.2f", status, c.get_balance())

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

    log.info("Shutting down. Trades: %d", trade_count)


if __name__ == "__main__":
    main()
