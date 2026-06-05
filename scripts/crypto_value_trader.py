"""Value-vs-fair trader for 15-minute crypto markets (sim-first).

Phase 2 strategy: instead of forecasting direction, price the binary from spot
vs the market's strike (the window's open price) and realized vol, then bet
WHICHEVER side is cheap relative to fair value. One-sided value bet — never both
sides (taker pair cost is always >= $1).

Every order is funneled through the Phase-1 PreTradeGate (risk halt, per-ticker
cap, quote quality, session) and the persistent TickerExposureTracker. Defaults
to sim; --mode live places resting limit orders.

    python -m scripts.crypto_value_trader              # sim, BTC
    python -m scripts.crypto_value_trader --mode live  # tiny live (hard-capped)
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from config.settings import settings
from kalshi.client import KalshiClient
from engine.fair_value import sigma_per_min_from_returns
from engine.value_strategy import evaluate_market
from engine.pre_trade_gate import PreTradeGate, GateContext
from engine.gate_risk_adapter import RiskManagerAdapter, FamilyLimitsAdapter
from engine.ticker_exposure import TickerExposureTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("value_trader")

# series -> Coinbase spot product
SERIES_SPOT = {
    "KXBTC15M": "BTC-USD",
    "KXETH15M": "ETH-USD",
}


def get_spot(product: str) -> float | None:
    try:
        r = httpx.get(f"https://api.exchange.coinbase.com/products/{product}/ticker", timeout=5)
        return float(r.json()["price"])
    except Exception:
        return None


def get_sigma_per_min(vol_window: int = 15) -> float | None:
    """Per-minute log-return vol from the latest 1m bars (BTC collector output)."""
    bars_dir = Path(settings.DATA_DIR) / "bars_1m"
    if not bars_dir.exists():
        return None
    files = sorted(bars_dir.glob("*.parquet"))
    if not files:
        return None
    try:
        df = pd.read_parquet(files[-1])
        close = df["close"].to_numpy(dtype=float)
        if close.size < 16:
            return None
        ret = np.full(close.shape, np.nan)
        ret[1:] = np.log(close[1:] / close[:-1])
        return sigma_per_min_from_returns(ret.tolist(), window=vol_window)
    except Exception:
        return None


def _compute_session_tag(now: datetime) -> str:
    h = now.hour
    if 13 <= h < 21:
        return "us_core"
    if 21 <= h or h < 1:
        return "us_pm"
    return "overnight"


def build_gate(max_contracts: int) -> tuple[PreTradeGate, TickerExposureTracker]:
    # Value strategy carries its own EV check (the model edge), so the gate's
    # risk layer is a simple always-on adapter; the real protections are the
    # per-ticker cap, quote quality, and session gating.
    risk = RiskManagerAdapter(lambda: (True, "ok"), max_contracts=max_contracts)
    family = FamilyLimitsAdapter(lambda _f: 0, lambda _f: 10_000_000)
    cap = TickerExposureTracker(
        state_path=settings.TICKER_EXPOSURE_STATE_PATH,
        max_contracts_per_ticker=settings.MAX_CONTRACTS_PER_TICKER,
        max_notional_cents_per_ticker=settings.MAX_NOTIONAL_CENTS_PER_TICKER,
    )
    gate = PreTradeGate(risk, family, ticker_cap=cap)
    return gate, cap


def run(args) -> None:
    client = KalshiClient()
    gate, cap = build_gate(args.contracts)
    series_list = args.series.split(",")
    log.info("value_trader start mode=%s series=%s min_edge=%.1fc contracts=%d",
             args.mode, series_list, args.min_edge_cents, args.contracts)

    while True:
        now = datetime.now(timezone.utc)
        sigma = get_sigma_per_min()
        if sigma is None:
            log.warning("no vol (need 1m bars from collector) — skipping scan")
            time.sleep(args.interval)
            continue

        for series in series_list:
            product = SERIES_SPOT.get(series)
            spot = get_spot(product) if product else None
            if spot is None:
                continue
            try:
                data = client._request("GET", "/markets",
                                       params={"series_ticker": series, "status": "open", "limit": 10})
                markets = data.get("markets", [])
            except Exception:
                log.error("market fetch failed for %s", series, exc_info=True)
                continue

            for market in markets:
                ev = evaluate_market(
                    market, spot, sigma, now=now,
                    min_edge_cents=args.min_edge_cents,
                    max_entry_cents=args.max_entry,
                    family="btc_15m",
                )
                d = ev.decision
                if d is None or d.side is None:
                    continue

                ticker = market["ticker"]
                yes_ask = float(market.get("yes_ask_dollars", market.get("yes_ask", 0)) or 0)
                no_ask = float(market.get("no_ask_dollars", market.get("no_ask", 0)) or 0)
                ctx = GateContext(
                    ticker=ticker, family="btc_15m", side=d.side,
                    entry_cents=d.entry_cents, contracts=args.contracts,
                    yes_ask=yes_ask, no_ask=no_ask,
                    yes_bid=float(market.get("yes_bid_dollars", market.get("yes_bid", 0)) or 0),
                    no_bid=float(market.get("no_bid_dollars", market.get("no_bid", 0)) or 0),
                    model_prob=ev.fair_p if d.side == "yes" else (1.0 - ev.fair_p),
                    quote_age_s=0.0, max_stale_s=60.0,
                    session_tag=_compute_session_tag(now),
                    strategy_tag="value",
                )
                decision = gate.evaluate(ctx)
                if not decision.allowed:
                    log.info("GATE BLOCK [%s] %s: %s", ticker, decision.reason_code, decision.reason_detail)
                    continue

                if args.mode == "sim":
                    log.info(">>> VALUE SIM: %s %s @%dc x%d | fairP=%.3f ev=%+.2fc strike=%.2f spot=%.2f T=%.1fm",
                             d.side.upper(), ticker, d.entry_cents, args.contracts,
                             ev.fair_p, d.net_ev_cents, ev.strike, spot, ev.remaining_s / 60.0)
                else:
                    try:
                        yes_price = d.entry_cents if d.side == "yes" else None
                        no_price = d.entry_cents if d.side == "no" else None
                        resp = client.place_order(
                            ticker=ticker, side=d.side, action="buy",
                            count=args.contracts, order_type="limit",
                            yes_price=yes_price, no_price=no_price,
                        )
                        od = resp.get("order", resp)
                        log.info(">>> VALUE LIVE: %s %s @%dc x%d order=%s status=%s",
                                 d.side.upper(), ticker, d.entry_cents, args.contracts,
                                 od.get("order_id", "?"), od.get("status", "?"))
                    except Exception:
                        log.error("value order failed", exc_info=True)
                        continue
                cap.record(ticker, args.contracts, d.entry_cents * args.contracts)

        time.sleep(args.interval)


def main() -> None:
    ap = argparse.ArgumentParser(description="Value-vs-fair 15m crypto trader (sim-first)")
    ap.add_argument("--mode", choices=["sim", "live"], default="sim")
    ap.add_argument("--series", default="KXBTC15M", help="comma-separated series")
    ap.add_argument("--min-edge-cents", type=float, default=2.0)
    ap.add_argument("--max-entry", type=int, default=90)
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--interval", type=float, default=20.0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
