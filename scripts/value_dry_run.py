"""Live-data dry run of the value strategy — no Kalshi auth needed.

Mirrors crypto_value_trader's scan loop read-only, for boxes without API creds:
  - spot + fresh 1m candles from the Coinbase public API (-> calibrated EWMA sigma)
  - open KXBTC15M markets from Kalshi's public market-data endpoint
  - engine.value_strategy.evaluate_market with the production defaults
Logs fairP / net EV / adaptive threshold / would-trade decision every scan.
Places no orders and touches no state.

    python -m scripts.value_dry_run --scans 30 --interval 20
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import httpx
import numpy as np

from engine.fair_value import calibrated_sigma_per_min
from engine.value_strategy import evaluate_market, EDGE_SLOPE_CENTS_PER_MIN

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
CB = "https://api.exchange.coinbase.com"


def get_sigma(client: httpx.Client) -> float | None:
    r = client.get(f"{CB}/products/BTC-USD/candles", params={"granularity": 60}, timeout=10)
    r.raise_for_status()
    rows = sorted(r.json(), key=lambda x: x[0])  # [time, low, high, open, close, vol]
    closes = np.array([row[4] for row in rows], dtype=float)
    ret = np.full(closes.shape, np.nan)
    ret[1:] = np.log(closes[1:] / closes[:-1])
    return calibrated_sigma_per_min(ret)


def get_spot(client: httpx.Client) -> float:
    r = client.get(f"{CB}/products/BTC-USD/ticker", timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])


def get_markets(client: httpx.Client) -> list[dict]:
    r = client.get(f"{KALSHI}/markets",
                   params={"series_ticker": "KXBTC15M", "status": "open", "limit": 10},
                   timeout=10)
    r.raise_for_status()
    return r.json().get("markets", [])


def main() -> None:
    ap = argparse.ArgumentParser(description="No-auth live dry run of the value strategy")
    ap.add_argument("--scans", type=int, default=30)
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--min-edge-cents", type=float, default=2.0)
    args = ap.parse_args()

    signals = 0
    with httpx.Client() as client:
        for _scan in range(args.scans):
            now = datetime.now(timezone.utc)
            try:
                sigma = get_sigma(client)
                spot = get_spot(client)
                markets = get_markets(client)
            except Exception as e:
                print(f"[{now:%H:%M:%S}] fetch error: {e}", flush=True)
                time.sleep(args.interval)
                continue
            if sigma is None:
                print(f"[{now:%H:%M:%S}] no sigma", flush=True)
                time.sleep(args.interval)
                continue
            for m in markets:
                ev = evaluate_market(m, spot, sigma, now=now, min_edge_cents=args.min_edge_cents)
                t_min = ev.remaining_s / 60.0
                req = args.min_edge_cents + EDGE_SLOPE_CENTS_PER_MIN * t_min
                d = ev.decision
                yes_a = m.get("yes_ask_dollars", m.get("yes_ask"))
                no_a = m.get("no_ask_dollars", m.get("no_ask"))
                if d is None:
                    print(f"[{now:%H:%M:%S}] {m['ticker'][-12:]} T={t_min:5.1f}m  skip: {ev.reason}", flush=True)
                elif d.side is None:
                    print(f"[{now:%H:%M:%S}] {m['ticker'][-12:]} T={t_min:5.1f}m spot={spot:.0f} "
                          f"strike={ev.strike:.0f} fairP={ev.fair_p:.3f} ya={yes_a} na={no_a} "
                          f"ev={d.net_ev_cents:+.1f}c req={req:.1f}c  PASS", flush=True)
                else:
                    signals += 1
                    print(f"[{now:%H:%M:%S}] {m['ticker'][-12:]} T={t_min:5.1f}m spot={spot:.0f} "
                          f"strike={ev.strike:.0f} fairP={ev.fair_p:.3f} ya={yes_a} na={no_a} "
                          f">>> WOULD BUY {d.side.upper()} @{d.entry_cents}c "
                          f"ev={d.net_ev_cents:+.1f}c req={req:.1f}c", flush=True)
            time.sleep(args.interval)
    print(f"\ndone: {args.scans} scans, {signals} would-trade signals", flush=True)


if __name__ == "__main__":
    main()
