"""Backtest the value-vs-fair model for 15-minute crypto markets.

Two things this answers:

  1. CALIBRATION (model quality, needs only 1m bars): for many (window, minute)
     samples, does the model's P(YES) match the realized outcome (close >= open)?
     Reported as Brier score vs the 0.25 coin-flip baseline + a reliability table.
     If the model isn't calibrated, no trading rule on top of it can be trusted.

  2. EDGE vs QUOTES (needs archived Kalshi quotes, optional): where archived
     yes_ask/no_ask snapshots exist, replay decide_value_trade and report the
     realized net PnL after fees. This is the actual strategy P&L.

Run on the box with data (Windows D:/kalshi-data):
    python -m scripts.backtest_value_strategy --bars
    python -m scripts.backtest_value_strategy --synthetic   # self-check, no data
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from engine.fair_value import prob_yes, sigma_per_min_from_returns
from engine.value_decision import decide_value_trade
from engine.fees import multiplier_for_family, kalshi_fee_cents, DEFAULT_MULTIPLIER

WINDOW_MIN = 15
# Decision minutes (minutes ELAPSED into the window) we evaluate. The edge lives
# late, but we sample 2..13 elapsed (T = 13..2 remaining) to match the live gate.
DECISION_ELAPSED = range(2, 14)


def _log_returns(close: np.ndarray) -> np.ndarray:
    out = np.full(close.shape, np.nan)
    out[1:] = np.log(close[1:] / close[:-1])
    return out


def run_calibration_backtest(bars: pd.DataFrame, vol_window: int = 15) -> dict:
    """Sample (predicted P(YES), realized YES) over aligned 15-min windows.

    ``bars`` needs columns: ``timestamp`` (datetime64 or ms int) and ``close``.
    Windows align to clock 15-min boundaries (:00/:15/:30/:45). Strike = price at
    window open; realized YES = close-at-window-end >= strike.
    """
    df = bars.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["minute"] = df["timestamp"].dt.floor("min")
    df["window"] = df["timestamp"].dt.floor(f"{WINDOW_MIN}min")
    close = df["close"].to_numpy(dtype=float)
    ret1m = _log_returns(close)
    df["ret_1m"] = ret1m

    preds: list[float] = []
    reals: list[int] = []

    for win, g in df.groupby("window"):
        if len(g) < WINDOW_MIN:  # need a full window to know the outcome
            continue
        g = g.sort_values("timestamp")
        idx = g.index.to_numpy()
        strike = float(close[idx[0]])
        realized_yes = int(close[idx[-1]] >= strike)
        for elapsed in DECISION_ELAPSED:
            if elapsed >= len(g):
                continue
            here = idx[elapsed]
            spot = float(close[here])
            t_remaining = WINDOW_MIN - elapsed
            # trailing per-minute vol up to this bar (no lookahead)
            sigma = sigma_per_min_from_returns(ret1m[: here + 1].tolist(), window=vol_window)
            if sigma is None:
                continue
            p = prob_yes(spot, strike, t_remaining, sigma)
            preds.append(p)
            reals.append(realized_yes)

    return _calibration_metrics(np.asarray(preds), np.asarray(reals, dtype=float))


def _calibration_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    if p.size == 0:
        return {"n": 0, "error": "no samples"}
    brier = float(np.mean((p - y) ** 2))
    baseline = float(np.mean((0.5 - y) ** 2))  # always-0.5
    base_rate = float(np.mean(y))
    # Directional accuracy: model says YES (p>0.5) and it resolved YES, etc.
    pred_yes = p > 0.5
    acc = float(np.mean(pred_yes == (y > 0.5)))
    # Reliability table over deciles.
    table = []
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        table.append({
            "bucket": f"{lo:.1f}-{hi:.1f}",
            "n": int(mask.sum()),
            "mean_pred": float(p[mask].mean()),
            "mean_real": float(y[mask].mean()),
        })
    return {
        "n": int(p.size),
        "brier": brier,
        "baseline_brier_0.5": baseline,
        "beats_baseline": brier < baseline,
        "base_rate_yes": base_rate,
        "directional_accuracy": acc,
        "reliability": table,
    }


def simulate_strategy_pnl(
    bars: pd.DataFrame,
    *,
    lag_min: int = 2,
    half_spread_cents: int = 2,
    min_edge_cents: float = 2.0,
    max_entry_cents: int = 90,
    vol_window: int = 15,
    fee_multiplier: float = DEFAULT_MULTIPLIER,
) -> dict:
    """Realized PnL of the value strategy against a market that LAGS fair value.

    Models the edge observed live: the book prices off a slightly stale spot, so
    when price moves the quote lags the now-near-decided outcome. We price fair
    value off the CURRENT spot and the market off ``lag_min`` minutes ago, add a
    half-spread, then run the real decision + fee logic and settle on the actual
    window outcome. With ``lag_min=0`` and a wide spread, no edge should exist —
    a useful sanity floor.
    """
    df = bars.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["window"] = df["timestamp"].dt.floor(f"{WINDOW_MIN}min")
    close = df["close"].to_numpy(dtype=float)
    ret1m = _log_returns(close)

    def _clamp(c: int) -> int:
        return max(1, min(99, int(round(c))))

    pnls: list[float] = []
    wins = 0
    for win, g in df.groupby("window"):
        if len(g) < WINDOW_MIN:
            continue
        idx = g.index.to_numpy()
        strike = float(close[idx[0]])
        realized_yes = close[idx[-1]] >= strike
        for elapsed in DECISION_ELAPSED:
            if elapsed >= len(g):
                continue
            here = idx[elapsed]
            lag_pos = max(idx[0], here - lag_min)
            spot_now = float(close[here])
            spot_lag = float(close[lag_pos])
            t_remaining = WINDOW_MIN - elapsed
            sigma = sigma_per_min_from_returns(ret1m[: here + 1].tolist(), window=vol_window)
            if sigma is None:
                continue
            fair_p = prob_yes(spot_now, strike, t_remaining, sigma)
            market_p = prob_yes(spot_lag, strike, t_remaining, sigma)
            yes_ask = _clamp(market_p * 100 + half_spread_cents)
            no_ask = _clamp((1 - market_p) * 100 + half_spread_cents)
            d = decide_value_trade(
                fair_p, yes_ask, no_ask,
                min_edge_cents=min_edge_cents, multiplier=fee_multiplier,
                max_entry_cents=max_entry_cents,
            )
            if d.side is None:
                continue
            won = (d.side == "yes" and realized_yes) or (d.side == "no" and not realized_yes)
            payoff = 100 if won else 0
            fee = kalshi_fee_cents(d.entry_cents, 1, maker=False, multiplier=fee_multiplier)
            pnls.append(payoff - d.entry_cents - fee)
            wins += int(won)

    arr = np.asarray(pnls)
    if arr.size == 0:
        return {"trades": 0, "note": "no qualifying trades"}
    return {
        "trades": int(arr.size),
        "win_rate": wins / arr.size,
        "total_pnl_cents": float(arr.sum()),
        "avg_pnl_cents": float(arr.mean()),
        "lag_min": lag_min,
        "half_spread_cents": half_spread_cents,
        "min_edge_cents": min_edge_cents,
    }


def synthetic_bars(n_minutes: int = 6000, sigma_per_min: float = 0.0009, seed: int = 7) -> pd.DataFrame:
    """GBM price path for a self-check. Model assumes this process, so it should
    come out well-calibrated (Brier << 0.25)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, sigma_per_min, size=n_minutes)
    price = 60000 * np.exp(np.cumsum(rets))
    ts = pd.date_range("2026-01-01", periods=n_minutes, freq="min", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "close": price})


def _print_report(metrics: dict) -> None:
    if metrics.get("n", 0) == 0:
        print("No samples:", metrics.get("error"))
        return
    print(f"samples            : {metrics['n']}")
    print(f"base rate P(YES)   : {metrics['base_rate_yes']:.3f}")
    print(f"Brier (model)      : {metrics['brier']:.4f}")
    print(f"Brier (always 0.5) : {metrics['baseline_brier_0.5']:.4f}")
    print(f"beats coin flip    : {metrics['beats_baseline']}")
    print(f"directional acc    : {metrics['directional_accuracy']:.3f}")
    print("reliability (pred -> real):")
    for row in metrics["reliability"]:
        print(f"  {row['bucket']}  n={row['n']:6d}  pred={row['mean_pred']:.3f}  real={row['mean_real']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest value-vs-fair crypto model")
    ap.add_argument("--bars", action="store_true", help="use real 1m bars from DATA_DIR/bars_1m")
    ap.add_argument("--synthetic", action="store_true", help="self-check on a synthetic GBM path")
    ap.add_argument("--vol-window", type=int, default=15)
    ap.add_argument("--min-edge-cents", type=float, default=2.0)
    args = ap.parse_args()

    if args.synthetic or not args.bars:
        print("=== SYNTHETIC self-check (GBM) ===")
        sb = synthetic_bars()
        _print_report(run_calibration_backtest(sb, vol_window=args.vol_window))
        print("\n--- strategy PnL vs a lagging market (synthetic) ---")
        for lag in (0, 2, 4):
            r = simulate_strategy_pnl(sb, lag_min=lag, half_spread_cents=2, min_edge_cents=args.min_edge_cents)
            if r.get("trades", 0):
                print(f"  lag={lag}m: {r['trades']:5d} trades  win={r['win_rate']:.3f}  "
                      f"avg={r['avg_pnl_cents']:+.2f}c/trade  total={r['total_pnl_cents']:+.0f}c")
            else:
                print(f"  lag={lag}m: no qualifying trades")
        if not args.bars:
            return

    from config.settings import settings
    bars_dir = Path(settings.DATA_DIR) / "bars_1m"
    files = sorted(bars_dir.glob("*.parquet")) if bars_dir.exists() else []
    if not files:
        print(f"No 1m bars found under {bars_dir} — run on the data box (Windows).")
        return
    frames = [pd.read_parquet(f) for f in files]
    bars = pd.concat(frames, ignore_index=True)
    print(f"\n=== REAL bars: {len(files)} files, {len(bars)} rows ===")
    _print_report(run_calibration_backtest(bars, vol_window=args.vol_window))
    print("\n--- strategy PnL vs a lagging market (real bars) ---")
    for lag in (0, 2, 4):
        r = simulate_strategy_pnl(bars, lag_min=lag, half_spread_cents=2, min_edge_cents=args.min_edge_cents)
        if r.get("trades", 0):
            print(f"  lag={lag}m: {r['trades']:6d} trades  win={r['win_rate']:.3f}  "
                  f"avg={r['avg_pnl_cents']:+.2f}c/trade  total={r['total_pnl_cents']:+.0f}c")
        else:
            print(f"  lag={lag}m: no qualifying trades")


if __name__ == "__main__":
    main()
