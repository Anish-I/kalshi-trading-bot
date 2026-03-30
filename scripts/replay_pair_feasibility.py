"""
Replay pair feasibility on archived orderbook data.

Answers: how often could both legs have filled below pair_cap?
Uses archived KXBTC15M snapshots from D:/kalshi-data/market_archive/

Run: python scripts/replay_pair_feasibility.py
"""
import logging
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

from engine.pair_pricing import PAIR_FEE_CENTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pair_replay")

ARCHIVE_DIR = Path("D:/kalshi-data/market_archive")
PAIR_CAPS = [93, 94, 95, 96, 97, 98]  # test multiple caps


def load_all_snapshots() -> pd.DataFrame:
    """Load all archived KXBTC15M snapshots."""
    frames = []
    for day_dir in sorted(ARCHIVE_DIR.iterdir()):
        if not day_dir.is_dir():
            continue
        path = day_dir / "KXBTC15M.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                frames.append(df)
            except Exception:
                log.warning("Failed to read %s", path)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def analyze(df: pd.DataFrame) -> None:
    """Run pair feasibility analysis on archived data."""
    # Need yes_ask and no_ask columns (from fixed archiver)
    has_asks = "yes_ask" in df.columns and "no_ask" in df.columns
    has_depth = "yes_ask_0_price" in df.columns

    if has_asks:
        # Use top-of-book from archiver
        df = df.copy()
        df["ya_cents"] = (df["yes_ask"] * 100).round().astype(int)
        df["na_cents"] = (df["no_ask"] * 100).round().astype(int)
    elif has_depth:
        df = df.copy()
        df["ya_cents"] = df["yes_ask_0_price"].fillna(0).astype(int)
        df["na_cents"] = df["no_ask_0_price"].fillna(0).astype(int)
    else:
        log.error("No orderbook data in archive. Need yes_ask or yes_ask_0_price columns.")
        return

    # Filter to rows with actual quotes
    quoted = df[(df["ya_cents"] > 0) & (df["na_cents"] > 0)].copy()
    log.info("Total snapshots: %d, with quotes: %d (%.0f%%)",
             len(df), len(quoted), len(quoted) / len(df) * 100 if len(df) else 0)

    if quoted.empty:
        log.warning("No snapshots with both YES and NO quotes. Need more archive data.")
        log.info("Archive has %d rows but quotes are all zero (archiver was broken, now fixed).", len(df))
        log.info("Wait 3-5 days for new data to accumulate, then re-run.")
        return

    quoted["pair_cost"] = quoted["ya_cents"] + quoted["na_cents"]
    quoted["gross_profit"] = 100 - quoted["pair_cost"]
    quoted["net_profit"] = quoted["gross_profit"] - PAIR_FEE_CENTS

    print("\n" + "=" * 60)
    print("PAIR FEASIBILITY REPLAY")
    print("=" * 60)
    print(f"Snapshots with quotes: {len(quoted)}")
    print(f"Unique tickers: {quoted['ticker'].nunique()}")
    print(f"Date range: {quoted['timestamp'].min()[:10]} to {quoted['timestamp'].max()[:10]}")

    # Pair cost distribution
    print(f"\n--- PAIR COST DISTRIBUTION ---")
    print(f"Mean: {quoted['pair_cost'].mean():.1f}c")
    print(f"Median: {quoted['pair_cost'].median():.0f}c")
    print(f"Min: {quoted['pair_cost'].min()}c")
    print(f"Max: {quoted['pair_cost'].max()}c")
    print(f"Std: {quoted['pair_cost'].std():.1f}c")

    # Opportunity frequency at various caps
    print(f"\n--- OPPORTUNITY FREQUENCY BY PAIR CAP ---")
    for cap in PAIR_CAPS:
        opps = quoted[quoted["pair_cost"] <= cap]
        pct = len(opps) / len(quoted) * 100
        avg_net = opps["net_profit"].mean() if len(opps) else 0
        print(f"  Cap {cap}c: {len(opps):>5}/{len(quoted)} snapshots ({pct:>5.1f}%) "
              f"avg net={avg_net:+.1f}c/pair")

    # Per-ticker analysis
    print(f"\n--- PER-MARKET ANALYSIS (cap=96c) ---")
    for ticker, group in quoted.groupby("ticker"):
        n = len(group)
        opps = group[group["pair_cost"] <= 96]
        if n < 3:
            continue
        pct = len(opps) / n * 100
        avg_cost = group["pair_cost"].mean()
        print(f"  {ticker}: {n} snaps, {len(opps)} opps ({pct:.0f}%), avg cost={avg_cost:.0f}c")

    # Time analysis (if we have enough data)
    if "timestamp" in quoted.columns and len(quoted) > 10:
        quoted["hour"] = pd.to_datetime(quoted["timestamp"]).dt.hour
        print(f"\n--- OPPORTUNITY BY HOUR (UTC, cap=96c) ---")
        for hour in sorted(quoted["hour"].unique()):
            h_data = quoted[quoted["hour"] == hour]
            h_opps = h_data[h_data["pair_cost"] <= 96]
            pct = len(h_opps) / len(h_data) * 100 if len(h_data) else 0
            avg_net = h_opps["net_profit"].mean() if len(h_opps) else 0
            print(f"  {hour:02d}:00 UTC: {len(h_data):>4} snaps, {len(h_opps):>4} opps ({pct:>5.1f}%) avg net={avg_net:+.1f}c")

    # Summary
    best_cap_opps = quoted[quoted["pair_cost"] <= 96]
    print(f"\n--- SUMMARY ---")
    if len(best_cap_opps) > 0:
        total_potential = best_cap_opps["net_profit"].sum()
        print(f"Total potential profit at 96c cap: {total_potential:+.0f}c (${total_potential/100:+.2f})")
        print(f"Avg net per opportunity: {best_cap_opps['net_profit'].mean():+.1f}c")
        print(f"Opportunity rate: {len(best_cap_opps)/len(quoted)*100:.1f}%")
        print(f"\nVERDICT: {'GO' if best_cap_opps['net_profit'].mean() > 1.0 and len(best_cap_opps)/len(quoted) > 0.05 else 'NEED MORE DATA'}")
    else:
        print("No opportunities found at 96c cap.")
        print("VERDICT: NEED MORE DATA (archiver was broken, fixed now)")


if __name__ == "__main__":
    df = load_all_snapshots()
    if df.empty:
        log.error("No archive data found at %s", ARCHIVE_DIR)
    else:
        log.info("Loaded %d total snapshots", len(df))
        analyze(df)
