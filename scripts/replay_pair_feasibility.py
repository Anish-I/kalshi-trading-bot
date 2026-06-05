"""
Replay MAKER pair feasibility on archived orderbook data.

IMPORTANT: taker "buy both sides" is structurally impossible — implied asks
always sum to >= 100c, so the only viable pair is MAKER (post limits at
best_bid + 1). Maker economics derive directly from the book spread:

    maker_gross = spread - 2     (spread = yes_ask + no_ask - 100)
    maker_net   = maker_gross - real_maker_fee   (engine.fees)

So you need spread >= ~2 + fee + min_net (~>=5c) to clear. This script reports
the spread distribution (the orphan-risk proxy: tight spread => high orphan
rate) and the maker-net opportunity rate per series.

CAVEAT: these are POTENTIAL fills assuming BOTH legs fill. It does NOT model
orphan risk, which is the real-world killer on tight books. Treat a positive
maker-net rate as necessary-but-not-sufficient until the orphan-unwind state
machine is shipped and paper orphan rates are measured.

Run (on the data box): python -m scripts.replay_pair_feasibility --series KXDOGE15M
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

from config.settings import settings
from engine.pair_pricing import maker_economics_from_asks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pair_replay")

ARCHIVE_DIR = Path(settings.DATA_DIR) / "market_archive"
MIN_NETS = [1.0, 2.0, 3.0]  # min maker-net thresholds to report


def load_all_snapshots(series: str) -> pd.DataFrame:
    """Load all archived snapshots for a series."""
    frames = []
    if not ARCHIVE_DIR.exists():
        return pd.DataFrame()
    for day_dir in sorted(ARCHIVE_DIR.iterdir()):
        if not day_dir.is_dir():
            continue
        path = day_dir / f"{series}.parquet"
        if path.exists():
            try:
                frames.append(pd.read_parquet(path))
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

    # Maker economics per snapshot (derived from implied asks).
    econ = quoted.apply(
        lambda r: maker_economics_from_asks(int(r["ya_cents"]), int(r["na_cents"])),
        axis=1, result_type="expand",
    )
    quoted = pd.concat([quoted.reset_index(drop=True), econ.reset_index(drop=True)], axis=1)

    print("\n" + "=" * 64)
    print("MAKER PAIR FEASIBILITY REPLAY  (taker arb is impossible — maker only)")
    print("=" * 64)
    print(f"Snapshots with quotes: {len(quoted)}")
    print(f"Unique tickers: {quoted['ticker'].nunique()}")

    # Spread distribution = the orphan-risk proxy (tight => high orphan rate).
    print(f"\n--- SPREAD DISTRIBUTION (orphan-risk proxy) ---")
    print(f"Mean: {quoted['spread_cents'].mean():.1f}c  Median: {quoted['spread_cents'].median():.0f}c  "
          f"Min: {quoted['spread_cents'].min()}c  Max: {quoted['spread_cents'].max()}c")
    for thr in (1, 2, 3, 5, 7):
        pct = (quoted['spread_cents'] >= thr).mean() * 100
        print(f"  spread >= {thr}c: {pct:5.1f}% of snapshots")

    # Maker-net opportunity rate (BOTH-legs-fill assumption — orphan risk NOT modeled).
    print(f"\n--- MAKER-NET OPPORTUNITY RATE (assumes both legs fill) ---")
    for mn in MIN_NETS:
        opps = quoted[quoted["maker_net"] >= mn]
        pct = len(opps) / len(quoted) * 100
        avg = opps["maker_net"].mean() if len(opps) else 0
        print(f"  maker_net >= {mn:.0f}c: {len(opps):>6}/{len(quoted)} ({pct:5.1f}%)  avg={avg:+.1f}c/pair")

    print(f"\n--- PER-MARKET (maker_net >= 2c) ---")
    for ticker, g in quoted.groupby("ticker"):
        if len(g) < 3:
            continue
        opps = g[g["maker_net"] >= 2.0]
        print(f"  {ticker}: {len(g)} snaps, {len(opps)} opps ({len(opps)/len(g)*100:.0f}%), "
              f"avg spread={g['spread_cents'].mean():.1f}c")

    print(f"\n--- VERDICT ---")
    rate = (quoted["maker_net"] >= 2.0).mean()
    print(f"maker_net>=2c rate: {rate*100:.1f}%  |  median spread: {quoted['spread_cents'].median():.0f}c")
    print("REMINDER: positive here is necessary but NOT sufficient — orphan risk is")
    print("not modeled. Ship the orphan-unwind state machine and measure paper orphan")
    print("rate before enabling live (see docs/value_and_pair_findings.md).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Maker pair feasibility replay")
    ap.add_argument("--series", default="KXBTC15M", help="series ticker to analyze")
    args = ap.parse_args()
    df = load_all_snapshots(args.series)
    if df.empty:
        log.error("No archive data for %s under %s", args.series, ARCHIVE_DIR)
    else:
        log.info("Loaded %d snapshots for %s", len(df), args.series)
        analyze(df)
