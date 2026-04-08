#!/usr/bin/env python3
# TODO: Replace with real data source.
# Candidate sources (all free tier / public APIs):
#   - Funding rate: Binance /fapi/v1/fundingRate
#   - Open interest: Coinglass / Coinalyze / Binance /fapi/v1/openInterest
#   - Exchange netflow: CryptoQuant (paid), Glassnode (paid)
#   - Coinbase premium: (coinbase_spot - binance_spot) / binance_spot, sampled
# For now this script writes empty parquets so the feature join returns NaN
# and MACRO_FEATURES_ENABLED stays False until real data is wired.
"""Placeholder updater for macro crypto datasets.

Writes empty parquet files with the correct schemas under MACRO_DATA_ROOT so the
feature-join path is exercised end-to-end without any real data. A no-op if the
files already exist.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCHEMAS: dict[str, str] = {
    "funding_rate":    "funding_rate",
    "open_interest":   "open_interest",
    "exchange_netflow": "netflow",
    "coinbase_premium": "premium",
}


def write_empty(path: Path, metric_col: str) -> None:
    df = pd.DataFrame({
        "ts": pd.Series([], dtype="datetime64[ns, UTC]"),
        metric_col: pd.Series([], dtype="float64"),
    })
    df.to_parquet(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="local")
    parser.add_argument("--root", default="D:/kalshi-data/macro")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    existing = [p for p in (root / f"{s}.parquet" for s in SCHEMAS) if p.exists()]
    if len(existing) == len(SCHEMAS):
        print("macro data exists, no-op")
        return 0

    created = []
    for stem, metric_col in SCHEMAS.items():
        path = root / f"{stem}.parquet"
        if path.exists():
            continue
        write_empty(path, metric_col)
        created.append(str(path))

    print(f"Created {len(created)} empty placeholder parquet(s) under {root}:")
    for p in created:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
