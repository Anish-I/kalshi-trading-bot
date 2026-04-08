"""Optional macro crypto features. Default OFF (see config.settings.MACRO_FEATURES_ENABLED)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MACRO_FEATURE_NAMES: list[str] = [
    "macro_funding_rate",       # perp funding (btc, 8h)
    "macro_open_interest",      # btc futures open interest
    "macro_exchange_netflow",   # btc exchange netflow (inflow - outflow)
    "macro_coinbase_premium",   # (coinbase_price - binance_price) / binance_price
]


def _empty(metric_col: str) -> pd.DataFrame:
    return pd.DataFrame({
        "ts": pd.Series([], dtype="datetime64[ns, UTC]"),
        metric_col: pd.Series([], dtype="float64"),
    })


def load_funding_rates(path: Path) -> pd.DataFrame:
    """Columns: ts (UTC), funding_rate (float)."""
    if not Path(path).exists():
        return _empty("funding_rate")
    return pd.read_parquet(path)


def load_open_interest(path: Path) -> pd.DataFrame:
    """Columns: ts (UTC), open_interest (float)."""
    if not Path(path).exists():
        return _empty("open_interest")
    return pd.read_parquet(path)


def load_exchange_netflow(path: Path) -> pd.DataFrame:
    """Columns: ts (UTC), netflow (float)."""
    if not Path(path).exists():
        return _empty("netflow")
    return pd.read_parquet(path)


def load_coinbase_premium(path: Path) -> pd.DataFrame:
    """Columns: ts (UTC), premium (float)."""
    if not Path(path).exists():
        return _empty("premium")
    return pd.read_parquet(path)


def join_macro_asof(
    bars_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    bars_ts_col: str = "ts",
    macro_ts_col: str = "ts",
    tolerance: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Strict BACKWARD as-of merge — decision time must not see future macro rows.

    CRITICAL: direction='backward' only. No 'nearest' allowed.
    """
    if bars_df.empty or macro_df.empty:
        return bars_df.copy()
    b = bars_df.sort_values(bars_ts_col).reset_index(drop=True)
    m = macro_df.sort_values(macro_ts_col).reset_index(drop=True)
    merged = pd.merge_asof(
        b, m,
        left_on=bars_ts_col, right_on=macro_ts_col,
        direction="backward",
        tolerance=tolerance,
    )
    return merged


def attach_macro_features(
    bars_df: pd.DataFrame,
    macro_root: Path,
    ts_col: str = "ts",
) -> pd.DataFrame:
    """Load all 4 macro metrics from `macro_root/{metric}.parquet` and join to bars
    via backward as-of. Missing metrics are filled with NaN columns. Returns bars_df
    with MACRO_FEATURE_NAMES columns added."""
    out = bars_df.copy()
    macro_root = Path(macro_root)
    loaders = {
        "macro_funding_rate":     (load_funding_rates,   "funding_rate"),
        "macro_open_interest":    (load_open_interest,   "open_interest"),
        "macro_exchange_netflow": (load_exchange_netflow, "netflow"),
        "macro_coinbase_premium": (load_coinbase_premium, "premium"),
    }
    # If bars has no ts column, just fill NaN columns.
    if ts_col not in out.columns:
        for col_name in MACRO_FEATURE_NAMES:
            out[col_name] = float("nan")
        return out

    for col_name, (loader, src_col) in loaders.items():
        file_stem = col_name.replace("macro_", "")
        path = macro_root / f"{file_stem}.parquet"
        m = loader(path)
        if m is None or m.empty:
            out[col_name] = float("nan")
            continue
        if src_col in m.columns:
            m = m[["ts", src_col]].rename(columns={src_col: col_name})
        elif col_name not in m.columns:
            out[col_name] = float("nan")
            continue
        joined = join_macro_asof(
            out[[ts_col]].copy(), m,
            bars_ts_col=ts_col, macro_ts_col="ts",
        )
        out[col_name] = joined[col_name].values
    return out
