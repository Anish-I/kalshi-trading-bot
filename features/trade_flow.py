import math

import numpy as np
import pandas as pd


class TradeFlowFeatures:
    def __init__(self, windows=("10s", "30s", "2min")):
        self.windows = windows

    def compute(self, bars_5s: pd.DataFrame) -> dict[str, float]:
        nan = float("nan")
        result: dict[str, float] = {}

        if bars_5s is None or bars_5s.empty:
            for w in self.windows:
                result[f"taker_imbalance_{w}"] = nan
                result[f"cvd_{w}"] = nan
            result["cvd_slope_30s"] = nan
            result["cvd_slope_2m"] = nan
            result["trade_intensity_10s"] = nan
            result["avg_trade_size_30s"] = nan
            result["large_trade_ratio_2m"] = nan
            result["buy_sell_ratio_1m"] = nan
            return result

        ts = bars_5s["timestamp"]
        latest = ts.iloc[-1]

        for w in self.windows:
            td = pd.Timedelta(w)
            mask = ts >= (latest - td)
            window_df = bars_5s.loc[mask]

            if window_df.empty:
                result[f"taker_imbalance_{w}"] = nan
                result[f"cvd_{w}"] = nan
                continue

            buy_vol = window_df["buy_volume"].sum()
            sell_vol = window_df["sell_volume"].sum()
            total_vol = window_df["volume"].sum()
            delta = buy_vol - sell_vol

            result[f"taker_imbalance_{w}"] = (delta / total_vol) if total_vol > 0 else nan
            result[f"cvd_{w}"] = float(delta)

        # CVD slope over last 30s
        result["cvd_slope_30s"] = self._cvd_slope(bars_5s, ts, latest, pd.Timedelta("30s"))

        # CVD slope over last 2min
        result["cvd_slope_2m"] = self._cvd_slope(bars_5s, ts, latest, pd.Timedelta("2min"))

        # Trade intensity: trade_count over last 10s / 10
        mask_10s = ts >= (latest - pd.Timedelta("10s"))
        df_10s = bars_5s.loc[mask_10s]
        if df_10s.empty:
            result["trade_intensity_10s"] = nan
        else:
            result["trade_intensity_10s"] = float(df_10s["trade_count"].sum()) / 10.0

        # Avg trade size over 30s
        mask_30s = ts >= (latest - pd.Timedelta("30s"))
        df_30s = bars_5s.loc[mask_30s]
        if df_30s.empty:
            result["avg_trade_size_30s"] = nan
        else:
            total_vol_30 = df_30s["volume"].sum()
            total_tc_30 = df_30s["trade_count"].sum()
            result["avg_trade_size_30s"] = (float(total_vol_30) / float(total_tc_30)) if total_tc_30 > 0 else nan

        # Large trade ratio over 2min
        mask_2m = ts >= (latest - pd.Timedelta("2min"))
        df_2m = bars_5s.loc[mask_2m]
        if df_2m.empty or len(df_2m) < 2:
            result["large_trade_ratio_2m"] = nan
        else:
            median_vol = df_2m["volume"].median()
            if median_vol <= 0:
                result["large_trade_ratio_2m"] = nan
            else:
                large_mask = df_2m["volume"] > 2.0 * median_vol
                total_vol_2m = df_2m["volume"].sum()
                large_vol = df_2m.loc[large_mask, "volume"].sum()
                result["large_trade_ratio_2m"] = (float(large_vol) / float(total_vol_2m)) if total_vol_2m > 0 else nan

        # Buy/sell ratio over 1min
        mask_1m = ts >= (latest - pd.Timedelta("1min"))
        df_1m = bars_5s.loc[mask_1m]
        if df_1m.empty:
            result["buy_sell_ratio_1m"] = nan
        else:
            buy_1m = df_1m["buy_volume"].sum()
            sell_1m = df_1m["sell_volume"].sum()
            result["buy_sell_ratio_1m"] = (float(buy_1m) / float(sell_1m)) if sell_1m > 0 else nan

        return result

    @staticmethod
    def _cvd_slope(bars_5s: pd.DataFrame, ts: pd.Series, latest, td: pd.Timedelta) -> float:
        mask = ts >= (latest - td)
        window_df = bars_5s.loc[mask]

        if len(window_df) < 2:
            return float("nan")

        cvd = (window_df["buy_volume"] - window_df["sell_volume"]).cumsum().values.astype(np.float64)
        x = np.arange(len(cvd), dtype=np.float64)
        x_mean = x.mean()
        y_mean = cvd.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            return float("nan")
        slope = np.sum((x - x_mean) * (cvd - y_mean)) / denom
        return float(slope)
