import math

import numpy as np
import pandas as pd


class Labeler:
    def __init__(self, horizon_minutes: int = 15, threshold_bps: float = 3.0):
        self.horizon_minutes = horizon_minutes
        self.threshold_bps = threshold_bps

    def label_direction(self, bars_1m: pd.DataFrame) -> pd.Series:
        """Label each bar with +1 / 0 / -1 based on forward log-return vs threshold."""
        close = bars_1m["close"].astype(float)
        horizon = self.horizon_minutes
        threshold = self.threshold_bps / 10000.0

        labels = pd.Series(np.nan, index=bars_1m.index, dtype=float)

        max_i = len(close) - horizon
        if max_i <= 0:
            return labels

        future_close = close.shift(-horizon)
        log_return = np.log(future_close / close)

        labels = pd.Series(0.0, index=bars_1m.index, dtype=float)
        labels[log_return > threshold] = 1.0
        labels[log_return < -threshold] = -1.0
        # Last `horizon` rows have no future data
        labels.iloc[-horizon:] = np.nan

        return labels

    def label_triple_barrier(
        self,
        bars_1m: pd.DataFrame,
        tp_bps: float = 10.0,
        sl_bps: float = 10.0,
    ) -> pd.Series:
        """Triple-barrier labeling: +1 if TP hit first, -1 if SL hit first, 0 if neither."""
        close = bars_1m["close"].values.astype(float)
        n = len(close)
        horizon = self.horizon_minutes
        tp_frac = tp_bps / 10000.0
        sl_frac = sl_bps / 10000.0

        labels = np.full(n, np.nan)

        for i in range(n - 1):
            end = min(i + horizon, n - 1)
            if i >= end:
                break

            entry = close[i]
            if entry == 0:
                continue

            tp_price = entry * (1.0 + tp_frac)
            sl_price = entry * (1.0 - sl_frac)

            label = 0.0
            for j in range(i + 1, end + 1):
                if close[j] >= tp_price:
                    label = 1.0
                    break
                if close[j] <= sl_price:
                    label = -1.0
                    break

            labels[i] = label

        return pd.Series(labels, index=bars_1m.index, dtype=float)
