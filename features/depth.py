import numpy as np


class DepthFeatures:
    def __init__(self, levels=(5, 10, 20)):
        self.levels = levels

    def compute(self, orderbook: dict) -> dict[str, float]:
        nan = float("nan")
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        result: dict[str, float] = {}

        for L in self.levels:
            bid_sizes = [b[1] for b in bids[:L]]
            ask_sizes = [a[1] for a in asks[:L]]

            bid_depth = sum(bid_sizes) if bid_sizes else 0.0
            ask_depth = sum(ask_sizes) if ask_sizes else 0.0

            result[f"depth_bid_L{L}"] = bid_depth if bid_sizes else nan
            result[f"depth_ask_L{L}"] = ask_depth if ask_sizes else nan

            total = bid_depth + ask_depth
            if total > 0:
                result[f"book_imbalance_L{L}"] = (bid_depth - ask_depth) / total
            else:
                result[f"book_imbalance_L{L}"] = nan

        # Bid slope: linear regression of bid sizes across available levels (up to 20)
        max_levels = 20
        bid_sizes_all = [b[1] for b in bids[:max_levels]]
        ask_sizes_all = [a[1] for a in asks[:max_levels]]

        result["bid_slope"] = self._compute_slope(bid_sizes_all)
        result["ask_slope"] = self._compute_slope(ask_sizes_all)

        # Depth convexity: (size[0] + size[-1]) / 2 - mean(sizes) for bids
        if len(bid_sizes_all) >= 2:
            mean_size = np.mean(bid_sizes_all)
            result["depth_convexity"] = (bid_sizes_all[0] + bid_sizes_all[-1]) / 2.0 - mean_size
        else:
            result["depth_convexity"] = nan

        return result

    @staticmethod
    def _compute_slope(sizes: list[float]) -> float:
        if len(sizes) < 2:
            return float("nan")
        x = np.arange(len(sizes), dtype=np.float64)
        y = np.array(sizes, dtype=np.float64)
        # Linear regression slope via least squares
        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            return float("nan")
        slope = np.sum((x - x_mean) * (y - y_mean)) / denom
        return float(slope)
