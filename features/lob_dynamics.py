import math
from collections import deque

import numpy as np


class LOBDynamicsFeatures:
    def __init__(self):
        self._prev_orderbook: dict | None = None
        self._prev_spread: float | None = None
        self._spread_history: deque[float] = deque(maxlen=60)
        self._depth_history: deque[float] = deque(maxlen=60)
        self._best_bid_history: deque[float] = deque(maxlen=60)
        self._best_ask_history: deque[float] = deque(maxlen=60)

    def update(self, orderbook: dict):
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if bids and asks:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = best_ask - best_bid
            total_depth = sum(b[1] for b in bids) + sum(a[1] for a in asks)
        else:
            best_bid = float("nan")
            best_ask = float("nan")
            spread = float("nan")
            total_depth = float("nan")

        self._best_bid_history.append(best_bid)
        self._best_ask_history.append(best_ask)
        self._spread_history.append(spread)
        self._depth_history.append(total_depth)

        self._prev_spread = spread
        self._prev_orderbook = orderbook

    def compute(self) -> dict[str, float]:
        nan = float("nan")

        if self._prev_orderbook is None:
            return {
                "spread_change_rate_1m": nan,
                "depth_replenishment_30s": nan,
                "bid_retreat": nan,
                "ask_retreat": nan,
                "cancel_rate_proxy": nan,
                "quote_churn_30s": nan,
                "depth_volatility_30s": nan,
                "spread_volatility_30s": nan,
            }

        result: dict[str, float] = {}

        # spread_change_rate_1m: compare current spread to 60 samples ago
        spreads = list(self._spread_history)
        if len(spreads) >= 60 and not math.isnan(spreads[0]) and spreads[0] != 0:
            result["spread_change_rate_1m"] = (spreads[-1] - spreads[0]) / spreads[0]
        else:
            result["spread_change_rate_1m"] = nan

        # depth_replenishment_30s: current depth / depth 30 samples ago
        depths = list(self._depth_history)
        if len(depths) >= 30:
            depth_30_ago = depths[-30]
            if not math.isnan(depth_30_ago) and depth_30_ago > 0:
                result["depth_replenishment_30s"] = depths[-1] / depth_30_ago
            else:
                result["depth_replenishment_30s"] = nan
        else:
            result["depth_replenishment_30s"] = nan

        # bid_retreat: count of times best bid decreased in last 30 updates
        bids = list(self._best_bid_history)
        last_30_bids = bids[-30:] if len(bids) >= 30 else bids
        bid_retreat = 0
        for i in range(1, len(last_30_bids)):
            if not math.isnan(last_30_bids[i]) and not math.isnan(last_30_bids[i - 1]):
                if last_30_bids[i] < last_30_bids[i - 1]:
                    bid_retreat += 1
        result["bid_retreat"] = float(bid_retreat)

        # ask_retreat: count of times best ask increased in last 30 updates
        asks = list(self._best_ask_history)
        last_30_asks = asks[-30:] if len(asks) >= 30 else asks
        ask_retreat = 0
        for i in range(1, len(last_30_asks)):
            if not math.isnan(last_30_asks[i]) and not math.isnan(last_30_asks[i - 1]):
                if last_30_asks[i] > last_30_asks[i - 1]:
                    ask_retreat += 1
        result["ask_retreat"] = float(ask_retreat)

        # cancel_rate_proxy: absolute change in total depth not explained by spread changes
        if len(depths) >= 2 and not math.isnan(depths[-1]) and not math.isnan(depths[-2]):
            depth_change = abs(depths[-1] - depths[-2])
            spread_change = abs(spreads[-1] - spreads[-2]) if len(spreads) >= 2 and not math.isnan(spreads[-1]) and not math.isnan(spreads[-2]) else 0.0
            result["cancel_rate_proxy"] = depth_change - spread_change
        else:
            result["cancel_rate_proxy"] = nan

        # quote_churn_30s: count of times best bid or ask changed price in last 30 updates
        churn = 0
        for i in range(1, len(last_30_bids)):
            if not math.isnan(last_30_bids[i]) and not math.isnan(last_30_bids[i - 1]):
                if last_30_bids[i] != last_30_bids[i - 1]:
                    churn += 1
        for i in range(1, len(last_30_asks)):
            if not math.isnan(last_30_asks[i]) and not math.isnan(last_30_asks[i - 1]):
                if last_30_asks[i] != last_30_asks[i - 1]:
                    churn += 1
        result["quote_churn_30s"] = float(churn)

        # depth_volatility_30s: std dev of total depth over last 30 samples
        last_30_depths = depths[-30:] if len(depths) >= 30 else depths
        valid_depths = [d for d in last_30_depths if not math.isnan(d)]
        if len(valid_depths) >= 2:
            result["depth_volatility_30s"] = float(np.std(valid_depths, ddof=1))
        else:
            result["depth_volatility_30s"] = nan

        # spread_volatility_30s: std dev of spread over last 30 samples
        last_30_spreads = spreads[-30:] if len(spreads) >= 30 else spreads
        valid_spreads = [s for s in last_30_spreads if not math.isnan(s)]
        if len(valid_spreads) >= 2:
            result["spread_volatility_30s"] = float(np.std(valid_spreads, ddof=1))
        else:
            result["spread_volatility_30s"] = nan

        return result
