import math


class MicrostructureFeatures:
    def __init__(self):
        pass

    def compute(self, orderbook: dict) -> dict[str, float]:
        nan = float("nan")
        nan_dict = {
            "mid_price": nan,
            "spread": nan,
            "spread_bps": nan,
            "microprice": nan,
            "microprice_deviation": nan,
            "best_bid_size": nan,
            "best_ask_size": nan,
            "top_imbalance": nan,
        }

        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            return nan_dict

        best_bid_price, best_bid_qty = bids[0][0], bids[0][1]
        best_ask_price, best_ask_qty = asks[0][0], asks[0][1]

        mid_price = (best_bid_price + best_ask_price) / 2.0
        spread = best_ask_price - best_bid_price

        if mid_price == 0:
            spread_bps = nan
        else:
            spread_bps = spread / mid_price * 10000.0

        total_qty = best_bid_qty + best_ask_qty
        if total_qty == 0:
            microprice = nan
            top_imbalance = nan
        else:
            microprice = (best_bid_price * best_ask_qty + best_ask_price * best_bid_qty) / total_qty
            top_imbalance = (best_bid_qty - best_ask_qty) / total_qty

        microprice_deviation = microprice - mid_price if not math.isnan(microprice) else nan

        return {
            "mid_price": mid_price,
            "spread": spread,
            "spread_bps": spread_bps,
            "microprice": microprice,
            "microprice_deviation": microprice_deviation,
            "best_bid_size": best_bid_qty,
            "best_ask_size": best_ask_qty,
            "top_imbalance": top_imbalance,
        }
