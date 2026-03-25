import pandas as pd

from .microstructure import MicrostructureFeatures
from .depth import DepthFeatures
from .trade_flow import TradeFlowFeatures
from .technical import TechnicalFeatures
from .multi_timeframe import MultiTimeframeFeatures
from .derivatives import DerivativesFeatures

# Optional modules that may not exist yet
try:
    from .lob_dynamics import LOBDynamicsFeatures
except ImportError:
    LOBDynamicsFeatures = None  # type: ignore[misc, assignment]

try:
    from .calendar import CalendarFeatures
except ImportError:
    CalendarFeatures = None  # type: ignore[misc, assignment]


class FeaturePipeline:
    def __init__(self):
        self.microstructure = MicrostructureFeatures()
        self.depth = DepthFeatures()
        self.trade_flow = TradeFlowFeatures()
        self.technical = TechnicalFeatures()
        self.multi_timeframe = MultiTimeframeFeatures()
        self.derivatives = DerivativesFeatures()

        self.lob_dynamics = LOBDynamicsFeatures() if LOBDynamicsFeatures is not None else None
        self.calendar = CalendarFeatures() if CalendarFeatures is not None else None

    def update_orderbook(self, orderbook: dict) -> None:
        if self.lob_dynamics is not None:
            self.lob_dynamics.update(orderbook)

    def update_1m_bar(self, bar: dict) -> None:
        self.multi_timeframe.update(bar)

    def compute(
        self,
        orderbook: dict,
        bars_5s: pd.DataFrame,
        bars_1m: pd.DataFrame,
    ) -> dict[str, float]:
        features: dict[str, float] = {}

        features.update(self.microstructure.compute(orderbook))
        features.update(self.depth.compute(orderbook))
        features.update(self.trade_flow.compute(bars_5s))

        if self.lob_dynamics is not None:
            features.update(self.lob_dynamics.compute())

        features.update(self.technical.compute(bars_1m))
        features.update(self.multi_timeframe.compute())
        features.update(self.derivatives.compute())

        if self.calendar is not None:
            features.update(self.calendar.compute())

        return features

    def get_feature_names(self) -> list[str]:
        """Return sorted list of all feature keys produced by compute()."""
        # Build minimal dummy data to discover keys
        dummy_ob = {
            "bids": [[100.0, 1.0]] * 20,
            "asks": [[101.0, 1.0]] * 20,
        }
        dummy_5s = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=30, freq="5s"),
                "open": [100.0] * 30,
                "high": [101.0] * 30,
                "low": [99.0] * 30,
                "close": [100.0] * 30,
                "volume": [10.0] * 30,
                "buy_volume": [6.0] * 30,
                "sell_volume": [4.0] * 30,
                "trade_count": [5] * 30,
            }
        )
        dummy_1m = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=60, freq="1min"),
                "open": [100.0] * 60,
                "high": [101.0] * 60,
                "low": [99.0] * 60,
                "close": [100.0] * 60,
                "volume": [100.0] * 60,
            }
        )

        result = self.compute(dummy_ob, dummy_5s, dummy_1m)
        return sorted(result.keys())
