import math


class DerivativesFeatures:
    def __init__(self):
        self.funding_rate: float = float("nan")
        self.prev_funding_rate: float = float("nan")
        self.basis_proxy: float = float("nan")
        self.oi_change: float = float("nan")
        self.last_oi: float = float("nan")

    def update_funding(self, funding_rate: float) -> None:
        self.prev_funding_rate = self.funding_rate
        self.funding_rate = funding_rate

    def update_prices(self, spot_price: float, perp_price: float) -> None:
        if spot_price == 0:
            self.basis_proxy = float("nan")
        else:
            self.basis_proxy = (perp_price - spot_price) / spot_price

    def update_oi(self, open_interest: float) -> None:
        if not math.isnan(self.last_oi) and self.last_oi != 0:
            self.oi_change = (open_interest - self.last_oi) / self.last_oi
        self.last_oi = open_interest

    def compute(self) -> dict[str, float]:
        nan = float("nan")

        # Funding momentum
        if math.isnan(self.funding_rate) or math.isnan(self.prev_funding_rate):
            funding_momentum = nan
        else:
            funding_momentum = self.funding_rate - self.prev_funding_rate

        # Annualized funding premium: rate * 3 periods/day * 365 days
        if math.isnan(self.funding_rate):
            funding_premium = nan
        else:
            funding_premium = self.funding_rate * 3.0 * 365.0

        # Annualized basis: basis * (365 * 24 * 4) assuming 15m measurement
        if math.isnan(self.basis_proxy):
            basis_annualized = nan
        else:
            basis_annualized = self.basis_proxy * 365.0 * 24.0 * 4.0

        return {
            "funding_rate": self.funding_rate,
            "funding_momentum": funding_momentum,
            "basis_proxy": self.basis_proxy,
            "oi_change_pct": self.oi_change,
            "funding_premium": funding_premium,
            "basis_annualized": basis_annualized,
        }
