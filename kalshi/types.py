"""Typed response models for Kalshi API."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OrderResponse:
    order_id: str = ""
    status: str = ""  # resting, executed, canceled, pending
    ticker: str = ""
    side: str = ""
    action: str = ""
    count: int = 0
    yes_price: int = 0
    no_price: int = 0
    created_time: str = ""
    expiration_time: str = ""
    order_type: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "OrderResponse":
        return cls(
            order_id=d.get("order_id", ""),
            status=d.get("status", ""),
            ticker=d.get("ticker", ""),
            side=d.get("side", ""),
            action=d.get("action", ""),
            count=d.get("count", 0),
            yes_price=d.get("yes_price", 0),
            no_price=d.get("no_price", 0),
            created_time=d.get("created_time", ""),
            expiration_time=d.get("expiration_time", ""),
            order_type=d.get("type", d.get("order_type", "")),
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in ("executed", "canceled", "expired")

    @property
    def fill_price_cents(self) -> int:
        """Best guess at fill price from the order response."""
        if self.side == "yes" and self.yes_price:
            return self.yes_price
        if self.side == "no" and self.no_price:
            return self.no_price
        return 0


@dataclass
class MarketResponse:
    ticker: str = ""
    status: str = ""  # active, finalized, settled
    result: str = ""  # yes, no, or empty
    yes_bid: float = 0.0
    yes_ask: float = 0.0
    no_bid: float = 0.0
    no_ask: float = 0.0
    volume: int = 0
    title: str = ""
    close_time: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "MarketResponse":
        return cls(
            ticker=d.get("ticker", ""),
            status=d.get("status", ""),
            result=d.get("result", ""),
            yes_bid=float(d.get("yes_bid", d.get("yes_bid_dollars", 0)) or 0),
            yes_ask=float(d.get("yes_ask", d.get("yes_ask_dollars", 0)) or 0),
            no_bid=float(d.get("no_bid", d.get("no_bid_dollars", 0)) or 0),
            no_ask=float(d.get("no_ask", d.get("no_ask_dollars", 0)) or 0),
            volume=int(d.get("volume", 0) or 0),
            title=d.get("title", ""),
            close_time=d.get("close_time", ""),
        )

    @property
    def is_settled(self) -> bool:
        return self.status in ("settled", "finalized")


@dataclass
class OrderbookLevel:
    price: int = 0  # cents
    quantity: int = 0

    @classmethod
    def from_list(cls, items: list) -> list["OrderbookLevel"]:
        return [cls(price=int(item[0]), quantity=int(item[1])) for item in items]


@dataclass
class OrderbookResponse:
    yes: list[OrderbookLevel] = field(default_factory=list)
    no: list[OrderbookLevel] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "OrderbookResponse":
        return cls(
            yes=OrderbookLevel.from_list(d.get("yes", d.get("orderbook", {}).get("yes", []))),
            no=OrderbookLevel.from_list(d.get("no", d.get("orderbook", {}).get("no", []))),
        )

    @property
    def best_yes_ask(self) -> int:
        return self.yes[0].price if self.yes else 0

    @property
    def best_no_ask(self) -> int:
        return self.no[0].price if self.no else 0

    @property
    def spread_cents(self) -> int:
        if not self.yes or not self.no:
            return 0
        return 100 - self.best_yes_ask - self.best_no_ask
