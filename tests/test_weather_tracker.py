"""Tests for weather tracker state persistence and resting-order handling."""
import sys

sys.path.insert(0, ".")


class DummyKalshiClient:
    next_statuses: list[str] = []

    def __init__(self):
        self.order_responses: dict[str, dict] = {}

    def get_balance(self):
        return 100.0

    def place_order(self, ticker, side, action, count, order_type="limit", yes_price=None, no_price=None):
        status = self.next_statuses.pop(0) if self.next_statuses else "executed"
        price = yes_price if yes_price is not None else no_price
        order_id = f"order-{len(self.order_responses) + 1}"
        order = {
            "status": status,
            "order_id": order_id,
            "count": count,
            "average_fill_price": price,
        }
        self.order_responses[order_id] = order
        return {"order": order}

    def _request(self, method, path):
        order_id = path.rsplit("/", 1)[-1]
        return {"order": self.order_responses[order_id]}

    def get_market(self, ticker):
        return {"status": "open", "result": ""}


class DummyNWSClient:
    def __init__(self, user_agent=""):
        self.user_agent = user_agent


class DummyOpenMeteoClient:
    pass


class DummyForecastEngine:
    def __init__(self, nws_client, meteo_client):
        self.nws_client = nws_client
        self.meteo_client = meteo_client


class DummyAnalyzer:
    def __init__(self, kalshi_client, forecast_engine):
        self.kalshi_client = kalshi_client
        self.forecast_engine = forecast_engine


def _build_trader(monkeypatch, tmp_path):
    import weather.trader as weather_trader_module

    monkeypatch.setattr(weather_trader_module.settings, "DATA_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(weather_trader_module, "KalshiClient", DummyKalshiClient)
    monkeypatch.setattr(weather_trader_module, "NWSClient", DummyNWSClient)
    monkeypatch.setattr(weather_trader_module, "OpenMeteoClient", DummyOpenMeteoClient)
    monkeypatch.setattr(weather_trader_module, "WeatherForecastEngine", DummyForecastEngine)
    monkeypatch.setattr(weather_trader_module, "WeatherMarketAnalyzer", DummyAnalyzer)

    return weather_trader_module.WeatherTrader()


def test_weather_state_persists_positions_and_resting_orders(tmp_path, monkeypatch):
    trader = _build_trader(monkeypatch, tmp_path)
    trader.position_manager.open_position("KXTEST-ONE", "yes", 3, 54)
    trader._resting_orders["KXTEST-TWO"] = ("order-7", "no", 4, 42)
    trader.risk_manager.daily_pnl_cents = -125
    trader.risk_manager.consecutive_losses = 2
    trader.risk_manager.trades_today = [{"ticker": "KXTEST-ONE"}]
    trader._last_reset_date = "2026-03-26"
    trader._save_state()

    reloaded = _build_trader(monkeypatch, tmp_path)
    pos = reloaded.position_manager.get_position("KXTEST-ONE")

    assert pos is not None
    assert pos["count"] == 3
    assert pos["entry_price"] == 54
    assert reloaded._resting_orders["KXTEST-TWO"] == ("order-7", "no", 4, 42)
    assert reloaded.risk_manager.daily_pnl_cents == -125
    assert reloaded.risk_manager.consecutive_losses == 2
    assert len(reloaded.risk_manager.trades_today) == 1


def test_daily_reset_preserves_resting_orders(tmp_path, monkeypatch):
    trader = _build_trader(monkeypatch, tmp_path)
    trader._resting_orders["KXTEST-REST"] = ("order-3", "yes", 2, 63)
    trader.risk_manager.daily_pnl_cents = -250
    trader.risk_manager.trades_today = [{"ticker": "KXTEST-REST"}]
    trader._last_reset_date = "1900-01-01"

    trader.daily_reset_if_needed()

    assert trader._resting_orders["KXTEST-REST"] == ("order-3", "yes", 2, 63)
    assert trader.risk_manager.daily_pnl_cents == 0
    assert trader.risk_manager.trades_today == []


def test_execute_trades_tracks_status_and_skips_resting_duplicate(tmp_path, monkeypatch):
    DummyKalshiClient.next_statuses = ["resting"]
    trader = _build_trader(monkeypatch, tmp_path)
    opportunities = [
        {
            "ticker": "KXTEST-DUP",
            "side": "YES",
            "suggested_price_cents": 55,
            "edge": 0.2,
            "city": "Phoenix",
            "city_short": "PHX",
            "model_prob": 0.75,
            "market_mid": 0.52,
        },
        {
            "ticker": "KXTEST-DUP",
            "side": "YES",
            "suggested_price_cents": 55,
            "edge": 0.18,
            "city": "Phoenix",
            "city_short": "PHX",
            "model_prob": 0.73,
            "market_mid": 0.51,
        },
    ]

    submitted = trader.execute_trades(opportunities, max_trades=2, contracts_per_trade=5)

    assert len(submitted) == 1
    assert submitted[0]["status"] == "resting"
    assert trader.position_manager.get_position("KXTEST-DUP") is None
    assert trader._resting_orders["KXTEST-DUP"] == ("order-1", "yes", 5, 55)
