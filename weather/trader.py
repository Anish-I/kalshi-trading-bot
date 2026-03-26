"""Weather trading orchestrator — scan, trade, settle, and report."""

import logging
from datetime import datetime, timezone

from kalshi.client import KalshiClient
from engine.risk import RiskManager
from engine.position_manager import PositionManager
from weather.nws_client import NWSClient
from weather.open_meteo_client import OpenMeteoClient
from weather.forecast_engine import WeatherForecastEngine
from weather.market_analyzer import WeatherMarketAnalyzer
from config.settings import WEATHER_CITIES, settings

logger = logging.getLogger(__name__)


class WeatherTrader:
    """End-to-end weather trading bot: scan for edge, place trades, settle."""

    def __init__(self) -> None:
        # --- Kalshi API ---
        self.kalshi_client = KalshiClient()

        # --- Risk management ---
        self.risk_manager = RiskManager(
            max_contracts=settings.WEATHER_MAX_CONTRACTS,
            daily_loss_limit_cents=int(settings.DAILY_LOSS_LIMIT_CENTS * 0.4),
            consecutive_loss_halt=3,
        )

        # --- Position tracking ---
        self.position_manager = PositionManager()

        # --- Resting order tracking ---
        self._resting_orders: dict[str, tuple] = {}  # ticker -> (order_id, side, count, price)

        # --- Weather data sources ---
        self.nws_client = NWSClient(user_agent=settings.NWS_USER_AGENT)
        self.open_meteo_client = OpenMeteoClient()

        # --- Forecast engine ---
        self.forecast_engine = WeatherForecastEngine(
            nws_client=self.nws_client,
            meteo_client=self.open_meteo_client,
        )

        # --- Market analyzer ---
        self.analyzer = WeatherMarketAnalyzer(
            kalshi_client=self.kalshi_client,
            forecast_engine=self.forecast_engine,
        )

        self._last_reset_date: str = ""

        logger.info(
            "WeatherTrader initialised  |  max_contracts=%d  daily_loss=%dc  "
            "consecutive_halt=3  edge_threshold=%.0f%%",
            settings.WEATHER_MAX_CONTRACTS,
            int(settings.DAILY_LOSS_LIMIT_CENTS * 0.4),
            settings.WEATHER_EDGE_THRESHOLD * 100,
        )

    # ------------------------------------------------------------------ #
    #  Resting order reconciliation
    # ------------------------------------------------------------------ #

    def check_resting_orders(self) -> None:
        resolved = []
        for ticker, (order_id, side, count, price) in self._resting_orders.items():
            try:
                resp = self.kalshi_client._request("GET", f"/portfolio/orders/{order_id}")
                order_data = resp.get("order", resp)
                status = order_data.get("status", "")
                if status == "executed":
                    # NOW open the position since it filled — use actual fill data
                    fill_count = int(float(order_data.get("count", count)))
                    fill_cost = order_data.get("average_fill_price", price)
                    self.position_manager.open_position(
                        ticker=ticker, side=side, count=fill_count,
                        entry_price_cents=int(float(str(fill_cost))),
                    )
                    resolved.append(ticker)
                    logger.info("Weather resting order FILLED + position opened: %s %s x%d @ %dc",
                                ticker, side, fill_count, int(float(str(fill_cost))))
                elif status in ("canceled", "expired"):
                    resolved.append(ticker)
                    logger.info("Weather resting order %s: %s — no position", status, ticker)
            except Exception:
                pass
        for t in resolved:
            self._resting_orders.pop(t, None)

    # ------------------------------------------------------------------ #
    #  Daily reset
    # ------------------------------------------------------------------ #

    def daily_reset_if_needed(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._last_reset_date:
            if self._last_reset_date:
                logger.info("Weather daily reset")
            self.risk_manager.reset_daily()
            self._resting_orders.clear()
            self._last_reset_date = today

    # ------------------------------------------------------------------ #
    #  Scan
    # ------------------------------------------------------------------ #

    def run_scan(self) -> list[dict]:
        """Scan all configured cities for mispriced weather markets.

        Returns a list of opportunity dicts sorted by edge descending.
        """
        logger.info("Starting weather market scan across %d cities …", len(WEATHER_CITIES))

        opportunities = self.analyzer.find_best_trades(
            WEATHER_CITIES,
            min_edge=settings.WEATHER_EDGE_THRESHOLD,
        )

        if opportunities:
            logger.info("Scan complete: %d opportunities found", len(opportunities))
        else:
            logger.info("Scan complete: no opportunities above %.0f%% edge",
                        settings.WEATHER_EDGE_THRESHOLD * 100)

        return opportunities

    # ------------------------------------------------------------------ #
    #  Trade execution
    # ------------------------------------------------------------------ #

    def execute_trades(
        self,
        opportunities: list[dict],
        max_trades: int = 3,
        contracts_per_trade: int = 10,
    ) -> list[dict]:
        """Place orders on the best opportunities.

        Parameters
        ----------
        opportunities : list[dict]
            Output from ``run_scan()``.
        max_trades : int
            Maximum number of new trades to place.
        contracts_per_trade : int
            Contracts per order.

        Returns
        -------
        list[dict]
            Records of executed trades.
        """
        executed: list[dict] = []

        for opp in opportunities[:max_trades]:
            ticker = opp["ticker"]

            # --- Risk gate ---
            can_trade, reason = self.risk_manager.can_trade()
            if not can_trade:
                logger.warning("Risk manager blocked trading: %s", reason)
                break

            size_ok, size_reason = self.risk_manager.check_order_size(contracts_per_trade)
            if not size_ok:
                logger.warning("Order size rejected: %s", size_reason)
                continue

            # --- Duplicate position check ---
            if self.position_manager.get_position(ticker) is not None or ticker in self._resting_orders:
                logger.info("Already have open position on %s, skipping", ticker)
                continue

            # --- Build order ---
            side = opp["side"].lower()  # "yes" or "no"
            price_cents = int(opp["suggested_price_cents"])

            yes_price = price_cents if side == "yes" else None
            no_price = price_cents if side == "no" else None

            # --- Place order ---
            try:
                order_resp = self.kalshi_client.place_order(
                    ticker=ticker,
                    side=side,
                    action="buy",
                    count=contracts_per_trade,
                    order_type="limit",
                    yes_price=yes_price,
                    no_price=no_price,
                )
                logger.info(
                    "ORDER PLACED  ticker=%s  side=%s  price=%dc  contracts=%d  edge=%.1f%%  "
                    "order_id=%s",
                    ticker, side, price_cents, contracts_per_trade,
                    opp["edge"] * 100,
                    order_resp.get("order", {}).get("order_id", "unknown"),
                )
            except Exception:
                logger.error("Failed to place order on %s", ticker, exc_info=True)
                continue

            # --- Record position only if filled ---
            order_data = order_resp.get("order", order_resp)
            status = order_data.get("status", "")
            order_id = order_data.get("order_id", "")

            if status == "executed":
                # Use actual fill data if available
                fill_count = int(float(order_data.get("count", contracts_per_trade)))
                fill_cost = order_data.get("average_fill_price", price_cents)
                self.position_manager.open_position(
                    ticker=ticker, side=side, count=fill_count,
                    entry_price_cents=int(float(str(fill_cost))),
                )
            elif status == "resting" and order_id:
                self._resting_orders[ticker] = (order_id, side, contracts_per_trade, price_cents)
                logger.info("Weather order resting: %s (%s)", ticker, order_id)

            executed.append({
                "ticker": ticker,
                "side": side,
                "price_cents": price_cents,
                "contracts": contracts_per_trade,
                "edge": opp["edge"],
                "city": opp.get("city", ""),
                "model_prob": opp.get("model_prob", 0),
                "market_mid": opp.get("market_mid", 0),
                "order_response": order_resp,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        logger.info("Executed %d / %d possible trades", len(executed), len(opportunities))
        return executed

    # ------------------------------------------------------------------ #
    #  Settlement
    # ------------------------------------------------------------------ #

    def check_settlements(self) -> list[dict]:
        """Check all open positions for settlement and close resolved ones.

        Returns a list of settlement records.
        """
        settled: list[dict] = []
        tickers = list(self.position_manager.positions.keys())

        if not tickers:
            logger.debug("No open positions to check for settlement")
            return settled

        logger.info("Checking settlement for %d open positions …", len(tickers))

        for ticker in tickers:
            try:
                market = self.kalshi_client.get_market(ticker)
            except Exception:
                logger.error("Failed to fetch market info for %s", ticker, exc_info=True)
                continue

            status = market.get("status", "")
            if status != "settled":
                logger.debug("%s status=%s (not settled)", ticker, status)
                continue

            # Determine outcome
            result = market.get("result", "")
            pos = self.position_manager.get_position(ticker)
            if pos is None:
                continue

            position_side = pos["side"]
            settled_yes = result.lower() == "yes"

            # Close position and compute P&L
            pnl = self.position_manager.close_position(ticker, settled_yes)

            # Record in risk manager
            trade_info = {
                "ticker": ticker,
                "side": position_side,
                "result": result,
                "entry_price": pos["entry_price"],
                "count": pos["count"],
            }
            self.risk_manager.record_trade(pnl, trade_info)

            outcome = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAK-EVEN")
            logger.info(
                "SETTLED  %s  side=%s  result=%s  pnl=%+dc  outcome=%s",
                ticker, position_side, result, pnl, outcome,
            )

            settled.append({
                "ticker": ticker,
                "side": position_side,
                "result": result,
                "pnl_cents": pnl,
                "outcome": outcome,
                "settled_at": datetime.now(timezone.utc).isoformat(),
            })

        if settled:
            logger.info(
                "Settlement summary: %d closed, total P&L %+dc",
                len(settled), sum(s["pnl_cents"] for s in settled),
            )

        return settled

    # ------------------------------------------------------------------ #
    #  Status
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        """Return a snapshot of the trader's current state."""
        try:
            balance = self.kalshi_client.get_balance()
        except Exception:
            logger.warning("Could not fetch balance", exc_info=True)
            balance = None

        open_positions = {
            ticker: {
                "side": pos["side"],
                "count": pos["count"],
                "entry_price": pos["entry_price"],
                "opened_at": pos["timestamp"],
            }
            for ticker, pos in self.position_manager.positions.items()
        }

        can_trade, risk_reason = self.risk_manager.can_trade()

        return {
            "balance_usd": balance,
            "open_positions": open_positions,
            "open_position_count": len(open_positions),
            "daily_pnl_cents": self.risk_manager.daily_pnl_cents,
            "realized_pnl_cents": self.position_manager.realized_pnl_cents,
            "consecutive_losses": self.risk_manager.consecutive_losses,
            "trades_today": len(self.risk_manager.trades_today),
            "risk_active": self.risk_manager.is_active,
            "can_trade": can_trade,
            "risk_status": risk_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
