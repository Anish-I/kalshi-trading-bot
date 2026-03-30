"""Weather trading orchestrator — scan, trade, settle, and report."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from kalshi.client import KalshiClient
from engine.risk import RiskManager
from engine.position_manager import PositionManager
from weather.nws_client import NWSClient
from weather.open_meteo_client import OpenMeteoClient
from weather.forecast_engine import WeatherForecastEngine
from weather.market_analyzer import WeatherMarketAnalyzer
from config.settings import WEATHER_CITIES, CITY_TIERS, settings
from engine.alerts import alert_trade_placed, alert_settlement, alert_risk_halt
from engine.order_ledger import OrderLedger, OrderRecord

logger = logging.getLogger(__name__)


def _get_city_tier(city_short: str) -> int:
    for tier, cities in CITY_TIERS.items():
        if city_short in cities:
            return tier
    return 2  # default to tier 2


class WeatherTrader:
    """End-to-end weather trading bot: scan for edge, place trades, settle."""

    def __init__(self) -> None:
        self._data_dir = Path(settings.DATA_DIR)
        self._state_path = self._data_dir / "weather_tracker_state.json"

        # --- Kalshi API ---
        self.kalshi_client = KalshiClient()

        # --- Risk management ---
        self.risk_manager = RiskManager(
            max_contracts=settings.WEATHER_MAX_CONTRACTS,
            daily_loss_limit_cents=int(settings.DAILY_LOSS_LIMIT_CENTS * 0.4),
            consecutive_loss_halt=2,
        )

        # --- Position tracking ---
        self.position_manager = PositionManager(
            state_path=self._data_dir / "weather_positions.json",
        )

        # --- Order ledger ---
        self.ledger = OrderLedger(str(self._data_dir))

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
        self._load_state()

        logger.info(
            "WeatherTrader initialised  |  max_contracts=%d  daily_loss=%dc  "
            "consecutive_halt=2  edge_threshold=%.0f%%",
            settings.WEATHER_MAX_CONTRACTS,
            int(settings.DAILY_LOSS_LIMIT_CENTS * 0.4),
            settings.WEATHER_EDGE_THRESHOLD * 100,
        )

    def _load_state(self) -> None:
        """Restore resting orders and risk counters after a restart."""
        if not self._state_path.exists():
            return

        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load weather tracker state from %s", self._state_path, exc_info=True)
            return

        self._resting_orders = {
            ticker: (
                data["order_id"],
                data["side"],
                int(data["count"]),
                int(data["price"]),
            )
            for ticker, data in (payload.get("resting_orders") or {}).items()
        }
        self._last_reset_date = payload.get("last_reset_date", "") or ""
        self.risk_manager.daily_pnl_cents = int(payload.get("daily_pnl_cents", 0) or 0)
        self.risk_manager.consecutive_losses = int(payload.get("consecutive_losses", 0) or 0)
        self.risk_manager.trades_today = payload.get("trades_today", []) or []
        self.risk_manager.is_active = bool(payload.get("risk_active", True))
        logger.info(
            "Loaded weather tracker state: %d resting orders, %d open positions",
            len(self._resting_orders),
            len(self.position_manager.positions),
        )

    def _save_state(self) -> None:
        payload = {
            "resting_orders": {
                ticker: {
                    "order_id": order_id,
                    "side": side,
                    "count": count,
                    "price": price,
                }
                for ticker, (order_id, side, count, price) in self._resting_orders.items()
            },
            "last_reset_date": self._last_reset_date,
            "daily_pnl_cents": self.risk_manager.daily_pnl_cents,
            "consecutive_losses": self.risk_manager.consecutive_losses,
            "trades_today": self.risk_manager.trades_today,
            "risk_active": self.risk_manager.is_active,
        }
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        tmp_path.replace(self._state_path)

    @staticmethod
    def _extract_fill_details(order_data: dict, fallback_count: int, fallback_price_cents: int) -> tuple[int, int]:
        """Best-effort parse of Kalshi fill size and average fill price."""
        count_value = (
            order_data.get("fill_count")
            or order_data.get("fill_count_fp")
            or order_data.get("count")
            or order_data.get("count_fp")
            or fallback_count
        )
        try:
            fill_count = int(round(float(count_value)))
        except Exception:
            fill_count = fallback_count
        fill_count = max(fill_count, 1)

        fill_price_cents = fallback_price_cents
        for key, multiplier in (
            ("average_fill_price", 1),
            ("average_fill_price_dollars", 100),
            ("yes_price", 1),
            ("no_price", 1),
            ("yes_price_dollars", 100),
            ("no_price_dollars", 100),
        ):
            raw = order_data.get(key)
            if raw is None:
                continue
            try:
                fill_price_cents = int(round(float(raw) * multiplier))
                break
            except Exception:
                continue

        return fill_count, fill_price_cents

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
                    fill_count, fill_cost = self._extract_fill_details(order_data, count, price)
                    self.position_manager.open_position(
                        ticker=ticker, side=side, count=fill_count,
                        entry_price_cents=fill_cost,
                    )
                    resolved.append(ticker)
                    logger.info("Weather resting order FILLED + position opened: %s %s x%d @ %dc",
                                ticker, side, fill_count, fill_cost)
                    self.ledger.update_status(ticker, "filled", filled_price_cents=fill_cost, filled_count=fill_count)
                    self.ledger.save()
                elif status in ("canceled", "expired"):
                    resolved.append(ticker)
                    logger.info("Weather resting order %s: %s — no position", status, ticker)
            except Exception:
                pass
        for t in resolved:
            self._resting_orders.pop(t, None)
        if resolved:
            self._save_state()

    # ------------------------------------------------------------------ #
    #  Daily reset
    # ------------------------------------------------------------------ #

    def daily_reset_if_needed(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._last_reset_date:
            if self._last_reset_date:
                logger.info("Weather daily reset")
            self.risk_manager.reset_daily()
            self._last_reset_date = today
            self._save_state()

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
        submitted: list[dict] = []

        for opp in opportunities[:max_trades]:
            ticker = opp["ticker"]
            order_contracts = contracts_per_trade

            # --- Balance floor guardrail ---
            try:
                bal = self.kalshi_client.get_balance()
                if bal < 10.0:
                    logger.warning("Balance floor hit ($%.2f < $10). Stopping.", bal)
                    break
            except Exception:
                pass

            # --- Risk gate ---
            can_trade, reason = self.risk_manager.can_trade()
            if not can_trade:
                logger.warning("Risk manager blocked trading: %s", reason)
                alert_risk_halt(reason, self.risk_manager.daily_pnl_cents, strategy="weather")
                break

            # --- Duplicate position check ---
            if self.position_manager.get_position(ticker) is not None or ticker in self._resting_orders:
                logger.info("Already have open position on %s, skipping", ticker)
                continue

            # --- City tier sizing (uses config multipliers) ---
            city_short = opp.get("city_short", "")
            tier = _get_city_tier(city_short)
            if tier == 3 and not settings.WEATHER_TIER3_LIVE_ENABLED:
                logger.info("Skipping tier-3 city %s (MAE > 3.5F)", city_short)
                continue
            elif tier == 2:
                order_contracts = max(1, int(order_contracts * settings.WEATHER_TIER2_SIZE_MULTIPLIER))
            elif tier == 1:
                order_contracts = max(1, int(order_contracts * settings.WEATHER_TIER1_SIZE_MULTIPLIER))

            # --- Bracket sizing ---
            if opp.get("strike_type") == "between":
                if opp["edge"] < 0.25:
                    logger.info("Skipping bracket %s (edge %.0f%% < 25%%)", ticker, opp["edge"] * 100)
                    continue
                order_contracts = max(1, order_contracts // 3)

            size_ok, size_reason = self.risk_manager.check_order_size(order_contracts)
            if not size_ok:
                logger.warning("Order size rejected: %s", size_reason)
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
                    count=order_contracts,
                    order_type="limit",
                    yes_price=yes_price,
                    no_price=no_price,
                )
                logger.info(
                    "ORDER PLACED  ticker=%s  side=%s  price=%dc  contracts=%d  edge=%.1f%%  "
                    "order_id=%s",
                    ticker, side, price_cents, order_contracts,
                    opp["edge"] * 100,
                    order_resp.get("order", {}).get("order_id", "unknown"),
                )
                alert_trade_placed(ticker, side, price_cents, order_contracts,
                                   opp["edge"] * 100, strategy="weather")
            except Exception:
                logger.error("Failed to place order on %s", ticker, exc_info=True)
                continue

            # --- Record position only if filled ---
            order_data = order_resp.get("order", order_resp)
            status = order_data.get("status", "")
            order_id = order_data.get("order_id", "")

            if status == "executed":
                fill_count, fill_cost = self._extract_fill_details(
                    order_data,
                    order_contracts,
                    price_cents,
                )
                self.position_manager.open_position(
                    ticker=ticker, side=side, count=fill_count,
                    entry_price_cents=fill_cost,
                )
                city_type = opp.get("strike_type", "")
                self.ledger.add(OrderRecord(strategy="weather", market_type=f"weather_{city_type}", ticker=ticker, order_id=order_id, status="filled", submitted_side=side, submitted_price_cents=price_cents, submitted_count=fill_count, filled_price_cents=fill_cost, filled_count=fill_count))
                self.ledger.save()
            elif status == "resting" and order_id:
                self._resting_orders[ticker] = (order_id, side, order_contracts, price_cents)
                logger.info("Weather order resting: %s (%s)", ticker, order_id)
                self.ledger.add(OrderRecord(strategy="weather", market_type=f"weather_{opp.get('strike_type','')}", ticker=ticker, order_id=order_id, status="resting", submitted_side=side, submitted_price_cents=price_cents, submitted_count=order_contracts))
                self.ledger.save()

            submitted.append({
                "ticker": ticker,
                "side": side,
                "price_cents": price_cents,
                "contracts": order_contracts,
                "edge": opp["edge"],
                "city": opp.get("city", ""),
                "model_prob": opp.get("model_prob", 0),
                "market_mid": opp.get("market_mid", 0),
                "status": status,
                "order_response": order_resp,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        if submitted:
            self._save_state()

        logger.info("Submitted %d / %d possible weather orders", len(submitted), len(opportunities))
        return submitted

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
            if status not in ("settled", "finalized"):
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

            self.ledger.settle(ticker, result, pnl)
            self.ledger.save()

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
            alert_settlement(ticker, outcome, pnl, strategy="weather")

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
            self._save_state()

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
            "resting_order_count": len(self._resting_orders),
            "daily_pnl_cents": self.risk_manager.daily_pnl_cents,
            "realized_pnl_cents": self.position_manager.realized_pnl_cents,
            "consecutive_losses": self.risk_manager.consecutive_losses,
            "trades_today": len(self.risk_manager.trades_today),
            "risk_active": self.risk_manager.is_active,
            "can_trade": can_trade,
            "risk_status": risk_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
