"""
Crypto Pair Trader: buy both YES and NO for guaranteed profit.

Modes:
  --sim   (default): evaluate opportunities, log virtual trades
  --paper: place real limit orders and manage fills/orphans
  --live:  same execution path as paper with live-mode labeling

Run: python scripts/crypto_pair_trader.py --sim
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

sys.path.insert(0, ".")

from config.settings import settings
from engine.alerts import WARNING, alert, alert_trade_placed
from engine.kalshi_ws import KalshiWebSocketClient, dollars_to_cents, fp_to_int
from engine.pair_pricing import evaluate_pair_opportunity
from engine.pair_risk import PairRiskManager
from engine.pair_state import PairTrade, PairTracker
from engine.process_lock import ProcessLock
from engine.pre_trade_gate import PreTradeGate, GateContext
from engine.risk import RiskManager
from engine.family_limits import FamilyLimits

# Phase 1b: shared pre-trade gate (additive — existing risk checks remain authoritative).
_GATE_RISK_MGR = RiskManager(
    max_contracts=10_000,
    daily_loss_limit_cents=10_000_000,
    consecutive_loss_halt=10_000,
)
_GATE_FAMILY_LIMITS = FamilyLimits()
_GATE_FAMILY_LIMITS._cooldown_s = 0.0
from engine.family_scorecard import FamilyScorecard as _FamilyScorecard
_scorecard = _FamilyScorecard(
    window_hours=settings.SCORECARD_WINDOW_HOURS,
    cache_path=Path(settings.SCORECARD_CACHE_PATH),
    shadow_mode=settings.SCORECARD_SHADOW_MODE,
)
def _scorecard_hook(ctx):
    h = _scorecard.get_family_health(ctx.family)
    return (
        h.healthy or settings.SCORECARD_SHADOW_MODE,
        f"family={ctx.family} score={h.score:.2f}",
        h.metrics,
    )
_pre_trade_gate = PreTradeGate(
    risk_mgr=_GATE_RISK_MGR,
    family_limits=_GATE_FAMILY_LIMITS,
    scorecard_hook=_scorecard_hook,
)
_gate_state: dict = {"last": None}
from kalshi.client import KalshiClient
from kalshi.market_discovery import KXBTCMarketTracker

TERMINAL_ORDER_STATUSES = {"executed", "canceled", "cancelled", "expired"}
CANCELED_ORDER_STATUSES = {"canceled", "cancelled", "expired"}
HEARTBEAT_S = 1.0


@dataclass
class ActivePairExecution:
    ticker: str
    pair: PairTrade
    yes_price: int
    no_price: int
    pair_cost_cents: int
    placed_at_monotonic: float
    yes_order_id: str = ""
    no_order_id: str = ""
    yes_status: str = ""
    no_status: str = ""
    first_fill_at_monotonic: float | None = None
    cancel_requested: bool = False
    cancel_reason: str = ""

    def unfilled_side(self) -> str:
        if not self.pair.yes_filled:
            return "yes"
        if not self.pair.no_filled:
            return "no"
        return ""

    def unfilled_order_id(self) -> str:
        side = self.unfilled_side()
        if side == "yes":
            return self.yes_order_id
        if side == "no":
            return self.no_order_id
        return ""

    def unfilled_price_cents(self) -> int:
        side = self.unfilled_side()
        if side == "yes":
            return self.yes_price
        if side == "no":
            return self.no_price
        return 0

    def unfilled_leg_status(self) -> str:
        side = self.unfilled_side()
        if side == "yes":
            return self.yes_status
        if side == "no":
            return self.no_status
        return ""

    @property
    def orphan_age_s(self) -> float:
        if self.first_fill_at_monotonic is None:
            return 0.0
        return time.monotonic() - self.first_fill_at_monotonic

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "state": self.pair.state,
            "yes_order_id": self.yes_order_id,
            "no_order_id": self.no_order_id,
            "yes_status": self.yes_status,
            "no_status": self.no_status,
            "yes_filled": self.pair.yes_filled,
            "no_filled": self.pair.no_filled,
            "yes_price": self.pair.yes_filled_price if self.pair.yes_filled else self.yes_price,
            "no_price": self.pair.no_filled_price if self.pair.no_filled else self.no_price,
            "pair_cost_cents": self.pair.pair_cost_cents,
            "pair_profit_cents": self.pair.pair_profit_cents,
            "orphan_age_s": round(self.orphan_age_s, 1) if self.pair.is_orphan else 0.0,
            "cancel_requested": self.cancel_requested,
            "cancel_reason": self.cancel_reason,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sim", "paper", "live"], default="sim")
    parser.add_argument("--interval", type=int, default=30, help="Scan interval seconds")
    parser.add_argument("--pair-cap", type=int, default=96, help="Max pair cost in cents")
    return parser


def configure_logging(mode: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"crypto_pair_{mode}.log"),
        ],
    )
    return logging.getLogger("pair_trader")


def _extract_order_payload(response: dict | None) -> dict:
    if not isinstance(response, dict):
        return {}
    order = response.get("order")
    return order if isinstance(order, dict) else response


def _coerce_cents(value: object) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return dollars_to_cents(value)


def _extract_order_price_cents(order: dict, side: str, fallback: int) -> int:
    normalized_side = side.lower()
    cents_key = "yes_price" if normalized_side == "yes" else "no_price"
    dollars_key = "yes_price_dollars" if normalized_side == "yes" else "no_price_dollars"

    cents = _coerce_cents(order.get(cents_key))
    if cents:
        return cents

    dollars = dollars_to_cents(order.get(dollars_key))
    if dollars:
        return dollars

    return fallback


def _extract_order_fill_count(order: dict) -> int:
    fill_count = fp_to_int(order.get("fill_count_fp", order.get("fill_count")))
    status = str(order.get("status", "")).lower()
    if fill_count == 0 and status == "executed":
        return 1
    return fill_count


def _set_leg_status(execution: ActivePairExecution, side: str, status: str) -> None:
    normalized = status.lower()
    if side == "yes":
        execution.yes_status = normalized
    elif side == "no":
        execution.no_status = normalized


def _cleanup_execution(
    execution: ActivePairExecution,
    active_pairs: dict[str, ActivePairExecution],
    order_to_ticker: dict[str, str],
) -> None:
    active_pairs.pop(execution.ticker, None)
    if execution.yes_order_id:
        order_to_ticker.pop(execution.yes_order_id, None)
    if execution.no_order_id:
        order_to_ticker.pop(execution.no_order_id, None)


def _record_leg_fill_locked(
    execution: ActivePairExecution,
    side: str,
    price_cents: int,
    count: int,
    *,
    source: str,
    log: logging.Logger,
) -> None:
    qty = max(count, 1)
    fill_price = price_cents or (execution.yes_price if side == "yes" else execution.no_price)

    if side == "yes":
        if execution.pair.yes_filled:
            return
        execution.pair.record_yes_fill(fill_price, qty)
        execution.yes_status = "executed"
    elif side == "no":
        if execution.pair.no_filled:
            return
        execution.pair.record_no_fill(fill_price, qty)
        execution.no_status = "executed"
    else:
        return

    if execution.first_fill_at_monotonic is None:
        execution.first_fill_at_monotonic = time.monotonic()

    log.info(
        "PAIR FILL %s %s @ %dc via %s",
        execution.ticker,
        side.upper(),
        fill_price,
        source,
    )


def _resolve_orphan_locked(
    execution: ActivePairExecution,
    *,
    reason: str,
    pair_tracker: PairTracker,
    risk: PairRiskManager,
    active_pairs: dict[str, ActivePairExecution],
    order_to_ticker: dict[str, str],
    log: logging.Logger,
    release_exposure: bool,
) -> None:
    resolved = pair_tracker.resolve_orphan(execution.ticker, reason)
    if resolved is None:
        return

    loss_cents = (
        resolved.yes_filled_price
        or resolved.no_filled_price
        or (execution.yes_price if resolved.yes_filled else execution.no_price)
    )
    risk.record_orphan(loss_cents)
    if release_exposure:
        risk.exposure_cents = max(0, risk.exposure_cents - execution.pair_cost_cents)

    _cleanup_execution(execution, active_pairs, order_to_ticker)

    log.warning(
        "ORPHAN UNWIND %s filled_side=%s reason=%s loss<=%dc",
        execution.ticker,
        "YES" if resolved.yes_filled else "NO",
        reason,
        loss_cents,
    )
    alert(
        WARNING,
        "PAIR_ORPHAN",
        f"{execution.ticker} orphan unwind ({reason})",
        filled_side="yes" if resolved.yes_filled else "no",
        loss_cents=loss_cents,
    )


def _finalize_execution_locked(
    execution: ActivePairExecution,
    *,
    pair_tracker: PairTracker,
    risk: PairRiskManager,
    active_pairs: dict[str, ActivePairExecution],
    order_to_ticker: dict[str, str],
    log: logging.Logger,
) -> None:
    if execution.ticker not in active_pairs:
        return

    if execution.pair.is_complete:
        completed = pair_tracker.complete_pair(execution.ticker)
        if completed is not None:
            risk.record_pair_complete(execution.pair_cost_cents)
        _cleanup_execution(execution, active_pairs, order_to_ticker)
        log.info(
            "PAIR COMPLETE %s cost=%dc profit=%dc",
            execution.ticker,
            execution.pair.pair_cost_cents,
            execution.pair.pair_profit_cents,
        )
        return

    if execution.pair.is_orphan and execution.unfilled_leg_status() in CANCELED_ORDER_STATUSES:
        _resolve_orphan_locked(
            execution,
            reason=execution.cancel_reason or execution.unfilled_leg_status(),
            pair_tracker=pair_tracker,
            risk=risk,
            active_pairs=active_pairs,
            order_to_ticker=order_to_ticker,
            log=log,
            release_exposure=True,
        )


def _apply_order_snapshot_locked(
    execution: ActivePairExecution,
    order: dict,
    *,
    source: str,
    log: logging.Logger,
) -> None:
    side = str(order.get("side", "")).lower()
    if side not in {"yes", "no"}:
        return

    status = str(order.get("status", "")).lower()
    if status:
        _set_leg_status(execution, side, status)

    fill_count = _extract_order_fill_count(order)
    if fill_count > 0:
        fill_price = _extract_order_price_cents(
            order,
            side,
            execution.yes_price if side == "yes" else execution.no_price,
        )
        _record_leg_fill_locked(
            execution,
            side,
            fill_price,
            fill_count,
            source=source,
            log=log,
        )


def _build_state_payload(
    *,
    mode: str,
    scan_count: int,
    latest_scan: dict,
    pair_tracker: PairTracker,
    risk: PairRiskManager,
    active_pairs: dict[str, ActivePairExecution],
    ws_client: KalshiWebSocketClient | None,
    state_lock: Lock,
) -> dict:
    with state_lock:
        active = [execution.to_dict() for execution in active_pairs.values()]
        stats = pair_tracker.get_stats()
        risk_summary = risk.summary()

    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "scan": scan_count,
        "ws_connected": ws_client.is_connected if ws_client else False,
        "active_pairs": len(active),
        "active_pair_details": active,
        "stats": stats,
        "risk": risk_summary,
        "gate_last_decision": _gate_state["last"],
        **latest_scan,
    }


def _place_real_pair(
    *,
    client: KalshiClient,
    ticker: str,
    yes_price: int,
    no_price: int,
    pair_cost_cents: int,
    net_cents: float,
    close_time: str,
    strategy_label: str,
    traded_tickers: set[str],
    pair_tracker: PairTracker,
    risk: PairRiskManager,
    active_pairs: dict[str, ActivePairExecution],
    order_to_ticker: dict[str, str],
    state_lock: Lock,
    pair_log: Path,
    log: logging.Logger,
) -> bool:
    with state_lock:
        pair = pair_tracker.start_pair(ticker, yes_price, no_price, close_time)
        execution = ActivePairExecution(
            ticker=ticker,
            pair=pair,
            yes_price=yes_price,
            no_price=no_price,
            pair_cost_cents=pair_cost_cents,
            placed_at_monotonic=time.monotonic(),
        )
        active_pairs[ticker] = execution

        try:
            yes_order = _extract_order_payload(
                client.place_order(
                    ticker=ticker,
                    side="yes",
                    action="buy",
                    count=1,
                    order_type="limit",
                    yes_price=yes_price,
                )
            )
            yes_order.setdefault("side", "yes")
            execution.yes_order_id = yes_order.get("order_id", "")
            pair.yes_order_id = execution.yes_order_id
            if not execution.yes_order_id:
                raise RuntimeError("YES order placement returned no order_id")
            order_to_ticker[execution.yes_order_id] = ticker
            _apply_order_snapshot_locked(execution, yes_order, source="place_order_yes", log=log)

            no_order = _extract_order_payload(
                client.place_order(
                    ticker=ticker,
                    side="no",
                    action="buy",
                    count=1,
                    order_type="limit",
                    no_price=no_price,
                )
            )
            no_order.setdefault("side", "no")
            execution.no_order_id = no_order.get("order_id", "")
            pair.no_order_id = execution.no_order_id
            if not execution.no_order_id:
                raise RuntimeError("NO order placement returned no order_id")
            order_to_ticker[execution.no_order_id] = ticker
            _apply_order_snapshot_locked(execution, no_order, source="place_order_no", log=log)

            risk.record_entry(pair_cost_cents)
            traded_tickers.add(ticker)
            _finalize_execution_locked(
                execution,
                pair_tracker=pair_tracker,
                risk=risk,
                active_pairs=active_pairs,
                order_to_ticker=order_to_ticker,
                log=log,
            )

        except Exception:
            log.exception("Paired order placement failed for %s", ticker)

            for side, order_id in (("yes", execution.yes_order_id), ("no", execution.no_order_id)):
                status = execution.yes_status if side == "yes" else execution.no_status
                if not order_id or status in TERMINAL_ORDER_STATUSES:
                    continue
                try:
                    cancel_response = _extract_order_payload(client.cancel_order(order_id))
                    if not cancel_response:
                        cancel_response = {"side": side, "status": "canceled"}
                    cancel_response.setdefault("side", side)
                    _apply_order_snapshot_locked(
                        execution,
                        cancel_response,
                        source=f"placement_failure_cancel_{side}",
                        log=log,
                    )
                except Exception:
                    log.exception("Failed canceling %s order %s after placement error", side, order_id)

            if execution.pair.is_orphan:
                _resolve_orphan_locked(
                    execution,
                    reason="placement_failed",
                    pair_tracker=pair_tracker,
                    risk=risk,
                    active_pairs=active_pairs,
                    order_to_ticker=order_to_ticker,
                    log=log,
                    release_exposure=False,
                )
            else:
                execution.pair.transition("resolved", "placement_failed")
                archived = pair_tracker.active.pop(ticker, None)
                if archived is not None:
                    pair_tracker.completed.append(archived)
                _cleanup_execution(execution, active_pairs, order_to_ticker)

            return False

    log.info(
        ">>> PAIR %s (REST): %s YES@%dc id=%s + NO@%dc id=%s = %dc, +%.1fc net",
        strategy_label,
        ticker,
        yes_price,
        execution.yes_order_id,
        no_price,
        execution.no_order_id,
        pair_cost_cents,
        net_cents,
    )
    alert_trade_placed(ticker, "MAKER_PAIR", pair_cost_cents, 1, net_cents, strategy=f"crypto_pair:{strategy_label.lower()}")

    try:
        with pair_log.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{datetime.now(timezone.utc).isoformat()} {strategy_label}_REST_PAIR "
                f"{ticker} YES@{yes_price}c id={execution.yes_order_id} "
                f"NO@{no_price}c id={execution.no_order_id} cost={pair_cost_cents}c "
                f"net={net_cents:.1f}c\n"
            )
    except Exception:
        log.warning("Failed writing pair log", exc_info=True)

    return True


def _check_orphan_timeouts(
    *,
    client: KalshiClient,
    pair_tracker: PairTracker,
    risk: PairRiskManager,
    active_pairs: dict[str, ActivePairExecution],
    order_to_ticker: dict[str, str],
    state_lock: Lock,
    log: logging.Logger,
) -> None:
    with state_lock:
        due = [
            execution
            for execution in active_pairs.values()
            if execution.pair.is_orphan
            and execution.unfilled_order_id()
            and execution.unfilled_leg_status() not in TERMINAL_ORDER_STATUSES
            and risk.should_unwind_orphan(
                execution.orphan_age_s,
                execution.unfilled_price_cents(),
                execution.unfilled_price_cents(),
            )[0]
        ]

    for candidate in due:
        with state_lock:
            execution = active_pairs.get(candidate.ticker)
            if execution is None or not execution.pair.is_orphan:
                continue

            should_cancel, reason = risk.should_unwind_orphan(
                execution.orphan_age_s,
                execution.unfilled_price_cents(),
                execution.unfilled_price_cents(),
            )
            if not should_cancel:
                continue

            cancel_side = execution.unfilled_side()
            cancel_order_id = execution.unfilled_order_id()
            if cancel_side not in {"yes", "no"} or not cancel_order_id:
                continue

            execution.cancel_requested = True
            execution.cancel_reason = reason

        try:
            log.warning(
                "Orphan timeout %s: canceling %s order %s (%s)",
                candidate.ticker,
                cancel_side.upper(),
                cancel_order_id,
                reason,
            )
            cancel_response = _extract_order_payload(client.cancel_order(cancel_order_id))
        except Exception:
            log.exception(
                "Failed canceling orphan %s %s order %s",
                candidate.ticker,
                cancel_side.upper(),
                cancel_order_id,
            )
            continue

        with state_lock:
            execution = active_pairs.get(candidate.ticker)
            if execution is None:
                continue

            if not cancel_response:
                cancel_response = {"side": cancel_side, "status": "canceled"}
            cancel_response.setdefault("side", cancel_side)
            _apply_order_snapshot_locked(execution, cancel_response, source="cancel_orphan", log=log)
            _finalize_execution_locked(
                execution,
                pair_tracker=pair_tracker,
                risk=risk,
                active_pairs=active_pairs,
                order_to_ticker=order_to_ticker,
                log=log,
            )


def _run_scan(
    *,
    args: argparse.Namespace,
    scan_count: int,
    client: KalshiClient,
    tracker_mkt: KXBTCMarketTracker,
    pair_tracker: PairTracker,
    risk: PairRiskManager,
    ws_client: KalshiWebSocketClient | None,
    traded_tickers: set[str],
    active_pairs: dict[str, ActivePairExecution],
    order_to_ticker: dict[str, str],
    state_lock: Lock,
    pair_log: Path,
    log: logging.Logger,
) -> dict:
    market = tracker_mkt.get_next_market()
    if not market:
        return {"ticker": "", "remaining_s": 0, "market_available": False}

    ticker = market["ticker"]
    remaining = tracker_mkt.get_market_time_remaining(market)

    if ticker in traded_tickers:
        return {"ticker": ticker, "remaining_s": int(remaining), "already_traded": True}

    if remaining < 60 or remaining > 780:
        return {"ticker": ticker, "remaining_s": int(remaining), "skipped_window": True}

    try:
        orderbook = client.get_orderbook(ticker, depth=5)
    except Exception:
        log.warning("Failed fetching orderbook for %s", ticker, exc_info=True)
        return {"ticker": ticker, "remaining_s": int(remaining), "orderbook_error": True}

    opp = evaluate_pair_opportunity(orderbook, pair_cap_cents=args.pair_cap)
    state_update = {
        "ticker": ticker,
        "remaining_s": int(remaining),
        "best_yes_bid": opp.get("best_yes_bid", 0),
        "best_no_bid": opp.get("best_no_bid", 0),
        "implied_yes_ask": opp.get("implied_yes_ask", 0),
        "implied_no_ask": opp.get("implied_no_ask", 0),
        "taker_pair_cost": opp.get("taker_pair_cost", 0),
        "taker_arb": opp.get("taker_arb", False),
        "maker_yes": opp.get("maker_yes_price", 0),
        "maker_no": opp.get("maker_no_price", 0),
        "maker_cost": opp.get("maker_pair_cost", 0),
        "maker_net": opp.get("maker_net", 0),
        "maker_tradeable": opp.get("maker_tradeable", False),
        "spread": opp.get("spread_cents", 0),
    }

    if not opp["maker_tradeable"]:
        if scan_count % 10 == 0:
            log.info(
                "Scan #%d %s: taker=%dc(no arb) maker=%dc(net=%.1fc) spread=%dc",
                scan_count,
                ticker,
                opp["taker_pair_cost"],
                opp["maker_pair_cost"],
                opp["maker_net"],
                opp["spread_cents"],
            )
        return state_update

    with state_lock:
        active_count = len(active_pairs)

    if active_count >= risk.max_active_pairs:
        log.info("Risk blocked: active_pairs %d >= max_active_pairs %d", active_count, risk.max_active_pairs)
        state_update["risk_blocked"] = "max_active_pairs"
        return state_update

    can_open, reason = risk.can_open_pair(opp["maker_pair_cost"])
    if not can_open:
        log.info("Risk blocked: %s", reason)
        state_update["risk_blocked"] = reason
        return state_update

    yes_price = opp["maker_yes_price"]
    no_price = opp["maker_no_price"]
    pair_cost_cents = opp["maker_pair_cost"]
    net_cents = opp["maker_net"]

    if args.mode == "sim":
        pair = pair_tracker.start_pair(ticker, yes_price, no_price, market.get("close_time", ""))
        pair.record_yes_fill(yes_price, 1)
        pair.record_no_fill(no_price, 1)
        pair_tracker.complete_pair(ticker)
        risk.record_entry(pair_cost_cents)
        risk.record_pair_complete(pair_cost_cents)
        traded_tickers.add(ticker)

        log.info(
            ">>> PAIR SIM (MAKER): %s YES@%dc + NO@%dc = %dc cost, +%.1fc net "
            "[bids: Y%dc N%dc | asks: Y%dc N%dc | spread:%dc]",
            ticker,
            yes_price,
            no_price,
            pair_cost_cents,
            net_cents,
            opp["best_yes_bid"],
            opp["best_no_bid"],
            opp["implied_yes_ask"],
            opp["implied_no_ask"],
            opp["spread_cents"],
        )

        try:
            with pair_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{datetime.now(timezone.utc).isoformat()} SIM_MAKER_PAIR "
                    f"{ticker} YES@{yes_price}c NO@{no_price}c cost={pair_cost_cents}c "
                    f"gross={100 - pair_cost_cents}c net={net_cents:.1f}c "
                    f"spread={opp['spread_cents']}c\n"
                )
        except Exception:
            log.warning("Failed writing pair log", exc_info=True)

        alert_trade_placed(ticker, "MAKER_PAIR", pair_cost_cents, 1, net_cents, strategy="crypto_pair:sim")
        return state_update

    if ws_client is None or not ws_client.is_connected:
        log.warning("Skipping paired order placement on %s because Kalshi WS is not connected", ticker)
        state_update["risk_blocked"] = "ws_not_connected"
        return state_update

    # Phase 1b: pre-trade gate (covers both YES + NO legs of the pair).
    try:
        _gate_ctx = GateContext(
            ticker=ticker,
            family="btc_15m",
            side="yes",
            entry_cents=int(yes_price),
            contracts=1,
            yes_ask=float(yes_price) / 100.0,
            no_ask=float(no_price) / 100.0,
            model_prob=0.5,
            quote_age_s=0.0,
            max_stale_s=60.0,
            session_tag="any",
            strategy_tag="pair",
            calibration_artifact=None,
        )
        _gate_decision = _pre_trade_gate.evaluate(_gate_ctx)
    except Exception:
        log.exception("pre_trade_gate eval error — proceeding")
        _gate_decision = None
    if _gate_decision is not None:
        _gate_state["last"] = _gate_decision.to_json()
        if not _gate_decision.allowed:
            log.info(
                "GATE BLOCK [%s] %s: %s",
                ticker, _gate_decision.reason_code, _gate_decision.reason_detail,
            )
            state_update["risk_blocked"] = f"gate:{_gate_decision.reason_code}"
            return state_update

    _place_real_pair(
        client=client,
        ticker=ticker,
        yes_price=yes_price,
        no_price=no_price,
        pair_cost_cents=pair_cost_cents,
        net_cents=net_cents,
        close_time=market.get("close_time", ""),
        strategy_label=args.mode.upper(),
        traded_tickers=traded_tickers,
        pair_tracker=pair_tracker,
        risk=risk,
        active_pairs=active_pairs,
        order_to_ticker=order_to_ticker,
        state_lock=state_lock,
        pair_log=pair_log,
        log=log,
    )
    return state_update


def main() -> int:
    args = build_parser().parse_args()

    process_lock = ProcessLock("crypto_pair_trader")
    process_lock.kill_existing()
    if not process_lock.acquire():
        print("Another pair trader is running. Exiting.")
        return 1

    log = configure_logging(args.mode)

    data_dir = Path(settings.DATA_DIR)
    log_dir = data_dir / "logs"
    state_file = data_dir / "pair_trader_state.json"
    pair_log = log_dir / "pair_trades.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_file.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  Crypto Pair Trader: %s mode", args.mode.upper())
    log.info("  Pair cap: %dc | Scan: %ds", args.pair_cap, args.interval)
    log.info("=" * 60)

    client = KalshiClient()
    tracker_mkt = KXBTCMarketTracker(client)
    pair_tracker = PairTracker()
    risk = PairRiskManager(pair_cap_cents=args.pair_cap, budget_cents=2500)

    active_pairs: dict[str, ActivePairExecution] = {}
    order_to_ticker: dict[str, str] = {}
    traded_tickers: set[str] = set()
    state_lock = Lock()

    ws_client: KalshiWebSocketClient | None = None

    def on_fill(fill: dict) -> None:
        order_id = fill.get("order_id", "")
        if not order_id:
            return

        with state_lock:
            ticker = order_to_ticker.get(order_id)
            execution = active_pairs.get(ticker, None) if ticker else None
            if execution is None:
                return

            side = fill.get("side", "")
            _record_leg_fill_locked(
                execution,
                side,
                fill.get("price_cents", 0),
                fill.get("count", 1),
                source="ws_fill",
                log=log,
            )
            _finalize_execution_locked(
                execution,
                pair_tracker=pair_tracker,
                risk=risk,
                active_pairs=active_pairs,
                order_to_ticker=order_to_ticker,
                log=log,
            )

    def on_user_order(update: dict) -> None:
        order_id = update.get("order_id", "")
        if not order_id:
            return

        with state_lock:
            ticker = order_to_ticker.get(order_id)
            execution = active_pairs.get(ticker, None) if ticker else None
            if execution is None:
                return

            side = update.get("side", "")
            status = update.get("status", "")
            if side in {"yes", "no"} and status:
                _set_leg_status(execution, side, status)

            if side in {"yes", "no"} and update.get("fill_count", 0) > 0:
                _record_leg_fill_locked(
                    execution,
                    side,
                    update.get("price_cents", 0),
                    update["fill_count"],
                    source="ws_user_order",
                    log=log,
                )

            _finalize_execution_locked(
                execution,
                pair_tracker=pair_tracker,
                risk=risk,
                active_pairs=active_pairs,
                order_to_ticker=order_to_ticker,
                log=log,
            )

    if args.mode in {"paper", "live"}:
        ws_client = KalshiWebSocketClient(on_fill=on_fill, on_user_order=on_user_order)
        ws_client.start_in_background()
        if not ws_client.wait_until_connected(10.0):
            log.warning("Kalshi WS did not connect within startup window; trading will wait for reconnect")

    next_scan_at = time.monotonic()
    scan_count = 0
    latest_scan: dict = {}

    try:
        while True:
            if ws_client is not None:
                _check_orphan_timeouts(
                    client=client,
                    pair_tracker=pair_tracker,
                    risk=risk,
                    active_pairs=active_pairs,
                    order_to_ticker=order_to_ticker,
                    state_lock=state_lock,
                    log=log,
                )

            now = time.monotonic()
            if now >= next_scan_at:
                scan_count += 1
                latest_scan = _run_scan(
                    args=args,
                    scan_count=scan_count,
                    client=client,
                    tracker_mkt=tracker_mkt,
                    pair_tracker=pair_tracker,
                    risk=risk,
                    ws_client=ws_client,
                    traded_tickers=traded_tickers,
                    active_pairs=active_pairs,
                    order_to_ticker=order_to_ticker,
                    state_lock=state_lock,
                    pair_log=pair_log,
                    log=log,
                )
                next_scan_at = now + args.interval

            state_payload = _build_state_payload(
                mode=args.mode,
                scan_count=scan_count,
                latest_scan=latest_scan,
                pair_tracker=pair_tracker,
                risk=risk,
                active_pairs=active_pairs,
                ws_client=ws_client,
                state_lock=state_lock,
            )
            try:
                state_file.write_text(json.dumps(state_payload, indent=2, default=str), encoding="utf-8")
            except Exception:
                log.debug("Failed writing state file", exc_info=True)

            time.sleep(HEARTBEAT_S)

    except KeyboardInterrupt:
        log.info("Pair trader stopped. Stats: %s", pair_tracker.get_stats())
        return 0
    except Exception:
        log.error("Pair trader error", exc_info=True)
        return 1
    finally:
        if ws_client is not None:
            ws_client.stop()
        process_lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
