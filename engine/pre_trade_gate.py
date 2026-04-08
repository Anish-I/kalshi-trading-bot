"""Unified pre-trade gate composing risk, family, quote, session, and calibration checks.

Phase 1 of the unified gate plan. Order entry sites in the traders are NOT yet
funneled through this module — that lands in phase 1b. This file ships the gate
+ tests so the composition is covered before any caller is rewired.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable

from engine.quote_guard import check_quote_quality
from engine.crypto_decision import evaluate_calibrated_trade


# Default sessions allowed if caller doesn't override.
DEFAULT_ALLOWED_SESSIONS: frozenset[str] = frozenset(
    {"us_core", "us_open", "us_pm", "eu_core", "asia_core", "overnight", "any"}
)


@dataclass
class GateContext:
    ticker: str
    family: str
    side: str  # "yes" | "no"
    entry_cents: int
    contracts: int
    yes_ask: float
    no_ask: float
    model_prob: float
    quote_age_s: float
    max_stale_s: float
    session_tag: str
    strategy_tag: str  # "ml" | "pair" | "combined_ml" | "combined_pair"
    calibration_artifact: dict | None = None
    # Optional knobs forwarded to evaluate_calibrated_trade
    min_trades: int = 30
    ev_buffer_cents: float = 0.0
    min_net_ev_cents: float = 0.0
    # Optional quote-quality knobs
    yes_bid: float = 0.0
    no_bid: float = 0.0
    max_spread_cents: int = 20
    min_edge_after_fees_pct: float = 0.02


@dataclass
class GateDecision:
    allowed: bool
    reason_code: str
    reason_detail: str
    contracts_adjusted: int
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason_code": self.reason_code,
            "reason_detail": self.reason_detail,
            "contracts_adjusted": self.contracts_adjusted,
            "checks": self.checks,
        }


class PreTradeGate:
    """Compose pre-trade primitives into a single decision.

    Order:
        1. risk_mgr.can_trade()              -> "risk_halt"
        2. risk_mgr.check_order_size(n)      -> "order_size"
        3. family_limits.can_enter(...)      -> "family_budget"
        4. check_quote_quality(...)          -> "quote_quality"
        5. session gating                    -> "session_blocked"
        6. evaluate_calibrated_trade(...)    -> "bucket_not_tradable" | "negative_ev"
        7. (phase 2) family_scorecard hook   -> "family_unhealthy"
    """

    def __init__(
        self,
        risk_mgr,
        family_limits,
        calibration_loader: Callable[[GateContext], dict | None] | None = None,
        allowed_sessions: Iterable[str] | None = None,
        scorecard_hook: Callable[[GateContext], tuple[bool, str, dict]] | None = None,
    ):
        self.risk_mgr = risk_mgr
        self.family_limits = family_limits
        self.calibration_loader = calibration_loader
        self.allowed_sessions = (
            frozenset(allowed_sessions) if allowed_sessions is not None else DEFAULT_ALLOWED_SESSIONS
        )
        self.scorecard_hook = scorecard_hook

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _record(checks: dict, name: str, passed: bool, detail: str, values: dict) -> None:
        checks[name] = {"passed": passed, "detail": detail, "values": values}

    def _block(
        self,
        checks: dict,
        reason_code: str,
        reason_detail: str,
    ) -> GateDecision:
        return GateDecision(
            allowed=False,
            reason_code=reason_code,
            reason_detail=reason_detail,
            contracts_adjusted=0,
            checks=checks,
        )

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def evaluate(self, ctx: GateContext) -> GateDecision:
        checks: dict[str, dict[str, Any]] = {}

        # 1) Risk halt
        ok, detail = self.risk_mgr.can_trade()
        self._record(checks, "risk_halt", ok, detail, {})
        if not ok:
            return self._block(checks, "risk_halt", detail)

        # 2) Order size
        ok, detail = self.risk_mgr.check_order_size(ctx.contracts)
        self._record(
            checks,
            "order_size",
            ok,
            detail,
            {"contracts": ctx.contracts},
        )
        if not ok:
            return self._block(checks, "order_size", detail)

        # 3) Family budget
        proposed_cents = int(ctx.entry_cents) * int(ctx.contracts)
        ok, detail = self.family_limits.can_enter(ctx.family, proposed_cents)
        self._record(
            checks,
            "family_budget",
            ok,
            detail,
            {"family": ctx.family, "proposed_cents": proposed_cents},
        )
        if not ok:
            return self._block(checks, "family_budget", detail)

        # 4) Quote quality
        ok, detail = check_quote_quality(
            yes_ask=ctx.yes_ask,
            no_ask=ctx.no_ask,
            yes_bid=ctx.yes_bid,
            no_bid=ctx.no_bid,
            max_spread_cents=ctx.max_spread_cents,
            min_edge_after_fees_pct=ctx.min_edge_after_fees_pct,
            model_prob=ctx.model_prob,
            side=ctx.side,
            quote_age_s=ctx.quote_age_s,
            max_stale_s=ctx.max_stale_s,
        )
        self._record(
            checks,
            "quote_quality",
            ok,
            detail,
            {
                "yes_ask": ctx.yes_ask,
                "no_ask": ctx.no_ask,
                "quote_age_s": ctx.quote_age_s,
                "max_stale_s": ctx.max_stale_s,
                "model_prob": ctx.model_prob,
                "side": ctx.side,
            },
        )
        if not ok:
            return self._block(checks, "quote_quality", detail)

        # 5) Session gating
        session_ok = ctx.session_tag in self.allowed_sessions
        session_detail = (
            "ok"
            if session_ok
            else f"session '{ctx.session_tag}' not in allowed set"
        )
        self._record(
            checks,
            "session",
            session_ok,
            session_detail,
            {"session_tag": ctx.session_tag, "allowed": sorted(self.allowed_sessions)},
        )
        if not session_ok:
            return self._block(checks, "session_blocked", session_detail)

        # 6) Calibration / EV
        artifact = ctx.calibration_artifact
        if artifact is None and self.calibration_loader is not None:
            artifact = self.calibration_loader(ctx)

        if artifact is None:
            # No calibration provided — record skip and continue. Treat as pass.
            self._record(
                checks,
                "calibration",
                True,
                "no calibration artifact provided (skipped)",
                {},
            )
        else:
            cal = evaluate_calibrated_trade(
                artifact,
                ctx.side,
                int(ctx.entry_cents),
                min_trades=ctx.min_trades,
                ev_buffer_cents=ctx.ev_buffer_cents,
                min_net_ev_cents=ctx.min_net_ev_cents,
            )
            tradable = bool(cal.get("bucket_tradable", False))
            net_ev = cal.get("net_ev_cents_per_contract")
            cal_values = {
                "status": cal.get("status"),
                "calibrated_p_win": cal.get("calibrated_p_win"),
                "gross_ev_cents_per_contract": cal.get("gross_ev_cents_per_contract"),
                "net_ev_cents_per_contract": net_ev,
                "bucket_trade_count": cal.get("bucket_trade_count"),
                "bucket_tradable": tradable,
                "price_bucket": cal.get("price_bucket"),
            }
            if not tradable:
                self._record(
                    checks,
                    "calibration",
                    False,
                    f"bucket not tradable (status={cal.get('status')})",
                    cal_values,
                )
                return self._block(
                    checks,
                    "bucket_not_tradable",
                    f"calibration bucket not tradable for {ctx.side}@{ctx.entry_cents}c",
                )
            if net_ev is None or float(net_ev) <= 0.0:
                self._record(
                    checks,
                    "calibration",
                    False,
                    f"net EV {net_ev} <= 0",
                    cal_values,
                )
                return self._block(
                    checks,
                    "negative_ev",
                    f"net EV {net_ev} cents/contract is non-positive",
                )
            self._record(checks, "calibration", True, "ok", cal_values)

        # 7) Family scorecard (phase 2 placeholder)
        if self.scorecard_hook is not None:
            ok, detail, values = self.scorecard_hook(ctx)
            self._record(checks, "family_scorecard", ok, detail, values)
            if not ok:
                return self._block(checks, "family_unhealthy", detail)
        else:
            self._record(
                checks,
                "family_scorecard",
                True,
                "scorecard hook not wired (phase 2)",
                {},
            )

        return GateDecision(
            allowed=True,
            reason_code="ok",
            reason_detail="all checks passed",
            contracts_adjusted=int(ctx.contracts),
            checks=checks,
        )
