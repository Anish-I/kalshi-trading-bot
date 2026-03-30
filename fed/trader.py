"""Fed rate event trader — shadow/paper/live modes.

Vertical slice: scans KXFED markets, evaluates edge against simple
rate model, places shadow or live trades per Codex's acceptance gates.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from kalshi.client import KalshiClient
from engine.alerts import alert_trade_placed, alert_settlement
from engine.family_limits import FamilyLimits
from engine.quote_guard import check_quote_quality
from engine.order_ledger import OrderLedger, OrderRecord
from .data_source import get_fed_signal
from .signal import evaluate_fed_markets

logger = logging.getLogger(__name__)

FAMILY_NAME = "fed_cut"
STATE_FILE = Path("D:/kalshi-data/fed_trader_state.json")
SHADOW_LOG = Path("D:/kalshi-data/logs/fed_shadow.log")


class FedTrader:
    """Event-driven Fed rate trader.

    Modes:
    - shadow: evaluate markets, log what we'd trade, no orders
    - paper: place simulated orders, track P&L
    - live: real orders (only after shadow gate passes)
    """

    def __init__(self, mode: str = "shadow"):
        self.mode = mode  # shadow, paper, live
        self.client = KalshiClient()
        self.ledger = OrderLedger()
        self.family_limits = FamilyLimits()
        self._traded_tickers_path = Path("D:/kalshi-data/fed_traded_tickers.json")
        self._traded_tickers: set[str] = set()
        self._load_traded_tickers()
        SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)

    def _load_traded_tickers(self) -> None:
        if self._traded_tickers_path.exists():
            try:
                self._traded_tickers = set(json.loads(self._traded_tickers_path.read_text()))
            except Exception:
                pass

    def _save_traded_tickers(self) -> None:
        self._traded_tickers_path.write_text(json.dumps(list(self._traded_tickers)))

    def scan(self) -> dict:
        """Run one scan cycle. Returns state dict."""
        ts = datetime.now(timezone.utc)

        # 1. Get Fed signal
        signal = get_fed_signal()
        logger.info("Fed signal: rate=%.2f%% next_fomc=%s days=%s source=%s",
                     signal["current_rate"], signal["next_fomc"],
                     signal["days_to_fomc"], signal["rate_source"])

        # 2. Fetch Kalshi Fed markets
        try:
            data = self.client._request(
                "GET", "/markets",
                params={"series_ticker": "KXFED", "status": "open", "limit": 50},
            )
            markets = data.get("markets", [])
        except Exception:
            logger.error("Failed to fetch KXFED markets", exc_info=True)
            return {"error": "fetch_failed", "signal": signal}

        # 3. Evaluate opportunities
        opportunities = evaluate_fed_markets(markets, signal)
        logger.info("Fed scan: %d markets, %d opportunities", len(markets), len(opportunities))

        # 4. Act on best opportunity
        action = "no_edge"
        trade_info = None

        if opportunities:
            best = opportunities[0]
            logger.info("Best: %s %s edge=%.1f%% threshold=%.2f%% prob=%.1f%%",
                        best["side"], best["ticker"], best["edge"] * 100,
                        best["threshold"], best["model_prob_above"] * 100)

            # Already traded this ticker?
            if best["ticker"] in self._traded_tickers:
                action = "already_traded"
            else:
                # Quote guard
                mkt = next((m for m in markets if m.get("ticker") == best["ticker"]), {})
                ya = float(mkt.get("yes_ask", mkt.get("yes_ask_dollars", 0)) or 0)
                na = float(mkt.get("no_ask", mkt.get("no_ask_dollars", 0)) or 0)
                yb = float(mkt.get("yes_bid", mkt.get("yes_bid_dollars", 0)) or 0)

                tradeable, guard_reason = check_quote_quality(
                    yes_ask=ya, no_ask=na, yes_bid=yb,
                    model_prob=best["model_prob_above"],
                    side=best["side"].lower(),
                    max_spread_cents=30,
                    min_edge_after_fees_pct=0.02,
                )

                if not tradeable:
                    action = f"quote_guard: {guard_reason}"
                    logger.info("Quote guard blocked: %s", guard_reason)
                else:
                    # Family budget check
                    entry_cents = int(ya * 100) if best["side"] == "YES" else int(na * 100)
                    can_enter, limit_reason = self.family_limits.can_enter(FAMILY_NAME, entry_cents)

                    if not can_enter:
                        action = f"family_limit: {limit_reason}"
                    else:
                        action = self._execute(best, entry_cents)
                        trade_info = best

        # 5. Build state
        state = {
            "time": ts.isoformat(),
            "mode": self.mode,
            "signal": signal,
            "markets_scanned": len(markets),
            "opportunities": len(opportunities),
            "best": opportunities[0] if opportunities else None,
            "action": action,
            "trade": trade_info,
        }

        # Save state
        try:
            STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
        except Exception:
            pass

        return state

    def _execute(self, opp: dict, entry_cents: int) -> str:
        """Execute based on mode."""
        ticker = opp["ticker"]
        side = opp["side"].lower()
        self._traded_tickers.add(ticker)
        self._save_traded_tickers()

        if self.mode == "shadow":
            # Just log
            line = (f"{datetime.now(timezone.utc).isoformat()} SHADOW "
                    f"{side.upper()} {ticker} @{entry_cents}c "
                    f"edge={opp['edge']*100:.1f}% threshold={opp['threshold']}%\n")
            with open(SHADOW_LOG, "a") as f:
                f.write(line)
            logger.info("SHADOW TRADE: %s %s @%dc edge=%.1f%%",
                        side.upper(), ticker, entry_cents, opp["edge"] * 100)
            return "shadow_trade"

        elif self.mode == "paper":
            # Log + record in ledger as simulated
            self.ledger.add(OrderRecord(
                strategy="fed_cut",
                market_type="fed_rate",
                ticker=ticker,
                status="simulated",
                submitted_side=side,
                submitted_price_cents=entry_cents,
                submitted_count=1,
                filled_price_cents=entry_cents,
                filled_count=1,
                signal_ref=f"fed:threshold:{opp['threshold']}",
            ))
            self.ledger.save()
            self.family_limits.record_entry(FAMILY_NAME, entry_cents)
            alert_trade_placed(ticker, side, entry_cents, 1, opp["edge"] * 100, strategy="fed_paper")
            logger.info("PAPER TRADE: %s %s @%dc", side.upper(), ticker, entry_cents)
            return "paper_trade"

        elif self.mode == "live":
            # Real order — 1 contract max per Codex's gate
            try:
                yes_price = entry_cents if side == "yes" else None
                no_price = entry_cents if side == "no" else None
                order_resp = self.client.place_order(
                    ticker=ticker, side=side, action="buy",
                    count=1, order_type="limit",
                    yes_price=yes_price, no_price=no_price,
                )
                order_data = order_resp.get("order", order_resp)
                status = order_data.get("status", "")
                order_id = order_data.get("order_id", "")

                self.ledger.add(OrderRecord(
                    strategy="fed_cut",
                    market_type="fed_rate",
                    ticker=ticker,
                    order_id=order_id,
                    status="filled" if status == "executed" else status,
                    submitted_side=side,
                    submitted_price_cents=entry_cents,
                    submitted_count=1,
                    filled_price_cents=entry_cents,
                    filled_count=1 if status == "executed" else 0,
                    signal_ref=f"fed:threshold:{opp['threshold']}",
                ))
                self.ledger.save()
                self.family_limits.record_entry(FAMILY_NAME, entry_cents)
                alert_trade_placed(ticker, side, entry_cents, 1, opp["edge"] * 100, strategy="fed_live")
                logger.info("LIVE TRADE: %s %s @%dc order=%s status=%s",
                            side.upper(), ticker, entry_cents, order_id, status)
                return "live_trade"
            except Exception:
                logger.error("Failed to place Fed order", exc_info=True)
                return "order_failed"

        return "unknown_mode"
