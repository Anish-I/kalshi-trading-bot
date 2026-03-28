"""
Shared order lifecycle ledger. Both crypto and weather traders write here.
Append-only parquet at D:/kalshi-data/order_ledger.parquet

Status flow: pending -> resting -> filled -> settled
             pending -> cancelled/rejected
"""
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OrderRecord:
    strategy: str  # "crypto" or "weather"
    market_type: str  # "btc_15m", "weather_high", "weather_low"
    ticker: str
    order_id: str = ""
    status: str = "pending"  # pending/resting/filled/cancelled/rejected/settled/simulated
    submitted_side: str = ""  # "yes" or "no"
    submitted_price_cents: int = 0
    submitted_count: int = 0
    filled_price_cents: int = 0
    filled_count: int = 0
    settlement_result: str = ""  # "yes" or "no"
    pnl_cents: int = 0
    created_at: str = ""
    updated_at: str = ""
    session_tag: str = ""  # "us_core", "us_off", "weekend"
    signal_ref: str = ""  # vote summary or forecast ref
    # Execution metrics (Phase 1C)
    slippage_cents: int = 0  # filled_price - submitted_price
    time_to_fill_ms: int = 0  # submission to fill latency


class OrderLedger:
    def __init__(self, data_dir: str = "D:/kalshi-data"):
        self.path = Path(data_dir) / "order_ledger.parquet"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict] = []
        # Load existing
        if self.path.exists():
            try:
                df = pd.read_parquet(self.path)
                self._records = df.to_dict("records")
                logger.info("Loaded %d ledger records", len(self._records))
            except Exception:
                logger.warning("Failed to load ledger, starting fresh")

    def add(self, record: OrderRecord) -> None:
        if not record.created_at:
            record.created_at = datetime.now(timezone.utc).isoformat()
        record.updated_at = datetime.now(timezone.utc).isoformat()
        self._records.append(asdict(record))

    def update_status(self, ticker: str, new_status: str, **kwargs) -> None:
        """Update the most recent record for this ticker."""
        for rec in reversed(self._records):
            if rec["ticker"] == ticker and rec["status"] not in ("settled", "cancelled", "rejected"):
                rec["status"] = new_status
                rec["updated_at"] = datetime.now(timezone.utc).isoformat()
                for k, v in kwargs.items():
                    if k in rec:
                        rec[k] = v
                break

    def settle(self, ticker: str, result: str, pnl_cents: int) -> None:
        """Mark a filled order as settled."""
        for rec in reversed(self._records):
            if rec["ticker"] == ticker and rec["status"] == "filled":
                rec["status"] = "settled"
                rec["settlement_result"] = result
                rec["pnl_cents"] = pnl_cents
                rec["updated_at"] = datetime.now(timezone.utc).isoformat()
                break

    def save(self) -> None:
        if not self._records:
            return
        try:
            tmp = self.path.with_suffix(".tmp")
            pd.DataFrame(self._records).to_parquet(tmp, index=False)
            tmp.replace(self.path)
        except Exception:
            logger.exception("Failed to save ledger")

    def get_open_orders(self, strategy: str = None) -> list[dict]:
        """Get orders that are pending or resting."""
        return [r for r in self._records
                if r["status"] in ("pending", "resting")
                and (strategy is None or r["strategy"] == strategy)]

    def get_filled_unsettled(self, strategy: str = None) -> list[dict]:
        return [r for r in self._records
                if r["status"] == "filled"
                and (strategy is None or r["strategy"] == strategy)]

    def get_recent(self, n: int = 20) -> list[dict]:
        return self._records[-n:]

    def get_stats(self, strategy: str = None) -> dict:
        settled = [r for r in self._records
                   if r["status"] == "settled"
                   and (strategy is None or r["strategy"] == strategy)]
        if not settled:
            return {"trades": 0, "wins": 0, "losses": 0, "pnl": 0}
        wins = sum(1 for r in settled if r["pnl_cents"] > 0)
        pnl = sum(r["pnl_cents"] for r in settled)
        return {"trades": len(settled), "wins": wins, "losses": len(settled) - wins, "pnl": pnl}

    def get_execution_stats(self, strategy: str = None) -> dict:
        """Compute execution quality metrics."""
        all_recs = [r for r in self._records
                    if strategy is None or r["strategy"] == strategy]
        if not all_recs:
            return {"total_orders": 0}

        filled = [r for r in all_recs if r["status"] in ("filled", "settled", "simulated")]
        cancelled = [r for r in all_recs if r["status"] == "cancelled"]
        expired = [r for r in all_recs if r["status"] == "expired"]

        slippages = [r.get("slippage_cents", 0) for r in filled if r.get("slippage_cents", 0) != 0]
        fill_times = [r.get("time_to_fill_ms", 0) for r in filled if r.get("time_to_fill_ms", 0) > 0]

        total = len(all_recs)
        return {
            "total_orders": total,
            "filled_count": len(filled),
            "cancelled_count": len(cancelled),
            "expired_count": len(expired),
            "fill_ratio": len(filled) / total if total else 0,
            "cancel_ratio": len(cancelled) / total if total else 0,
            "missed_fill_count": len(expired),
            "slippage_mean": sum(slippages) / len(slippages) if slippages else 0,
            "slippage_worst": max(slippages, default=0),
            "fill_time_mean_ms": sum(fill_times) / len(fill_times) if fill_times else 0,
            "fill_time_worst_ms": max(fill_times, default=0),
        }
