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
    status: str = "pending"  # pending/resting/filled/cancelled/rejected/settled
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
