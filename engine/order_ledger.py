"""
Shared order lifecycle ledger. Multiple processes write concurrently.
Append-only parquet at D:/kalshi-data/order_ledger.parquet

Concurrency: lock file + reload-before-write. Every save() re-reads from
disk, merges local pending records, then writes atomically with a lock.
Read methods call _refresh() to get latest from all processes.

Status flow: pending -> resting -> filled -> settled
             pending -> cancelled/rejected
"""
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

LOCK_TIMEOUT_S = 5.0
LOCK_RETRY_S = 0.1


@dataclass
class OrderRecord:
    strategy: str  # "crypto", "weather", "fed_cut", "crypto_sim"
    market_type: str  # "btc_15m", "weather_high", "fed_rate"
    ticker: str
    order_id: str = ""
    status: str = "pending"
    submitted_side: str = ""
    submitted_price_cents: int = 0
    submitted_count: int = 0
    filled_price_cents: int = 0
    filled_count: int = 0
    settlement_result: str = ""
    pnl_cents: int = 0
    created_at: str = ""
    updated_at: str = ""
    session_tag: str = ""
    signal_ref: str = ""
    slippage_cents: int = 0
    time_to_fill_ms: int = 0


class OrderLedger:
    def __init__(self, data_dir: str = "D:/kalshi-data"):
        self.path = Path(data_dir) / "order_ledger.parquet"
        self._lock_path = self.path.with_suffix(".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict] = []
        self._pending: list[dict] = []  # new records not yet on disk
        self._load()

    def _load(self) -> None:
        """Load existing records from disk."""
        if self.path.exists():
            try:
                df = pd.read_parquet(self.path)
                self._records = df.to_dict("records")
                logger.info("Loaded %d ledger records", len(self._records))
            except Exception:
                logger.warning("Failed to load ledger, starting fresh")

    def _refresh(self) -> None:
        """Re-read from disk to pick up writes from other processes."""
        if self.path.exists():
            try:
                df = pd.read_parquet(self.path)
                self._records = df.to_dict("records")
            except Exception:
                pass

    def _acquire_lock(self) -> bool:
        """Acquire a file-based lock. Returns True if acquired."""
        start = time.monotonic()
        while time.monotonic() - start < LOCK_TIMEOUT_S:
            try:
                # Create lock file exclusively (fails if exists)
                fd = self._lock_path.open("x")
                fd.write(str(time.time()))
                fd.close()
                return True
            except FileExistsError:
                # Check for stale lock (older than 30s)
                try:
                    age = time.time() - self._lock_path.stat().st_mtime
                    if age > 30:
                        self._lock_path.unlink(missing_ok=True)
                        continue
                except Exception:
                    pass
                time.sleep(LOCK_RETRY_S)
        logger.warning("Failed to acquire ledger lock after %.1fs", LOCK_TIMEOUT_S)
        return False

    def _release_lock(self) -> None:
        """Release the file lock."""
        self._lock_path.unlink(missing_ok=True)

    def add(self, record: OrderRecord) -> None:
        if not record.created_at:
            record.created_at = datetime.now(timezone.utc).isoformat()
        record.updated_at = datetime.now(timezone.utc).isoformat()
        d = asdict(record)
        self._records.append(d)
        self._pending.append(d)

    def update_status(self, ticker: str, new_status: str, **kwargs) -> None:
        """Update the most recent record for this ticker."""
        self._refresh()
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
        self._refresh()
        for rec in reversed(self._records):
            if rec["ticker"] == ticker and rec["status"] in ("filled", "simulated"):
                rec["status"] = "settled"
                rec["settlement_result"] = result
                rec["pnl_cents"] = pnl_cents
                rec["updated_at"] = datetime.now(timezone.utc).isoformat()
                break

    def save(self) -> None:
        """Save with lock: re-read disk, merge pending, write atomically."""
        if not self._pending and not self._records:
            return

        if not self._acquire_lock():
            # Fallback: try without lock (better than losing data)
            logger.warning("Saving ledger without lock (fallback)")

        try:
            # Re-read from disk to get other processes' writes
            disk_records = []
            if self.path.exists():
                try:
                    df = pd.read_parquet(self.path)
                    disk_records = df.to_dict("records")
                except Exception:
                    disk_records = []

            # Merge: disk records + our pending (new) records
            if self._pending:
                merged = disk_records + self._pending
            else:
                merged = disk_records

            # Also apply any in-memory status updates (settle/update_status)
            # by matching on ticker+created_at and taking our version
            updated_keys = set()
            for rec in self._records:
                if rec.get("status") in ("settled",) and rec.get("updated_at"):
                    key = (rec.get("ticker"), rec.get("created_at"))
                    updated_keys.add(key)

            if updated_keys:
                local_by_key = {}
                for rec in self._records:
                    key = (rec.get("ticker"), rec.get("created_at"))
                    if key in updated_keys:
                        local_by_key[key] = rec

                final = []
                for rec in merged:
                    key = (rec.get("ticker"), rec.get("created_at"))
                    if key in local_by_key:
                        final.append(local_by_key.pop(key))
                    else:
                        final.append(rec)
                # Any remaining local records not on disk
                for rec in local_by_key.values():
                    final.append(rec)
                merged = final

            # Atomic write
            tmp = self.path.with_suffix(".tmp")
            pd.DataFrame(merged).to_parquet(tmp, index=False)
            tmp.replace(self.path)

            # Clear pending
            self._pending.clear()
            self._records = merged

        except Exception:
            logger.exception("Failed to save ledger")
        finally:
            self._release_lock()

    # --- Read methods (refresh from disk first) ---

    def get_open_orders(self, strategy: str = None) -> list[dict]:
        self._refresh()
        return [r for r in self._records
                if r["status"] in ("pending", "resting")
                and (strategy is None or r["strategy"] == strategy)]

    def get_filled_unsettled(self, strategy: str = None) -> list[dict]:
        self._refresh()
        return [r for r in self._records
                if r["status"] == "filled"
                and (strategy is None or r["strategy"] == strategy)]

    def get_recent(self, n: int = 20) -> list[dict]:
        self._refresh()
        return self._records[-n:]

    def get_stats(self, strategy: str = None) -> dict:
        self._refresh()
        settled = [r for r in self._records
                   if r["status"] == "settled"
                   and (strategy is None or r["strategy"] == strategy)]
        if not settled:
            return {"trades": 0, "wins": 0, "losses": 0, "pnl": 0}
        wins = sum(1 for r in settled if r["pnl_cents"] > 0)
        pnl = sum(r["pnl_cents"] for r in settled)
        return {"trades": len(settled), "wins": wins, "losses": len(settled) - wins, "pnl": pnl}

    def get_execution_stats(self, strategy: str = None) -> dict:
        self._refresh()
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
