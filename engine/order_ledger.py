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
import math
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
        self._local_mutations: dict[tuple[str, str], dict] = {}
        self._load()

    @staticmethod
    def _record_key(record: dict) -> tuple[str, str]:
        return (str(record.get("ticker") or ""), str(record.get("created_at") or ""))

    @staticmethod
    def _is_missing(value) -> bool:
        if value is None:
            return True
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    @classmethod
    def _to_int(cls, value, default: int = 0) -> int:
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if cls._is_missing(value):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _to_float(cls, value, default: float = 0.0) -> float:
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if cls._is_missing(value):
            return default
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default

    def _read_disk_records(self) -> list[dict]:
        if not self.path.exists():
            return []
        df = pd.read_parquet(self.path)
        return df.to_dict("records")

    def _track_local_mutation(self, record: dict) -> None:
        self._local_mutations[self._record_key(record)] = record

    def _merge_records(self, disk_records: list[dict]) -> list[dict]:
        if not self._pending and not self._local_mutations:
            return list(disk_records)

        merged: list[dict] = []
        seen_keys: set[tuple[str, str]] = set()

        for record in disk_records:
            key = self._record_key(record)
            merged.append(self._local_mutations.get(key, record))
            seen_keys.add(key)

        for record in self._pending:
            key = self._record_key(record)
            if key in seen_keys:
                continue
            merged.append(self._local_mutations.get(key, record))
            seen_keys.add(key)

        for key, record in self._local_mutations.items():
            if key not in seen_keys:
                merged.append(record)

        return merged

    def _load(self) -> None:
        """Load existing records from disk."""
        if self.path.exists():
            try:
                self._records = self._read_disk_records()
                logger.info("Loaded %d ledger records", len(self._records))
            except Exception:
                logger.warning("Failed to load ledger, starting fresh")

    def _refresh(self) -> None:
        """Re-read from disk to pick up writes from other processes."""
        try:
            self._records = self._merge_records(self._read_disk_records())
        except Exception:
            if self._pending or self._local_mutations:
                self._records = self._merge_records([])

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
        self._track_local_mutation(d)

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
                self._track_local_mutation(rec)
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
                self._track_local_mutation(rec)
                break

    def save(self) -> None:
        """Save with lock: re-read disk, merge pending, write atomically."""
        if not self._pending and not self._local_mutations and not self._records:
            return

        lock_acquired = self._acquire_lock()
        if not lock_acquired:
            # Fallback: try without lock (better than losing data)
            logger.warning("Saving ledger without lock (fallback)")

        try:
            # Re-read from disk to get other processes' writes
            try:
                disk_records = self._read_disk_records()
            except Exception:
                disk_records = []

            merged = self._merge_records(disk_records)

            # Atomic write
            tmp = self.path.with_suffix(".tmp")
            pd.DataFrame(merged).to_parquet(tmp, index=False)
            tmp.replace(self.path)

            # Clear pending
            self._pending.clear()
            self._local_mutations.clear()
            self._records = merged

        except Exception:
            logger.exception("Failed to save ledger")
        finally:
            if lock_acquired:
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
        pnl_values = [self._to_int(r.get("pnl_cents", 0)) for r in settled]
        wins = sum(1 for pnl in pnl_values if pnl > 0)
        pnl = sum(pnl_values)
        return {
            "trades": int(len(settled)),
            "wins": int(wins),
            "losses": int(len(settled) - wins),
            "pnl": int(pnl),
        }

    def get_execution_stats(self, strategy: str = None) -> dict:
        self._refresh()
        all_recs = [r for r in self._records
                    if strategy is None or r["strategy"] == strategy]
        if not all_recs:
            return {"total_orders": 0}

        filled = [r for r in all_recs if r["status"] in ("filled", "settled", "simulated")]
        cancelled = [r for r in all_recs if r["status"] == "cancelled"]
        expired = [r for r in all_recs if r["status"] == "expired"]

        slippages = []
        for record in filled:
            slippage = self._to_float(record.get("slippage_cents", 0))
            if slippage != 0:
                slippages.append(slippage)

        fill_times = []
        for record in filled:
            fill_time = self._to_float(record.get("time_to_fill_ms", 0))
            if fill_time > 0:
                fill_times.append(fill_time)

        total = len(all_recs)
        return {
            "total_orders": int(total),
            "filled_count": int(len(filled)),
            "cancelled_count": int(len(cancelled)),
            "expired_count": int(len(expired)),
            "fill_ratio": float(len(filled) / total) if total else 0.0,
            "cancel_ratio": float(len(cancelled) / total) if total else 0.0,
            "missed_fill_count": int(len(expired)),
            "slippage_mean": float(sum(slippages) / len(slippages)) if slippages else 0.0,
            "slippage_worst": float(max(slippages, default=0.0)),
            "fill_time_mean_ms": float(sum(fill_times) / len(fill_times)) if fill_times else 0.0,
            "fill_time_worst_ms": float(max(fill_times, default=0.0)),
        }
