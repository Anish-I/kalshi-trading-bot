"""Family performance scorecards + optional throttle.

Phase 2 of unified-gate plan. Joins OrderLedger + TradeJournal rows to
MARKET_FAMILIES by series_prefix, computes rolling health metrics per
family, and exposes an optional throttle multiplier for FamilyLimits.

Ships in shadow mode: when shadow_mode=True, throttle_multiplier is
always forced to 1.0 so the gate records health without changing
trader behavior.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from config.market_registry import MARKET_FAMILIES, PHASE4_CANDIDATES

logger = logging.getLogger(__name__)


@dataclass
class FamilyHealth:
    family: str
    healthy: bool
    score: float
    throttle_multiplier: float
    metrics: dict = field(default_factory=dict)


def _family_prefixes(registry: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, fam in registry.items():
        prefix_field = getattr(fam, "series_prefix", "") or ""
        parts = [p.strip() for p in str(prefix_field).split(",") if p.strip()]
        if parts:
            out[name] = parts
    return out


def _match_family(ticker: str, prefix_map: dict[str, list[str]]) -> Optional[str]:
    if not isinstance(ticker, str) or not ticker:
        return None
    # longest prefix wins to avoid e.g. KXBTC matching before KXBTC15M
    best: tuple[int, Optional[str]] = (0, None)
    for name, prefixes in prefix_map.items():
        for p in prefixes:
            if ticker.startswith(p) and len(p) > best[0]:
                best = (len(p), name)
    return best[1]


class FamilyScorecard:
    def __init__(
        self,
        order_ledger_path: Path = Path("D:/kalshi-data/order_ledger.parquet"),
        trade_journal_path: Path = Path("D:/kalshi-data/trade_journal.parquet"),
        market_registry: Optional[dict] = None,
        window_hours: int = 48,
        cache_path: Path = Path("D:/kalshi-data/family_scorecard.parquet"),
        shadow_mode: bool = True,
        combined_state_path: Path = Path("D:/kalshi-data/combined_trader_state.json"),
    ):
        self.order_ledger_path = Path(order_ledger_path)
        self.trade_journal_path = Path(trade_journal_path)
        if market_registry is None:
            market_registry = {**MARKET_FAMILIES, **PHASE4_CANDIDATES}
        self.registry = market_registry
        self.prefix_map = _family_prefixes(self.registry)
        self.window_hours = int(window_hours)
        self.cache_path = Path(cache_path)
        self.shadow_mode = bool(shadow_mode)
        self.combined_state_path = Path(combined_state_path)
        self._last: dict[str, FamilyHealth] = {}

    # ------------------------------------------------------------------ #
    # I/O helpers
    # ------------------------------------------------------------------ #

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(path)
        except Exception:
            logger.warning("FamilyScorecard: failed to read %s", path)
            return pd.DataFrame()

    def _read_orphan_state(self) -> dict[str, tuple[int, int]]:
        """Return {family_name: (orphans, completed)} from combined trader state."""
        out: dict[str, tuple[int, int]] = {}
        if not self.combined_state_path.exists():
            return out
        try:
            data = json.loads(self.combined_state_path.read_text())
        except Exception:
            return out
        per_series = data.get("per_series", {}) if isinstance(data, dict) else {}
        if not isinstance(per_series, dict):
            return out
        for series_key, rec in per_series.items():
            if not isinstance(rec, dict):
                continue
            fam = _match_family(str(series_key), self.prefix_map)
            if fam is None:
                continue
            orphans = int(rec.get("orphans", 0) or 0)
            completed = int(rec.get("completed", 0) or 0)
            prev = out.get(fam, (0, 0))
            out[fam] = (prev[0] + orphans, prev[1] + completed)
        return out

    # ------------------------------------------------------------------ #
    # Metric computation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_time(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, utc=True, errors="coerce")

    def _window_filter(self, df: pd.DataFrame, col: str, as_of: datetime) -> pd.DataFrame:
        if df.empty or col not in df.columns:
            return df
        ts = self._parse_time(df[col])
        cutoff = as_of - timedelta(hours=self.window_hours)
        return df[ts >= cutoff].copy()

    def _compute_family_metrics(
        self,
        family: str,
        ledger_df: pd.DataFrame,
        journal_df: pd.DataFrame,
        orphan_state: dict[str, tuple[int, int]],
    ) -> dict:
        # trade journal — settled rows for this family
        j_fam = journal_df[journal_df["_family"] == family] if not journal_df.empty else journal_df
        if not j_fam.empty and "settled" in j_fam.columns:
            settled = j_fam[j_fam["settled"] == True]  # noqa: E712
        else:
            settled = j_fam.iloc[0:0] if not j_fam.empty else j_fam

        trade_count = int(len(settled))
        if trade_count >= 5 and "won" in settled.columns:
            wins = int(settled["won"].fillna(False).astype(bool).sum())
            win_rate = wins / trade_count
        else:
            win_rate = 0.5

        if "pnl_cents" in settled.columns and trade_count > 0:
            pnl_series = pd.to_numeric(settled["pnl_cents"], errors="coerce").fillna(0)
            realized_pnl = int(pnl_series.sum())
            cum = pnl_series.cumsum()
            running_max = cum.cummax()
            drawdown = (cum - running_max).min()
            max_drawdown = int(drawdown) if pd.notna(drawdown) else 0
        else:
            realized_pnl = 0
            max_drawdown = 0

        # ledger — fill ratio
        l_fam = ledger_df[ledger_df["_family"] == family] if not ledger_df.empty else ledger_df
        if not l_fam.empty and {"submitted_count", "filled_count"}.issubset(l_fam.columns):
            submitted = pd.to_numeric(l_fam["submitted_count"], errors="coerce").fillna(0).sum()
            filled = pd.to_numeric(l_fam["filled_count"], errors="coerce").fillna(0).sum()
            fill_ratio = float(filled / submitted) if submitted > 0 else 1.0
        else:
            fill_ratio = 1.0

        # orphan rate
        orphans, completed = orphan_state.get(family, (0, 0))
        if orphans + completed > 0:
            orphan_rate = float(orphans / max(1, completed + orphans))
        else:
            orphan_rate = 0.0  # no real orphan data; assume healthy

        return {
            "trade_count": trade_count,
            "win_rate": float(win_rate),
            "realized_pnl_cents": realized_pnl,
            "max_drawdown_cents": max_drawdown,
            "fill_ratio": float(fill_ratio),
            "orphan_rate": float(orphan_rate),
        }

    @staticmethod
    def _score_and_multiplier(m: dict) -> tuple[float, float, bool]:
        score = 1.0
        if m["trade_count"] >= 5:
            if m["win_rate"] < 0.45:
                score -= 0.3
        if m["realized_pnl_cents"] < 0:
            score -= 0.2
        if m["orphan_rate"] > 0.3:
            score -= 0.3
        if m["fill_ratio"] < 0.5:
            score -= 0.1
        score = max(0.0, min(1.0, score))

        if score >= 0.9:
            mult = 1.0
        elif score >= 0.7:
            mult = 0.5
        elif score >= 0.5:
            mult = 0.25
        else:
            mult = 0.0

        healthy = score >= 0.7
        return score, mult, healthy

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def compute(self, as_of: Optional[datetime] = None) -> dict[str, FamilyHealth]:
        if as_of is None:
            as_of = datetime.now(timezone.utc)
        elif as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        ledger_df = self._read_parquet(self.order_ledger_path)
        journal_df = self._read_parquet(self.trade_journal_path)

        ledger_df = self._window_filter(ledger_df, "updated_at", as_of)
        if ledger_df.empty and "created_at" in self._read_parquet(self.order_ledger_path).columns:
            ledger_df = self._window_filter(self._read_parquet(self.order_ledger_path), "created_at", as_of)
        journal_df = self._window_filter(journal_df, "time", as_of)

        if not ledger_df.empty and "ticker" in ledger_df.columns:
            ledger_df["_family"] = ledger_df["ticker"].map(
                lambda t: _match_family(str(t), self.prefix_map)
            )
        if not journal_df.empty and "ticker" in journal_df.columns:
            journal_df["_family"] = journal_df["ticker"].map(
                lambda t: _match_family(str(t), self.prefix_map)
            )

        orphan_state = self._read_orphan_state()

        out: dict[str, FamilyHealth] = {}
        rows = []
        for family_name in self.registry.keys():
            metrics = self._compute_family_metrics(
                family_name, ledger_df, journal_df, orphan_state
            )
            score, mult, healthy = self._score_and_multiplier(metrics)
            effective_mult = 1.0 if self.shadow_mode else mult
            health = FamilyHealth(
                family=family_name,
                healthy=healthy,
                score=score,
                throttle_multiplier=effective_mult,
                metrics=metrics,
            )
            out[family_name] = health
            rows.append(
                {
                    "family": family_name,
                    "score": score,
                    "healthy": healthy,
                    "throttle_multiplier": effective_mult,
                    "trade_count": metrics["trade_count"],
                    "win_rate": metrics["win_rate"],
                    "realized_pnl_cents": metrics["realized_pnl_cents"],
                    "max_drawdown_cents": metrics["max_drawdown_cents"],
                    "fill_ratio": metrics["fill_ratio"],
                    "orphan_rate": metrics["orphan_rate"],
                    "as_of": as_of.isoformat(),
                }
            )

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(self.cache_path, index=False)
        except Exception:
            logger.warning("FamilyScorecard: failed to write cache %s", self.cache_path)

        self._last = out
        return out

    def get_family_health(
        self, family: str, as_of: Optional[datetime] = None
    ) -> FamilyHealth:
        if not self._last or as_of is not None:
            self.compute(as_of=as_of)
        if family in self._last:
            return self._last[family]
        # Unknown family: treat as healthy no-op
        return FamilyHealth(
            family=family,
            healthy=True,
            score=1.0,
            throttle_multiplier=1.0,
            metrics={
                "trade_count": 0,
                "win_rate": 0.5,
                "realized_pnl_cents": 0,
                "max_drawdown_cents": 0,
                "fill_ratio": 1.0,
                "orphan_rate": 0.0,
            },
        )
