"""
Trade Journal: Persists every trading decision + outcome.
Feeds back into model voting weights based on recent accuracy.

Stores to D:/kalshi-data/trade_journal.parquet
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TradeJournal:
    """Persistent trade journal with adaptive model weighting."""

    def __init__(self, data_dir: str = "D:/kalshi-data"):
        self.journal_path = Path(data_dir) / "trade_journal.parquet"
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing journal
        self.entries: list[dict] = []
        if self.journal_path.exists():
            try:
                df = pd.read_parquet(self.journal_path)
                self.entries = df.to_dict("records")
                logger.info("Loaded %d journal entries from %s", len(self.entries), self.journal_path)
            except Exception:
                logger.warning("Failed to load journal, starting fresh")

        # Model accuracy tracking (rolling window)
        self._model_names = ["xgboost", "momentum", "mean_reversion", "kalshi_consensus"]
        self._recalc_weights()

    def log_decision(
        self,
        ticker: str,
        btc_price: float,
        models: dict,  # {name: {vote, confidence}}
        vote_result: str,  # UP/DOWN/FLAT
        agreement: int,
        action: str,  # trading/no_consensus/etc
        side: str | None = None,
        entry_price: int | None = None,
        contracts: int | None = None,
        bet_dollars: float | None = None,
        edge: float | None = None,
        features_snapshot: dict | None = None,
    ) -> None:
        """Log a trading decision (trade or no-trade)."""
        entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "btc_price": btc_price,
            "vote_result": vote_result,
            "agreement": agreement,
            "action": action,
            "side": side,
            "entry_price": entry_price,
            "contracts": contracts,
            "bet_dollars": bet_dollars,
            "edge": edge,
        }

        # Per-model votes
        for name in self._model_names:
            m = models.get(name, {})
            entry[f"{name}_vote"] = m.get("vote", "FLAT")
            entry[f"{name}_conf"] = m.get("confidence", 50.0)

        # Key features for analysis
        if features_snapshot:
            for feat in ["rsi_14", "momentum_5m", "momentum_10m", "donchian_position",
                         "ema_9_slope", "vwap_deviation", "volume_sma_ratio", "stoch_k"]:
                entry[f"feat_{feat}"] = features_snapshot.get(feat)

        self.entries.append(entry)

    def log_outcome(self, ticker: str, won: bool, pnl_cents: int) -> None:
        """Update the most recent entry for this ticker with the outcome."""
        for entry in reversed(self.entries):
            if entry.get("ticker") == ticker and entry.get("action") == "trading":
                entry["won"] = won
                entry["pnl_cents"] = pnl_cents
                entry["settled"] = True
                logger.info("Journal: %s %s pnl=%+dc", ticker, "WIN" if won else "LOSS", pnl_cents)
                self._recalc_weights()
                break

    def save(self) -> None:
        """Persist journal to parquet."""
        if not self.entries:
            return
        try:
            df = pd.DataFrame(self.entries)
            df.to_parquet(self.journal_path, index=False)
        except Exception:
            logger.exception("Failed to save journal")

    # ------------------------------------------------------------------ #
    #  Adaptive model weights
    # ------------------------------------------------------------------ #

    def _recalc_weights(self) -> None:
        """Recalculate model voting weights from recent settled trades."""
        self.model_weights = {m: 1.0 for m in self._model_names}

        # Get settled trades only
        settled = [e for e in self.entries if e.get("settled")]
        if len(settled) < 5:
            return  # Not enough data

        # Use last 50 settled trades
        recent = settled[-50:]

        for model_name in self._model_names:
            correct = 0
            total = 0
            for entry in recent:
                model_vote = entry.get(f"{model_name}_vote", "FLAT")
                if model_vote == "FLAT":
                    continue  # Model abstained, don't count

                won = entry.get("won", False)
                actual_direction = "UP" if won and entry.get("side") == "yes" else \
                                  "DOWN" if won and entry.get("side") == "no" else \
                                  "DOWN" if not won and entry.get("side") == "yes" else "UP"

                if model_vote == actual_direction:
                    correct += 1
                total += 1

            if total >= 3:
                accuracy = correct / total
                # Weight: 0.5 at 30% accuracy, 1.0 at 50%, 2.0 at 70%
                self.model_weights[model_name] = max(0.3, min(2.5, accuracy * 3.0 - 0.5))

        logger.info("Model weights updated: %s",
                     {k: f"{v:.2f}" for k, v in self.model_weights.items()})

    def get_model_weights(self) -> dict[str, float]:
        """Return current adaptive weights for each model."""
        return self.model_weights.copy()

    # ------------------------------------------------------------------ #
    #  Analytics
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict:
        """Return summary statistics."""
        settled = [e for e in self.entries if e.get("settled")]
        if not settled:
            return {"total_decisions": len(self.entries), "settled_trades": 0}

        wins = sum(1 for e in settled if e.get("won"))
        losses = len(settled) - wins
        total_pnl = sum(e.get("pnl_cents", 0) for e in settled)

        # Per-model accuracy
        model_stats = {}
        for model_name in self._model_names:
            correct = total = 0
            for entry in settled:
                vote = entry.get(f"{model_name}_vote", "FLAT")
                if vote == "FLAT":
                    continue
                won = entry.get("won", False)
                actual = "UP" if (won and entry.get("side") == "yes") or \
                                 (not won and entry.get("side") == "no") else "DOWN"
                if vote == actual:
                    correct += 1
                total += 1
            model_stats[model_name] = {
                "accuracy": correct / total if total > 0 else 0,
                "total": total,
                "weight": self.model_weights.get(model_name, 1.0),
            }

        return {
            "total_decisions": len(self.entries),
            "settled_trades": len(settled),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(settled) if settled else 0,
            "total_pnl_cents": total_pnl,
            "model_stats": model_stats,
        }

    def get_condition_analysis(self) -> dict:
        """Analyze win rate by market conditions."""
        settled = [e for e in self.entries if e.get("settled")]
        if len(settled) < 10:
            return {}

        analysis = {}

        # Win rate by RSI bucket
        rsi_buckets = {"oversold(<35)": [], "neutral(35-65)": [], "overbought(>65)": []}
        for e in settled:
            rsi = e.get("feat_rsi_14", 50)
            if rsi is None:
                continue
            if rsi < 35:
                rsi_buckets["oversold(<35)"].append(e.get("won", False))
            elif rsi > 65:
                rsi_buckets["overbought(>65)"].append(e.get("won", False))
            else:
                rsi_buckets["neutral(35-65)"].append(e.get("won", False))

        analysis["by_rsi"] = {
            k: {"win_rate": sum(v) / len(v) if v else 0, "count": len(v)}
            for k, v in rsi_buckets.items()
        }

        # Win rate by agreement level
        agree_buckets = defaultdict(list)
        for e in settled:
            agree_buckets[e.get("agreement", 0)].append(e.get("won", False))
        analysis["by_agreement"] = {
            f"{k}/4": {"win_rate": sum(v) / len(v), "count": len(v)}
            for k, v in sorted(agree_buckets.items())
        }

        # Win rate by side
        side_buckets = defaultdict(list)
        for e in settled:
            side_buckets[e.get("side", "?")].append(e.get("won", False))
        analysis["by_side"] = {
            k: {"win_rate": sum(v) / len(v), "count": len(v)}
            for k, v in side_buckets.items()
        }

        return analysis
