"""Minimal alert system — file-first, Telegram optional.

Alerts are always written to D:/kalshi-data/logs/alerts.log.
Telegram delivery is best-effort and non-blocking.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ALERT_LOG = Path(os.environ.get("KALSHI_ALERT_LOG", "D:/kalshi-data/logs/alerts.log"))
ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)

# Severity levels
INFO = "INFO"
WARNING = "WARNING"
CRITICAL = "CRITICAL"


TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8730705383:AAHoqKnc_hAxGWlP4QvJQrnpW5ZjBq76R4g")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "5006911570")


def _send_telegram(text: str) -> None:
    """Best-effort Telegram notification via Bot API. Never blocks or raises."""
    try:
        import threading

        def _send():
            try:
                import httpx
                httpx.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
                    timeout=5.0,
                )
            except Exception:
                pass

        threading.Thread(target=_send, daemon=True).start()
    except Exception:
        pass


def alert(severity: str, category: str, message: str, **extra) -> None:
    """Write an alert to log file + send Telegram. Non-blocking, never raises."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    extra_str = " ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
    line = f"[{ts}] [{severity}] [{category}] {message} {extra_str}".rstrip()

    # Always write to file first
    try:
        with open(ALERT_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        logger.error("Failed to write alert to %s", ALERT_LOG, exc_info=True)

    # Log to Python logger too
    log_level = {"INFO": logging.INFO, "WARNING": logging.WARNING, "CRITICAL": logging.ERROR}
    logger.log(log_level.get(severity, logging.INFO), "ALERT: %s", line)

    # Telegram for trades and critical alerts only
    if category in ("TRADE_PLACED", "SETTLEMENT", "RISK_HALT", "BALANCE_LOW"):
        _send_telegram(f"🔔 {category}\n{message}")


# --- Convenience functions ---

def alert_trade_placed(ticker: str, side: str, price_cents: int, contracts: int,
                       edge_pct: float, strategy: str = "unknown") -> None:
    alert(INFO, "TRADE_PLACED",
          f"{strategy} {side.upper()} {ticker} @{price_cents}c x{contracts} edge={edge_pct:.1f}%")


def alert_settlement(ticker: str, outcome: str, pnl_cents: int,
                     strategy: str = "unknown") -> None:
    sev = INFO if pnl_cents >= 0 else WARNING
    alert(sev, "SETTLEMENT",
          f"{strategy} {ticker} {outcome} pnl={pnl_cents:+d}c")


def alert_risk_halt(reason: str, daily_pnl_cents: int = 0,
                    strategy: str = "unknown") -> None:
    alert(CRITICAL, "RISK_HALT",
          f"{strategy} halted: {reason} daily_pnl={daily_pnl_cents:+d}c")


def alert_balance_warning(balance_dollars: float, threshold: float = 50.0) -> None:
    alert(CRITICAL, "BALANCE_LOW",
          f"Balance ${balance_dollars:.2f} below ${threshold:.0f} threshold")


def alert_collector_stale(age_seconds: int) -> None:
    alert(WARNING, "COLLECTOR_STALE",
          f"Data collector stale: {age_seconds}s since last update")


def get_recent_alerts(n: int = 50) -> list[str]:
    """Read last N alerts from the log file."""
    try:
        lines = ALERT_LOG.read_text(encoding="utf-8").strip().splitlines()
        return lines[-n:]
    except Exception:
        return []
