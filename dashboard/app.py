"""FastAPI dashboard backend for Kalshi trading bot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import json

from kalshi.client import KalshiClient
from dashboard.bot_manager import BotManager
from config.settings import settings
from engine.collector_health import check_collector_freshness

app = FastAPI(title="Kalshi Trading Dashboard")
kalshi = KalshiClient()
bots = BotManager()

STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = Path(settings.DATA_DIR)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def get_status():
    try:
        balance = kalshi.get_balance()
    except Exception:
        balance = 0.0

    try:
        result = kalshi._request("GET", "/portfolio/positions", params={"limit": 50})
        positions = result.get("market_positions", result.get("positions", []))
    except Exception:
        positions = []

    pos_list = []
    for p in positions:
        ticker = p.get("ticker", "")
        position = float(p.get("position_fp", 0))
        exposure = p.get("market_exposure_dollars", "0")
        realized = p.get("realized_pnl_dollars", "0")
        total_traded = p.get("total_traded_dollars", "0")
        updated = p.get("last_updated_ts", "")

        if position != 0 or float(exposure) > 0:
            pos_list.append({
                "ticker": ticker,
                "side": "YES" if position > 0 else "NO",
                "count": abs(int(position)),
                "exposure": float(exposure),
                "realized_pnl": float(realized),
                "total_traded": float(total_traded),
                "updated": updated,
            })

    collector = check_collector_freshness(str(DATA_DIR))

    return {
        "balance": balance,
        "positions": pos_list,
        "bots": bots.status(),
        "collector_health": collector,
    }


@app.get("/api/trades")
def get_trades():
    try:
        result = kalshi._request("GET", "/portfolio/settlements", params={"limit": 30})
        settlements = result.get("settlements", [])
    except Exception:
        settlements = []

    trades = []
    total_pnl = 0
    wins = 0
    total = 0

    for s in settlements:
        ticker = s.get("ticker", "")
        market_result = s.get("market_result", "")
        yes_cost = s.get("yes_total_cost", 0)
        no_cost = s.get("no_total_cost", 0)
        revenue = s.get("revenue", 0)
        yes_count = float(s.get("yes_count_fp", "0"))
        no_count = float(s.get("no_count_fp", "0"))

        if not yes_count and not no_count:
            continue

        side = "YES" if yes_count > 0 else "NO"
        cost = yes_cost if side == "YES" else no_cost
        count = int(yes_count if side == "YES" else no_count)
        won = (side == "YES" and market_result == "yes") or (side == "NO" and market_result == "no")

        pnl = (count * 100 - cost) if won else -cost

        total += 1
        if won:
            wins += 1
        total_pnl += pnl

        trades.append({
            "ticker": ticker,
            "side": side,
            "count": count,
            "cost_cents": cost,
            "result": market_result,
            "won": won,
            "pnl_cents": pnl,
            "settled_time": s.get("settled_time", ""),
        })

    return {
        "trades": trades,
        "summary": {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": (wins / total * 100) if total > 0 else 0,
            "total_pnl_cents": total_pnl,
            "total_pnl_dollars": total_pnl / 100,
        },
    }


@app.get("/api/bots")
def get_bots():
    return bots.status()


@app.post("/api/bots/{name}/start")
def start_bot(name: str):
    return bots.start(name)


@app.post("/api/bots/{name}/stop")
def stop_bot(name: str):
    return bots.stop(name)


@app.post("/api/bots/{name}/restart")
def restart_bot(name: str):
    return bots.restart(name)


@app.get("/api/logs")
def get_logs():
    """Return live state from both traders."""
    crypto_state = {}
    weather_state = {}
    notification = {}

    try:
        f = DATA_DIR / "ml_trader_state.json"
        if f.exists():
            crypto_state = json.loads(f.read_text())
    except Exception:
        pass

    try:
        f = DATA_DIR / "weather_trader_state.json"
        if f.exists():
            weather_state = json.loads(f.read_text())
    except Exception:
        pass

    try:
        f = DATA_DIR / "notifications.json"
        if f.exists():
            notification = json.loads(f.read_text())
    except Exception:
        pass

    # Paper trade journal
    paper_trades = []
    try:
        journal_path = DATA_DIR / "trade_journal.parquet"
        if journal_path.exists():
            import pandas as pd
            df = pd.read_parquet(journal_path)
            sim_trades = df[df["action"] == "simulated"].tail(20)
            for _, row in sim_trades.iterrows():
                paper_trades.append({
                    "ticker": row.get("ticker", ""),
                    "side": row.get("side", ""),
                    "entry_price": int(row.get("entry_price", 0) or 0),
                    "won": row.get("won"),
                    "pnl_cents": int(row.get("pnl_cents", 0) or 0),
                    "settled": bool(row.get("settled", False)),
                    "time": str(row.get("time", "")),
                })
    except Exception:
        pass

    return {"crypto": crypto_state, "weather": weather_state, "notification": notification, "paper_trades": paper_trades}
