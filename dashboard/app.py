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
from engine.order_ledger import OrderLedger

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

    ledger = OrderLedger(str(DATA_DIR))

    return {
        "balance": balance,
        "positions": pos_list,
        "bots": bots.status(),
        "collector_health": collector,
        "resting_orders": len(ledger.get_open_orders()),
        "ledger_stats": ledger.get_stats(),
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


@app.get("/api/execution")
def get_execution():
    """Execution quality metrics from order ledger."""
    _ledger = OrderLedger(str(DATA_DIR))
    return {
        "all": _ledger.get_execution_stats(),
        "crypto": _ledger.get_execution_stats("crypto"),
        "crypto_sim": _ledger.get_execution_stats("crypto_sim"),
        "weather": _ledger.get_execution_stats("weather"),
    }


@app.get("/api/alerts")
def get_alerts():
    """Recent alerts from file."""
    from engine.alerts import get_recent_alerts
    return {"alerts": get_recent_alerts(50)}


@app.get("/api/archive/status")
def get_archive_status():
    """Archive health info."""
    from engine.market_archive import get_archive_status as _get_status
    return _get_status()


@app.get("/api/calibration/weather")
def get_weather_calibration():
    """Weather calibration summary."""
    from engine.weather_calibration import calibration_summary
    return calibration_summary()


@app.get("/api/weather/breakdown")
def get_weather_breakdown():
    """Per-city weather trading breakdown from order ledger + positions."""
    import math
    _ledger = OrderLedger(str(DATA_DIR))
    weather_recs = [r for r in _ledger._records if r.get("strategy") == "weather" and r.get("status") == "settled"]

    # Group by city (extract from ticker: KXHIGHNY → NYC, KXHIGHTPHX → PHX)
    city_stats = {}
    for r in weather_recs:
        ticker = r.get("ticker", "")
        # Extract city code from ticker
        city = ticker.replace("KXHIGH", "").replace("KXLOW", "").replace("T", "").replace("-", " ").split()[0] if ticker else "?"
        if city not in city_stats:
            city_stats[city] = {"trades": 0, "wins": 0, "pnl": 0}
        city_stats[city]["trades"] += 1
        pnl = r.get("pnl_cents", 0)
        if isinstance(pnl, float) and math.isnan(pnl):
            pnl = 0
        city_stats[city]["pnl"] += int(pnl)
        if int(pnl) > 0:
            city_stats[city]["wins"] += 1

    # Sort by P&L
    breakdown = []
    for city, stats in sorted(city_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        breakdown.append({
            "city": city,
            "trades": stats["trades"],
            "wins": stats["wins"],
            "losses": stats["trades"] - stats["wins"],
            "win_rate": round(stats["wins"] / stats["trades"] * 100) if stats["trades"] else 0,
            "pnl_cents": stats["pnl"],
        })

    total_trades = sum(s["trades"] for s in city_stats.values())
    total_pnl = sum(s["pnl"] for s in city_stats.values())
    total_wins = sum(s["wins"] for s in city_stats.values())

    return {
        "breakdown": breakdown,
        "summary": {
            "total_trades": total_trades,
            "total_wins": total_wins,
            "total_pnl_cents": total_pnl,
            "win_rate": round(total_wins / total_trades * 100) if total_trades else 0,
        },
    }


@app.get("/api/crypto/stats")
def get_crypto_stats():
    """Conjunction vs single-model stats from trade journal."""
    import math
    import pandas as pd

    stats = {"conjunction": {"trades": 0, "wins": 0, "pnl": 0},
             "exploration": {"trades": 0, "wins": 0, "pnl": 0}}

    try:
        journal_path = DATA_DIR / "trade_journal.parquet"
        if journal_path.exists():
            df = pd.read_parquet(journal_path)
            sim = df[df["action"] == "simulated"]
            for _, row in sim.iterrows():
                rule = str(row.get("rule_name", ""))
                settled = row.get("settled", False)
                if not settled:
                    continue
                won = row.get("won", False)
                pnl = row.get("pnl_cents", 0)
                if isinstance(pnl, float) and math.isnan(pnl):
                    pnl = 0
                pnl = int(pnl)

                if "single_model" in rule or "exploration" in rule:
                    key = "exploration"
                else:
                    key = "conjunction"

                stats[key]["trades"] += 1
                stats[key]["pnl"] += pnl
                if won:
                    stats[key]["wins"] += 1
    except Exception:
        pass

    for key in stats:
        s = stats[key]
        s["losses"] = s["trades"] - s["wins"]
        s["win_rate"] = round(s["wins"] / s["trades"] * 100) if s["trades"] else 0
        s["avg_pnl"] = round(s["pnl"] / s["trades"], 1) if s["trades"] else 0

    return stats


@app.get("/api/pairs")
def get_pairs():
    """Pair trader state and trade log."""
    state = {}
    try:
        f = DATA_DIR / "pair_trader_state.json"
        if f.exists():
            state = json.loads(f.read_text())
    except Exception:
        pass

    trades = []
    try:
        log_path = DATA_DIR / "logs" / "pair_trades.log"
        if log_path.exists():
            for line in log_path.read_text().splitlines()[-20:]:
                trades.append(line)
    except Exception:
        pass

    return {"state": state, "trades": trades}


@app.get("/api/families")
def api_families():
    """Return per-family health from FamilyScorecard."""
    try:
        from engine.family_scorecard import FamilyScorecard
        sc = FamilyScorecard(
            shadow_mode=settings.SCORECARD_SHADOW_MODE,
            window_hours=settings.SCORECARD_WINDOW_HOURS,
        )
        healths = sc.compute()
        return {
            "shadow_mode": settings.SCORECARD_SHADOW_MODE,
            "window_hours": settings.SCORECARD_WINDOW_HOURS,
            "families": [
                {
                    "family": h.family,
                    "healthy": h.healthy,
                    "score": h.score,
                    "throttle_multiplier": h.throttle_multiplier,
                    "metrics": h.metrics,
                }
                for h in healths.values()
            ],
        }
    except Exception as e:
        return {
            "shadow_mode": getattr(settings, "SCORECARD_SHADOW_MODE", True),
            "window_hours": getattr(settings, "SCORECARD_WINDOW_HOURS", 48),
            "error": str(e),
            "families": [],
        }


GATE_STATE_FILES = {
    "combined_trader": DATA_DIR / "combined_trader_state.json",
    "ml_trader": DATA_DIR / "ml_trader_state.json",
    "pair_trader": DATA_DIR / "pair_trader_state.json",
}


@app.get("/api/gate/recent")
def api_gate_recent():
    """Return last gate_last_decision from each trader's state file."""
    out = {}
    for name, path in GATE_STATE_FILES.items():
        try:
            if Path(path).exists():
                data = json.loads(Path(path).read_text())
                out[name] = {
                    "state_time": data.get("time"),
                    "gate_last_decision": data.get("gate_last_decision"),
                }
            else:
                out[name] = {
                    "state_time": None,
                    "gate_last_decision": None,
                    "error": "state file missing",
                }
        except Exception as e:
            out[name] = {"error": str(e)}
    return out


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
                import math
                pnl = row.get("pnl_cents", 0)
                won = row.get("won")
                ep = row.get("entry_price", 0)
                paper_trades.append({
                    "ticker": row.get("ticker", ""),
                    "side": row.get("side", ""),
                    "entry_price": int(ep) if ep is not None and not (isinstance(ep, float) and math.isnan(ep)) else 0,
                    "won": bool(won) if won is not None and not (isinstance(won, float) and math.isnan(won)) else None,
                    "pnl_cents": int(pnl) if pnl is not None and not (isinstance(pnl, float) and math.isnan(pnl)) else 0,
                    "settled": bool(row.get("settled", False)),
                    "time": str(row.get("time", "")),
                    "signal_ref": str(row.get("rule_name", "")),
                })
    except Exception:
        pass

    return {"crypto": crypto_state, "weather": weather_state, "notification": notification, "paper_trades": paper_trades}


# -----------------------------------------------------------------------------
# Phase A — Agents + proposals endpoints
# -----------------------------------------------------------------------------

import os as _os
import re as _re
from fastapi import HTTPException as _HTTPException

_AGENT_PROPOSAL_ROOT = Path(
    _os.environ.get("KALSHI_AGENT_PROPOSAL_ROOT", "D:/kalshi-data/agent_proposals")
)
_AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_agent_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        try:
            from scripts.agents.orchestrator import _tiny_yaml  # type: ignore
            return _tiny_yaml(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


@app.get("/api/agents")
def list_agents():
    """List all agent YAMLs under agents/ with their metadata."""
    out = []
    if not _AGENTS_DIR.exists():
        return {"agents": []}
    for p in sorted(_AGENTS_DIR.glob("*.yaml")):
        try:
            cfg = _load_agent_yaml(p)
            out.append({
                "name": cfg.get("name", p.stem),
                "version": cfg.get("version", ""),
                "authority_class": cfg.get("authority_class", "reporter"),
                "writable_config_keys": cfg.get("writable_config_keys") or [],
                "path": str(p.relative_to(_REPO_ROOT)),
            })
        except Exception:
            continue
    return {"agents": out}


@app.get("/api/agents/{name}/config")
def agent_config(name: str):
    """Return current live values of this agent's writable_config_keys."""
    path = _AGENTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise _HTTPException(status_code=404, detail="agent not found")
    cfg = _load_agent_yaml(path)
    keys = cfg.get("writable_config_keys") or []
    current: dict = {}
    for entry in keys:
        if not isinstance(entry, dict):
            continue
        module = entry.get("module")
        key = entry.get("key")
        default = entry.get("default")
        if not module or not key:
            continue
        mod_path = _REPO_ROOT / module
        val = default
        if mod_path.exists():
            try:
                text = mod_path.read_text(encoding="utf-8")
                m = _re.search(rf"^\s*{_re.escape(key)}\s*=\s*(.+)$", text, _re.MULTILINE)
                if m:
                    raw = m.group(1).split("#", 1)[0].strip().rstrip(",")
                    try:
                        val = int(raw)
                    except ValueError:
                        try:
                            val = float(raw)
                        except ValueError:
                            val = raw.strip("'\"")
            except Exception:
                pass
        current[key] = val
    return {"agent": name, "current_config": current, "writable_config_keys": keys}


def _proposals_dir(state: str) -> Path:
    return _AGENT_PROPOSAL_ROOT / state


@app.get("/api/proposals")
def list_proposals(state: str = "pending"):
    """List proposals in pending|approved|rejected|applied state."""
    if state not in ("pending", "approved", "rejected", "applied"):
        raise _HTTPException(status_code=400, detail="invalid state")
    d = _proposals_dir(state)
    items = []
    if d.exists():
        for p in sorted(d.glob("*.json")):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                items.append({
                    "id": obj.get("id", p.stem),
                    "agent": obj.get("agent", ""),
                    "created_at": obj.get("created_at", ""),
                    "codex_verdict": (obj.get("codex_review") or {}).get("verdict", ""),
                    "state": obj.get("state", state),
                })
            except Exception:
                continue
    return {"state": state, "proposals": items}


@app.get("/api/proposals/{proposal_id}")
def get_proposal(proposal_id: str):
    """Return full proposal JSON including codex review."""
    for state in ("pending", "approved", "rejected", "applied"):
        p = _proposals_dir(state) / f"{proposal_id}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                raise _HTTPException(status_code=500, detail="proposal JSON corrupt")
    raise _HTTPException(status_code=404, detail="proposal not found")


def _move_proposal(proposal_id: str, src: str, dst: str, mutate: dict | None = None) -> dict:
    src_path = _proposals_dir(src) / f"{proposal_id}.json"
    if not src_path.exists():
        raise _HTTPException(status_code=404, detail=f"proposal not found in {src}")
    try:
        obj = json.loads(src_path.read_text(encoding="utf-8"))
    except Exception:
        raise _HTTPException(status_code=500, detail="proposal JSON corrupt")
    obj["state"] = dst
    if mutate:
        obj.update(mutate)
    dst_dir = _proposals_dir(dst)
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / f"{proposal_id}.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")
    try:
        src_path.unlink()
    except Exception:
        pass
    return obj


@app.post("/api/proposals/{proposal_id}/approve")
def approve_proposal(proposal_id: str):
    obj = _move_proposal(
        proposal_id, "pending", "approved",
        mutate={"human_decision": {"action": "approve", "at": _now_iso()}},
    )
    return {"ok": True, "proposal": obj}


@app.post("/api/proposals/{proposal_id}/reject")
def reject_proposal(proposal_id: str, reason: str = ""):
    obj = _move_proposal(
        proposal_id, "pending", "rejected",
        mutate={"human_decision": {"action": "reject", "reason": reason, "at": _now_iso()}},
    )
    return {"ok": True, "proposal": obj}


def _now_iso() -> str:
    from datetime import datetime as _dt, timezone as _tz
    return _dt.now(_tz.utc).isoformat()
