"""Manage trading bot subprocesses — start, stop, restart."""

import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BOT_CONFIGS = {
    "crypto_ml_trader": {
        "script": "scripts/crypto_ml_trader.py",
        "label": "Crypto ML Trader",
        "log": "crypto_ml_trader_bg.log",
    },
    "weather_trade": {
        "script": "scripts/weather_trade.py",
        "label": "Weather Trader",
        "log": "weather_trade_bg.log",
    },
    "collect_data": {
        "script": "scripts/collect_data.py",
        "label": "Data Collector",
        "log": "collect_data_bg.log",
    },
}


class BotManager:
    def __init__(self):
        self._procs: dict[str, subprocess.Popen] = {}
        self._start_times: dict[str, float] = {}

    def start(self, name: str) -> dict:
        if name not in BOT_CONFIGS:
            return {"ok": False, "error": f"Unknown bot: {name}"}

        if name in self._procs and self._procs[name].poll() is None:
            return {"ok": False, "error": f"{name} already running (PID {self._procs[name].pid})"}

        cfg = BOT_CONFIGS[name]
        script = str(PROJECT_ROOT / cfg["script"])
        log_path = str(PROJECT_ROOT / cfg["log"])

        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=open(log_path, "a"),
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
        self._procs[name] = proc
        self._start_times[name] = time.time()
        logger.info("Started %s (PID %d)", name, proc.pid)
        return {"ok": True, "pid": proc.pid}

    def stop(self, name: str) -> dict:
        if name not in self._procs:
            return {"ok": False, "error": f"{name} not tracked"}

        proc = self._procs[name]
        if proc.poll() is not None:
            del self._procs[name]
            return {"ok": False, "error": f"{name} already stopped"}

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        pid = proc.pid
        del self._procs[name]
        self._start_times.pop(name, None)
        logger.info("Stopped %s (PID %d)", name, pid)
        return {"ok": True, "pid": pid}

    def restart(self, name: str) -> dict:
        self.stop(name)
        time.sleep(1)
        return self.start(name)

    def status(self) -> dict:
        result = {}
        for name, cfg in BOT_CONFIGS.items():
            proc = self._procs.get(name)
            running = proc is not None and proc.poll() is None
            uptime = int(time.time() - self._start_times[name]) if name in self._start_times and running else 0

            result[name] = {
                "label": cfg["label"],
                "running": running,
                "pid": proc.pid if proc and running else None,
                "uptime_s": uptime,
                "log_file": cfg["log"],
            }
        return result
