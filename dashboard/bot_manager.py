"""Manage trading bot subprocesses — start, stop, restart."""

import logging
import os
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
        "controllable": True,
    },
    "weather_trade": {
        "script": "scripts/weather_trade.py",
        "label": "Weather Trader",
        "log": "weather_trade_bg.log",
        "controllable": True,
    },
    "collect_data": {
        "script": "scripts/collect_data.py",
        "label": "Data Collector",
        "log": "collect_data_bg.log",
        "controllable": False,
    },
}


def _find_process(script_name: str) -> int | None:
    """Find PID of a running Python process by script name."""
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             f"Get-Process python* -ErrorAction SilentlyContinue | "
             f"Where-Object {{$_.CommandLine -like '*{script_name}*'}} | "
             f"Select-Object -ExpandProperty Id"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [int(p.strip()) for p in result.stdout.strip().split("\n") if p.strip().isdigit()]
        return pids[0] if pids else None
    except Exception:
        return None


class BotManager:
    def __init__(self):
        self._procs: dict[str, subprocess.Popen] = {}
        self._start_times: dict[str, float] = {}

    def start(self, name: str) -> dict:
        if name not in BOT_CONFIGS:
            return {"ok": False, "error": f"Unknown bot: {name}"}

        cfg = BOT_CONFIGS[name]
        if not cfg.get("controllable", True):
            return {"ok": False, "error": f"{cfg['label']} is read-only"}

        if name in self._procs and self._procs[name].poll() is None:
            return {"ok": False, "error": f"{name} already running (PID {self._procs[name].pid})"}

        script = str(PROJECT_ROOT / cfg["script"])
        log_path = str(PROJECT_ROOT / cfg["log"])

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=open(log_path, "a"),
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        self._procs[name] = proc
        self._start_times[name] = time.time()
        logger.info("Started %s (PID %d)", name, proc.pid)
        return {"ok": True, "pid": proc.pid}

    def stop(self, name: str) -> dict:
        if name not in BOT_CONFIGS:
            return {"ok": False, "error": f"Unknown bot: {name}"}

        cfg = BOT_CONFIGS[name]
        if not cfg.get("controllable", True):
            return {"ok": False, "error": f"{cfg['label']} is read-only"}

        if name in self._procs:
            proc = self._procs[name]
            if proc.poll() is None:
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
            else:
                del self._procs[name]

        return {"ok": False, "error": f"{name} not running"}

    def restart(self, name: str) -> dict:
        self.stop(name)
        time.sleep(1)
        return self.start(name)

    def status(self) -> dict:
        result = {}
        for name, cfg in BOT_CONFIGS.items():
            proc = self._procs.get(name)
            managed_running = proc is not None and proc.poll() is None

            # For uncontrollable bots, detect if running externally
            if not managed_running and not cfg.get("controllable", True):
                ext_pid = _find_process(cfg["script"].split("/")[-1])
                running = ext_pid is not None
                pid = ext_pid
                uptime = 0
            elif managed_running:
                running = True
                pid = proc.pid
                uptime = int(time.time() - self._start_times.get(name, time.time()))
            else:
                running = False
                pid = None
                uptime = 0

            result[name] = {
                "label": cfg["label"],
                "running": running,
                "pid": pid,
                "uptime_s": uptime,
                "log_file": cfg["log"],
                "controllable": cfg.get("controllable", True),
            }
        return result
