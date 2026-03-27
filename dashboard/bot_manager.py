"""Manage trading bot subprocesses — start, stop, restart.

Uses PID lock files (D:/kalshi-data/locks/) to detect bots launched
from ANY source — terminal, dashboard, or scheduled task.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from engine.process_lock import _is_pid_alive, LOCK_DIR

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BOT_CONFIGS = {
    "crypto_live": {
        "script": "scripts/crypto_ml_trader.py",
        "label": "Crypto LIVE",
        "log": "crypto_live_bg.log",
        "controllable": True,
        "mutual_exclude": "crypto_sim",
    },
    "crypto_sim": {
        "script": "scripts/crypto_ml_trader.py",
        "label": "Crypto SIM",
        "log": "crypto_sim_bg.log",
        "controllable": True,
        "extra_args": ["--simulate"],
        "mutual_exclude": "crypto_live",
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


class BotManager:
    def __init__(self):
        self._procs: dict[str, subprocess.Popen] = {}
        self._start_times: dict[str, float] = {}

    def _get_lock_pid(self, name: str) -> int | None:
        """Read PID from lock file. Both crypto_live and crypto_sim share crypto_ml_trader.pid."""
        # The actual script uses "crypto_ml_trader" as lock name regardless of sim/live
        lock_name = "crypto_ml_trader" if name.startswith("crypto_") else name
        lock_file = LOCK_DIR / f"{lock_name}.pid"
        if not lock_file.exists():
            return None
        try:
            return int(lock_file.read_text().strip())
        except (ValueError, OSError):
            return None

    def _is_running(self, name: str) -> tuple[bool, int | None]:
        """Check if bot is running via managed process OR lock file."""
        # Check managed process first
        proc = self._procs.get(name)
        if proc is not None and proc.poll() is None:
            return True, proc.pid

        # Check lock file (catches terminal-launched processes)
        pid = self._get_lock_pid(name)
        if pid is not None and _is_pid_alive(pid):
            return True, pid

        return False, None

    def start(self, name: str) -> dict:
        if name not in BOT_CONFIGS:
            return {"ok": False, "error": f"Unknown bot: {name}"}

        cfg = BOT_CONFIGS[name]
        if not cfg.get("controllable", True):
            return {"ok": False, "error": f"{cfg['label']} is read-only"}

        running, pid = self._is_running(name)
        if running:
            return {"ok": False, "error": f"{name} already running (PID {pid})"}

        # Mutual exclusion: can't run live and sim at the same time
        exclude = cfg.get("mutual_exclude")
        if exclude:
            ex_running, ex_pid = self._is_running(exclude)
            if ex_running:
                return {"ok": False, "error": f"Stop {exclude} first (PID {ex_pid})"}

        script = str(PROJECT_ROOT / cfg["script"])
        log_path = str(PROJECT_ROOT / cfg["log"])

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        cmd = [sys.executable, script] + cfg.get("extra_args", [])

        proc = subprocess.Popen(
            cmd,
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

        # Try managed process first
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
            del self._procs[name]

        # Try lock file PID (externally launched)
        pid = self._get_lock_pid(name)
        if pid is not None and _is_pid_alive(pid):
            try:
                os.kill(pid, 15)  # SIGTERM
                time.sleep(2)
                if _is_pid_alive(pid):
                    os.kill(pid, 9)  # force
            except OSError:
                pass
            # Clean lock file
            lock_file = LOCK_DIR / f"{name}.pid"
            lock_file.unlink(missing_ok=True)
            logger.info("Stopped external %s (PID %d)", name, pid)
            return {"ok": True, "pid": pid}

        return {"ok": False, "error": f"{name} not running"}

    def restart(self, name: str) -> dict:
        self.stop(name)
        time.sleep(2)
        return self.start(name)

    def status(self) -> dict:
        result = {}
        for name, cfg in BOT_CONFIGS.items():
            running, pid = self._is_running(name)
            uptime = 0
            if running and name in self._start_times:
                uptime = int(time.time() - self._start_times[name])

            result[name] = {
                "label": cfg["label"],
                "running": running,
                "pid": pid,
                "uptime_s": uptime,
                "log_file": cfg["log"],
                "controllable": cfg.get("controllable", True),
            }
        return result
