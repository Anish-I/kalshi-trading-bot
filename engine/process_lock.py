"""
Process lock: prevents multiple instances of the same bot from running.

Uses a PID file + process validation. If an old lock exists but the
process is dead, the lock is automatically cleaned up.

Usage:
    lock = ProcessLock("crypto_ml_trader")
    if not lock.acquire():
        print("Another instance is already running!")
        sys.exit(1)
    # ... run bot ...
    lock.release()  # called automatically on normal exit
"""

import atexit
import logging
import os
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

LOCK_DIR = Path("D:/kalshi-data/locks")


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    if sys.platform == "win32":
        # Windows: use tasklist to check if PID exists
        import subprocess
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            return str(pid) in result.stdout
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


class ProcessLock:
    """File-based singleton lock for bot processes."""

    def __init__(self, name: str):
        self.name = name
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        self.lock_file = LOCK_DIR / f"{name}.pid"
        self._acquired = False

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns True if successful.

        If a stale lock exists (process dead), cleans it up automatically.
        If another live instance is running, returns False.
        """
        if self.lock_file.exists():
            try:
                old_pid = int(self.lock_file.read_text().strip())
                if _is_pid_alive(old_pid):
                    logger.error(
                        "BLOCKED: %s already running (PID %d). "
                        "Kill it first or delete %s",
                        self.name, old_pid, self.lock_file,
                    )
                    return False
                else:
                    logger.warning(
                        "Stale lock for %s (PID %d dead). Cleaning up.",
                        self.name, old_pid,
                    )
                    self.lock_file.unlink()
            except (ValueError, OSError):
                self.lock_file.unlink(missing_ok=True)

        # Write our PID
        self.lock_file.write_text(str(os.getpid()))
        self._acquired = True

        # Auto-release on exit
        atexit.register(self.release)

        logger.info(
            "Lock acquired: %s (PID %d) -> %s",
            self.name, os.getpid(), self.lock_file,
        )
        return True

    def release(self) -> None:
        """Release the lock."""
        if self._acquired and self.lock_file.exists():
            try:
                stored_pid = int(self.lock_file.read_text().strip())
                if stored_pid == os.getpid():
                    self.lock_file.unlink()
                    logger.info("Lock released: %s", self.name)
            except (ValueError, OSError):
                pass
            self._acquired = False

    def kill_existing(self) -> bool:
        """Kill any existing instance holding this lock. Returns True if killed."""
        if not self.lock_file.exists():
            return False

        try:
            old_pid = int(self.lock_file.read_text().strip())
            if _is_pid_alive(old_pid):
                logger.warning("Killing existing %s (PID %d)", self.name, old_pid)
                try:
                    if sys.platform == "win32":
                        import subprocess
                        subprocess.run(["taskkill", "/F", "/PID", str(old_pid)],
                                       capture_output=True, timeout=10)
                    else:
                        os.kill(old_pid, signal.SIGTERM)
                    import time
                    for _ in range(10):
                        if not _is_pid_alive(old_pid):
                            break
                        time.sleep(0.5)
                except OSError:
                    pass
                self.lock_file.unlink(missing_ok=True)
                logger.info("Killed existing %s (PID %d)", self.name, old_pid)
                return True
        except (ValueError, OSError):
            self.lock_file.unlink(missing_ok=True)

        return False
