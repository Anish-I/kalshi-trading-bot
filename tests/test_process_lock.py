"""Tests for singleton process lock."""
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, ".")


def test_acquire_and_release():
    """Lock can be acquired and released."""
    from engine.process_lock import ProcessLock, LOCK_DIR
    # Use temp dir to avoid conflicts
    import engine.process_lock as pl
    old_dir = pl.LOCK_DIR
    pl.LOCK_DIR = Path(tempfile.mkdtemp())

    lock = ProcessLock("test_bot")
    lock.lock_file = pl.LOCK_DIR / "test_bot.pid"
    assert lock.acquire()
    assert lock.lock_file.exists()
    assert lock.lock_file.read_text().strip() == str(os.getpid())
    lock.release()
    assert not lock.lock_file.exists()

    pl.LOCK_DIR = old_dir


def test_blocks_second_instance():
    """Second lock on same name is blocked if first is alive."""
    from engine.process_lock import ProcessLock
    import engine.process_lock as pl
    pl.LOCK_DIR = Path(tempfile.mkdtemp())

    lock1 = ProcessLock("test_bot2")
    lock1.lock_file = pl.LOCK_DIR / "test_bot2.pid"
    assert lock1.acquire()

    lock2 = ProcessLock("test_bot2")
    lock2.lock_file = pl.LOCK_DIR / "test_bot2.pid"
    # Same PID (same process), so it should detect as alive
    assert not lock2.acquire()

    lock1.release()


def test_cleans_stale_lock():
    """Stale lock (dead PID) is cleaned up automatically."""
    from engine.process_lock import ProcessLock
    import engine.process_lock as pl
    pl.LOCK_DIR = Path(tempfile.mkdtemp())

    lock = ProcessLock("test_stale")
    lock.lock_file = pl.LOCK_DIR / "test_stale.pid"
    # Write a fake PID that doesn't exist
    lock.lock_file.write_text("999999999")

    # Should clean up stale lock and acquire
    assert lock.acquire()
    lock.release()
