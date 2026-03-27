"""Check if the data collector is producing fresh bars."""
import time
from pathlib import Path

def check_collector_freshness(data_dir: str = "D:/kalshi-data", stale_seconds: int = 120) -> dict:
    """Returns dict with freshness info."""
    results = []
    for subdir in ["bars_1m", "bars_5s"]:
        bars_dir = Path(data_dir) / subdir
        if not bars_dir.exists():
            continue
        files = sorted(bars_dir.glob("*.parquet"))
        if not files:
            continue
        age = time.time() - files[-1].stat().st_mtime
        results.append({"subdir": subdir, "age": int(age), "file": files[-1].name})

    if not results:
        return {"healthy": False, "reason": "no bar data", "age_seconds": None}

    worst = max(results, key=lambda r: r["age"])
    healthy = worst["age"] < stale_seconds
    return {
        "healthy": healthy,
        "age_seconds": worst["age"],
        "file": worst["file"],
        "reason": "ok" if healthy else f"stale ({worst['age']}s > {stale_seconds}s)",
    }
