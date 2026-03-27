"""Trading session classification by time and day."""
from datetime import datetime, timezone

def get_session_tag(dt: datetime = None) -> str:
    """Return session tag: us_core, us_off, or weekend."""
    if dt is None:
        dt = datetime.now(timezone.utc)

    dow = dt.weekday()  # 0=Monday, 6=Sunday
    hour = dt.hour + dt.minute / 60.0

    # Weekend: Saturday or Sunday
    if dow >= 5:
        return "weekend"

    # US core: 14:30-21:00 UTC (9:30am-4pm ET)
    if 14.5 <= hour < 21.0:
        return "us_core"

    return "us_off"

def is_live_session(session_tag: str, allowed: str = "us_core") -> bool:
    """Check if session allows live trading."""
    allowed_list = [s.strip() for s in allowed.split(",")]
    return session_tag in allowed_list
