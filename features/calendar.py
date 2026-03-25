import math
from datetime import datetime, time, timezone


class CalendarFeatures:
    def __init__(self):
        pass

    def compute(self, timestamp: datetime = None) -> dict[str, float]:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        hour = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
        minute = timestamp.minute + timestamp.second / 60.0
        dow = timestamp.weekday()  # Monday=0

        result: dict[str, float] = {
            "hour_sin": math.sin(2 * math.pi * hour / 24.0),
            "hour_cos": math.cos(2 * math.pi * hour / 24.0),
            "minute_sin": math.sin(2 * math.pi * minute / 60.0),
            "minute_cos": math.cos(2 * math.pi * minute / 60.0),
            "dow_sin": math.sin(2 * math.pi * dow / 7.0),
            "dow_cos": math.cos(2 * math.pi * dow / 7.0),
        }

        t = timestamp.time()

        # US session: 14:30 - 21:00 UTC
        result["is_us_session"] = 1.0 if time(14, 30) <= t < time(21, 0) else 0.0

        # Asia session: 00:00 - 06:30 UTC
        result["is_asia_session"] = 1.0 if t < time(6, 30) else 0.0

        # Europe session: 08:00 - 16:30 UTC
        result["is_europe_session"] = 1.0 if time(8, 0) <= t < time(16, 30) else 0.0

        return result
