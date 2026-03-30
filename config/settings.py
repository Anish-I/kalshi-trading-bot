from pathlib import Path

from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    KALSHI_API_KEY_ID: str = ""
    KALSHI_PRIVATE_KEY_PATH: str = "./kalshi_private_key.pem"
    KALSHI_BASE_URL: str = "https://api.elections.kalshi.com/trade-api/v2"
    BINANCE_WS_URL: str = "wss://stream.binance.com:9443"
    BINANCE_SYMBOL: str = "btcusdt"
    DATA_DIR: str = "D:/kalshi-data"
    MODEL_DIR: str = "D:/kalshi-models"
    MAX_POSITION_CONTRACTS: int = 10
    DAILY_LOSS_LIMIT_CENTS: int = 5000
    CONFIDENCE_THRESHOLD: float = 0.65
    CONSECUTIVE_LOSS_HALT: int = 5
    COLLECTOR_STALE_SECONDS: int = 120
    CRYPTO_LIVE_SESSIONS: str = "us_core"
    CRYPTO_DECISION_MODE: str = "calibrated_ev"
    CRYPTO_CALIBRATION_PATH: str = "D:/kalshi-models/crypto_conjunction_calibration.json"
    CRYPTO_CALIBRATION_MIN_TRADES: int = 50
    CRYPTO_EV_BUFFER_CENTS: float = 2.0
    CRYPTO_MIN_NET_EV_CENTS: float = 1.0

    # Weather bot settings
    NWS_USER_AGENT: str = "KalshiWeatherBot (kalshi-weather-bot@example.com)"
    WEATHER_EDGE_THRESHOLD: float = 0.20
    WEATHER_MAX_CONTRACTS: int = 10
    WEATHER_SCAN_INTERVAL_MINUTES: int = 15
    WEATHER_TIER1_SIZE_MULTIPLIER: float = 1.0
    WEATHER_TIER2_SIZE_MULTIPLIER: float = 0.5
    WEATHER_TIER3_LIVE_ENABLED: bool = False

    # Coinbase CDP API Key (for authenticated WebSocket)
    COINBASE_CDP_KEY_ID: str = ""
    COINBASE_CDP_PRIVATE_KEY: str = ""

    # Kraken API Key
    KRAKEN_API_KEY: str = ""
    KRAKEN_PRIVATE_KEY: str = ""

    model_config = {"env_file": str(_PROJECT_ROOT / ".env"), "env_file_encoding": "utf-8"}


WEATHER_CITIES = [
    # --- HIGH temp markets ---
    {"name": "New York", "short": "NYC", "lat": 40.7829, "lon": -73.9654, "series_ticker": "KXHIGHNY", "type": "high"},
    {"name": "Los Angeles", "short": "LA", "lat": 33.9425, "lon": -118.4081, "series_ticker": "KXHIGHLAX", "type": "high"},
    {"name": "Chicago", "short": "CHI", "lat": 41.786, "lon": -87.752, "series_ticker": "KXHIGHCHI", "type": "high"},
    {"name": "Miami", "short": "MIA", "lat": 25.7617, "lon": -80.1918, "series_ticker": "KXHIGHMIA", "type": "high"},
    {"name": "Phoenix", "short": "PHX", "lat": 33.4484, "lon": -112.074, "series_ticker": "KXHIGHTPHX", "type": "high"},
    {"name": "Atlanta", "short": "ATL", "lat": 33.749, "lon": -84.388, "series_ticker": "KXHIGHTATL", "type": "high"},
    {"name": "San Francisco", "short": "SFO", "lat": 37.6213, "lon": -122.379, "series_ticker": "KXHIGHTSFO", "type": "high"},
    {"name": "Las Vegas", "short": "LV", "lat": 36.1699, "lon": -115.1398, "series_ticker": "KXHIGHTLV", "type": "high"},
    {"name": "Austin", "short": "AUS", "lat": 30.2672, "lon": -97.7431, "series_ticker": "KXHIGHAUS", "type": "high"},
    {"name": "Denver", "short": "DEN", "lat": 39.8561, "lon": -104.6737, "series_ticker": "KXHIGHDEN", "type": "high"},
    {"name": "Dallas", "short": "DAL", "lat": 32.8998, "lon": -97.0403, "series_ticker": "KXHIGHTDAL", "type": "high"},
    {"name": "Seattle", "short": "SEA", "lat": 47.4502, "lon": -122.3088, "series_ticker": "KXHIGHTSEA", "type": "high"},
    {"name": "Washington DC", "short": "DC", "lat": 38.8510, "lon": -77.0402, "series_ticker": "KXHIGHTDC", "type": "high"},
    {"name": "Minneapolis", "short": "MIN", "lat": 44.8831, "lon": -93.2289, "series_ticker": "KXHIGHTMIN", "type": "high"},
    {"name": "Houston", "short": "HOU", "lat": 29.9844, "lon": -95.3414, "series_ticker": "KXHIGHTHOU", "type": "high"},
    {"name": "San Antonio", "short": "SAT", "lat": 29.5300, "lon": -98.4681, "series_ticker": "KXHIGHTSATX", "type": "high"},
    {"name": "Boston", "short": "BOS", "lat": 42.3601, "lon": -71.0589, "series_ticker": "KXHIGHTBOS", "type": "high"},
    {"name": "New Orleans", "short": "NOLA", "lat": 29.9511, "lon": -90.0715, "series_ticker": "KXHIGHTNOLA", "type": "high"},
    {"name": "Philadelphia", "short": "PHL", "lat": 39.8744, "lon": -75.2424, "series_ticker": "KXHIGHPHIL", "type": "high"},
    {"name": "Oklahoma City", "short": "OKC", "lat": 35.4676, "lon": -97.5164, "series_ticker": "KXHIGHTOKC", "type": "high"},
    # --- LOW temp markets ---
    {"name": "New York Low", "short": "NYC-L", "lat": 40.7829, "lon": -73.9654, "series_ticker": "KXLOWTNYC", "type": "low"},
    {"name": "Los Angeles Low", "short": "LA-L", "lat": 33.9425, "lon": -118.4081, "series_ticker": "KXLOWTLAX", "type": "low"},
    {"name": "Chicago Low", "short": "CHI-L", "lat": 41.786, "lon": -87.752, "series_ticker": "KXLOWTCHI", "type": "low"},
    {"name": "Miami Low", "short": "MIA-L", "lat": 25.7617, "lon": -80.1918, "series_ticker": "KXLOWTMIA", "type": "low"},
    {"name": "Denver Low", "short": "DEN-L", "lat": 39.8561, "lon": -104.6737, "series_ticker": "KXLOWTDEN", "type": "low"},
    {"name": "Austin Low", "short": "AUS-L", "lat": 30.2672, "lon": -97.7431, "series_ticker": "KXLOWTAUS", "type": "low"},
    {"name": "Philadelphia Low", "short": "PHL-L", "lat": 39.8744, "lon": -75.2424, "series_ticker": "KXLOWTPHIL", "type": "low"},
]

CITY_BIAS_F = {
    "NYC": -2.81, "LA": -2.43, "CHI": -0.80, "MIA": -3.78,
    "PHX": +0.39, "ATL": -4.11, "SFO": -1.20, "LV": +1.70,
    "AUS": +0.52, "DEN": +0.15, "DAL": -0.95, "SEA": -3.42,
    "DC": -1.55, "MIN": -1.20, "HOU": -1.85, "SAT": +0.30,
    "BOS": -2.10, "NOLA": -1.65, "PHL": -2.30, "OKC": -0.70,
    "NYC-L": -4.72, "LA-L": -1.15, "CHI-L": -2.90, "MIA-L": -2.45,
    "DEN-L": -1.80, "AUS-L": +3.11, "PHL-L": -3.20,
}

CITY_TIERS = {
    1: ["PHX", "DEN", "LV", "SAT", "DC"],
    2: ["HOU", "NOLA", "CHI", "OKC", "AUS", "BOS", "SFO", "DAL", "MIN"],
    3: ["NYC", "LA", "MIA", "ATL", "SEA", "NYC-L", "LA-L", "CHI-L", "MIA-L", "DEN-L", "AUS-L", "PHL-L"],
}

settings = Settings()
