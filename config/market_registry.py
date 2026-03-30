"""
Market Registry — one config per market family.

Defines series, discovery rules, underlying feeds, strategy types,
and per-family bankroll caps. Used by Phase 4 market expansion.
"""
from dataclasses import dataclass, field


@dataclass
class MarketFamily:
    name: str
    series_prefix: str  # Kalshi series prefix (e.g., "KXBTC15M")
    strategy_type: str  # microstructure_5m, macro_event, threshold_barrier, weather
    underlying_feed: str  # btc, eth, spx, fed, gold, boil, gfs_ensemble
    settlement_source: str  # kalshi_api, nws_observation
    entry_window_s: tuple[int, int] = (120, 780)  # (min_remaining, max_remaining)
    max_contracts: int = 10
    budget_cents: int = 5000  # per-family bankroll cap
    live_sessions: str = "us_core"  # when to trade live
    sim_enabled: bool = True
    live_enabled: bool = False  # must pass sim gates first
    calibration_path: str = ""
    notes: str = ""


# --- Active market families ---

MARKET_FAMILIES = {
    "btc_15m": MarketFamily(
        name="BTC 15-minute",
        series_prefix="KXBTC15M",
        strategy_type="microstructure_5m",
        underlying_feed="btc",
        settlement_source="kalshi_api",
        entry_window_s=(120, 780),
        max_contracts=10,
        budget_cents=5000,
        live_sessions="us_core",
        sim_enabled=True,
        live_enabled=False,
        calibration_path="D:/kalshi-models/crypto_conjunction_calibration.json",
        notes="XGB+MOM conjunction. Sim running.",
    ),
    "weather": MarketFamily(
        name="Weather Temperature",
        series_prefix="KXHIGH,KXLOW",
        strategy_type="weather",
        underlying_feed="gfs_ensemble",
        settlement_source="nws_observation",
        max_contracts=5,
        budget_cents=3000,
        live_sessions="always",
        sim_enabled=False,
        live_enabled=True,
        calibration_path="D:/kalshi-models/weather_calibration.json",
        notes="GFS ensemble + NWS. Risk-halted after 0/10.",
    ),
    "boil_b5": MarketFamily(
        name="BOIL $5 Barrier",
        series_prefix="BOILB5",
        strategy_type="threshold_barrier",
        underlying_feed="boil",
        settlement_source="kalshi_api",
        max_contracts=5,
        budget_cents=2000,
        sim_enabled=False,
        live_enabled=False,
        notes="Phase 4 stretch - only after semantics and feed quality are confirmed.",
    ),
}

# --- Phase 4 candidates (not yet active) ---

PHASE4_CANDIDATES = {
    "fed_cut": MarketFamily(
        name="Fed Rate Cut",
        series_prefix="KXFED",
        strategy_type="macro_event",
        underlying_feed="fed",
        settlement_source="kalshi_api",
        max_contracts=1,
        budget_cents=1000,
        live_sessions="always",
        sim_enabled=False,
        live_enabled=True,
        notes="LIVE — 1 contract max, $10 budget. High-conviction YES on thresholds below current rate.",
    ),
    "eth_5m": MarketFamily(
        name="ETH 5-minute",
        series_prefix="KXETH5M",
        strategy_type="microstructure_5m",
        underlying_feed="eth",
        settlement_source="kalshi_api",
        entry_window_s=(60, 240),
        max_contracts=10,
        budget_cents=3000,
        live_sessions="us_core",
        sim_enabled=False,
        live_enabled=False,
        notes="Phase 4 — sim first, 50 trades gate.",
    ),
    "spx_5m": MarketFamily(
        name="SPX 5-minute",
        series_prefix="KXSPX5M",
        strategy_type="microstructure_5m",
        underlying_feed="spx",
        settlement_source="kalshi_api",
        entry_window_s=(60, 240),
        max_contracts=10,
        budget_cents=3000,
        live_sessions="us_core",
        sim_enabled=False,
        live_enabled=False,
        notes="Phase 4 — weekday regular session only.",
    ),
    "gold_2400": MarketFamily(
        name="Gold $2400 Barrier",
        series_prefix="KXGOLD",
        strategy_type="threshold_barrier",
        underlying_feed="gold",
        settlement_source="kalshi_api",
        max_contracts=5,
        budget_cents=2000,
        sim_enabled=False,
        live_enabled=False,
        notes="Phase 4 stretch — needs threshold framework.",
    ),
}


def get_family(name: str) -> MarketFamily | None:
    """Look up a market family by name."""
    return MARKET_FAMILIES.get(name) or PHASE4_CANDIDATES.get(name)


def get_active_families() -> dict[str, MarketFamily]:
    """Get all currently active (sim or live enabled) families."""
    return {k: v for k, v in MARKET_FAMILIES.items()
            if v.sim_enabled or v.live_enabled}


def get_total_budget_cents() -> int:
    """Sum of all active family budgets."""
    return sum(f.budget_cents for f in get_active_families().values())


def validate_registry() -> list[str]:
    """Check for issues in the registry. Returns list of warnings."""
    warnings = []
    total_budget = get_total_budget_cents()
    if total_budget > 15000:  # $150 — more than typical account balance
        warnings.append(f"Total budget {total_budget}c exceeds $150 threshold")

    for name, family in MARKET_FAMILIES.items():
        if family.live_enabled and not family.calibration_path:
            warnings.append(f"{name}: live enabled but no calibration path")

    return warnings
