# Kalshi Trading Bot — Overhaul Verification Report

**Date:** 2026-04-08
**Branch:** master
**Verification:** read-only, Phase 9 of `twinkly-knitting-blum.md`

## Phases Shipped

| Phase | Description | Commit |
|-------|-------------|--------|
| 1  | Unified pre-trade gate module + tests | `50a4d69` |
| 1b | Funnel 7 order sites through PreTradeGate (sim-safe, additive) | `db4cdfc` |
| fix | pair-pricing: enforce PAIR_MIN_NET via min_maker_net | `10736e4` |
| fix | orphan-handler: flatten filled leg on pair orphan | `cc56145` |
| fix | fee-schedule: flag PAIR_FEE_CENTS for manual verification | `3b57d28` |
| 2  | Family scorecards + throttle (shadow mode) | `39596cd` |
| 3  | Dashboard `/api/families` + `/api/gate/recent` | `55a70cb` |
| 4  | Archive Kalshi trade history + `load_trade_archive` | `b7ad645` |
| 5  | Calibration `--use-kalshi-archive` | `0d7a0f8` |
| 6  | Optional macro crypto features (default off, no-leakage tests) | `454fda4` |
| 7  | `--features` flag on train/calibrate + ablation runner | `344686d` |
| 8  | Offline pair microstructure report | `77d10f1` |

## Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| Unified pre-trade gate module with tests | ✅ | `engine/pre_trade_gate.py` + tests green |
| Gate funneled into live order sites | ⚠ | 7 sites funneled with permissive defaults; sim path of pair trader NOT gated (only live `_place_real_pair`) |
| pair_pricing respects `PAIR_MIN_NET` | ✅ | `min_maker_net` parameter threaded |
| Orphan leg flatten w/ slippage cap | ⚠ | Tests cover a copied snippet, not the real code path (import-time side effects) |
| Kalshi fee schedule verified | ❌ | 429 rate-limited at fetch; `PAIR_FEE_CENTS` flagged TODO, needs manual confirmation |
| Family scorecards (shadow) + throttle injection | ✅ | `engine/family_scorecard.py`, shadow mode on |
| Dashboard endpoints | ✅ | `/api/families` and `/api/gate/recent` both return 200 via TestClient |
| Kalshi trade backfill + archive loader | ✅ | `scripts/backfill_kalshi_trades.py`, `engine/market_archive.py:load_trade_archive` |
| Calibration from Kalshi ground truth | ✅ | `--use-kalshi-archive` flag wired |
| Optional macro features + no-leakage test | ✅ | Default off; toggle `MACRO_FEATURES_ENABLED` |
| `--features {honest,honest_plus_macro}` on train+calibrate | ✅ | Present in both `--help` |
| Ablation runner | ✅ | `scripts/run_ablation.py` |
| Offline pair microstructure report | ✅ | `scripts/pair_microstructure_report.py --help` OK |
| All existing tests still pass | ⚠ | 115 passed, 3 pre-existing weather tracker failures unrelated to overhaul |

## Test Results

```
python -m pytest tests/ -q --ignore=tests/test_weather_tracker.py
  113 passed in 5.42s

python -m pytest tests/ -q
  115 passed, 3 failed  (failures all in tests/test_weather_tracker.py,
                         pre-existing, unrelated to this overhaul:
    - test_execute_trades_tracks_status_and_skips_resting_duplicate
    - test_weather_tier_sizing_does_not_mutate_following_orders
    - test_weather_trader_uses_recovery_consecutive_loss_halt)
```

## py_compile Results

All 20 touched files compiled cleanly:

```
engine/pre_trade_gate.py engine/family_scorecard.py engine/family_limits.py
engine/pair_pricing.py engine/market_archive.py engine/crypto_decision.py
features/macro_crypto_features.py features/honest_features.py
models/trade_journal.py models/signal_models.py
scripts/crypto_ml_trader.py scripts/crypto_combined_trader.py scripts/crypto_pair_trader.py
scripts/backfill_kalshi_trades.py scripts/build_crypto_calibration.py
scripts/train_honest.py scripts/run_ablation.py scripts/pair_microstructure_report.py
scripts/update_macro_crypto_data.py dashboard/app.py
=> ALL COMPILED OK
```

## Dashboard Smoke Test (TestClient)

`/api/families` returned `shadow_mode: True`, window 48h, 7 families all `healthy: True` score 1.0.
`/api/gate/recent` returned state for all three traders with `gate_last_decision: None` (no trades logged through gate yet — expected, sim hasn't persisted a gate decision into state).

## State JSON `gate_last_decision`

```
combined_trader_state.json: gate_last_decision = None
ml_trader_state.json:       gate_last_decision = None
pair_trader_state.json:     gate_last_decision = None
```

All three fields present in schema but unpopulated. This means **no real trader run has ever persisted a gate decision** — the field is wired but unexercised end-to-end.

## Live Status — NO live thresholds were changed by this overhaul

| Setting | Value | Source |
|---|---|---|
| `SCORECARD_SHADOW_MODE` | `True` | `config/settings.py` — gate is permissive, no blocking |
| `MACRO_FEATURES_ENABLED` | `False` | `config/settings.py` — macro features off |
| `PAIR_ENABLED_SERIES` | `set()` (empty) | `scripts/crypto_combined_trader.py:74` — no pairs |
| `ML_MAX_CONTRACTS` | `10` | `scripts/crypto_combined_trader.py:64` |
| `ML_MAX_ENTRY_PRICE` | `0.45` | `scripts/crypto_combined_trader.py:65` |
| `ML_MIN_EDGE` | `0.03` | `scripts/crypto_combined_trader.py:66` |
| `PAIR_MIN_NET` | `5.0` | `scripts/crypto_combined_trader.py:67` |
| `MAX_DRAWDOWN` | `2900` ($29) | `scripts/crypto_combined_trader.py:156` |
| `TRAILING_ACTIVATION` | `500` ($5) | `scripts/crypto_combined_trader.py:161` |

All values match pre-Phase-1 state. **No live thresholds were changed.**

## Sim Smoke Test

`python scripts/crypto_combined_trader.py --mode sim` ran for 25s, completed ≥2 scan cycles cleanly across all 6 crypto series (BTC/ETH/SOL/XRP/HYPE/DOGE), schema/model loaded, orderbooks fetched. No Phase 1b regression — gate is a no-op but doesn't break startup or the scan loop.

## Known Issues / TODOs

1. **Orphan flatten tests cover a copied snippet, not the real code path** due to import-time side effects on the pair trader module.
2. **Kalshi fee schedule fetch returned 429** — `PAIR_FEE_CENTS` flagged TODO, needs manual verification from the docs.
3. **Phase 1b gate is a no-op** for blocking: RiskManager / FamilyLimits injected permissively. A Phase 1c (not in plan, deferred) would tighten this.
4. **8 historical ML LIVE trades in log** — too few to compute a meaningful realized win-rate.
5. **Pair trader SIM path not funneled through gate** — only the live `_place_real_pair` call is gated.
6. **`gate_last_decision` never populated** in any state JSON — end-to-end write path unexercised.

## Next Steps for Going Live Conservatively

1. [ ] Manually read the Kalshi fee schedule and fix `PAIR_FEE_CENTS` constant (429 blocked auto-fetch).
2. [ ] Run `python scripts/backfill_kalshi_trades.py --days 30` against prod (needs PEM auth).
3. [ ] Run `python scripts/build_crypto_calibration.py --use-kalshi-archive` to compare Kalshi-derived calibration vs Binance replay.
4. [ ] Run `python scripts/pair_microstructure_report.py --days 30` and review spreads/fills.
5. [ ] Decide on conservative ramp S0 → S1 (BTC, 1 contract, $2 daily loss limit) per earlier table.
6. [ ] Flip `SCORECARD_SHADOW_MODE=False` only after 48h of clean shadow data showing `gate_last_decision` is actually being populated.
7. [ ] Build the option-D one-sided maker pair executor before re-enabling pair trading on any series.
