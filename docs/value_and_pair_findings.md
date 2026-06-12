# Crypto 15m Markets — Win-Rate Diagnosis, Value Strategy, and the Both-Sides Idea

Findings and decisions from the 2026-06 investigation. Read this before
re-touching the directional ML leg or re-enabling pair trading.

## How these markets actually resolve

KXBTC15M ("BTC price up in next 15 mins?") is a **single-strike binary per
15-minute window**. It resolves **YES iff the settlement price (CF Benchmarks
BRTI, 60s average at close) ≥ the window's opening reference price**. That
reference is the market's `floor_strike` (`strike_type = greater_or_equal`,
shown as `yes_sub_title: "Target Price: $X"`).

- One market open per event at a time; windows are sequential. **No strike
  ladder within an event** → the "cross-strike ladder arbitrage" idea does **not
  apply** to KXBTC15M. (It would apply only to range-ladder markets, which this
  bot does not trade.)

## Why the win rate was low (38.55%, -$102/wk)

1. **Two trades, not a broken model.** -7374c and -3686c (both BTC_15M NO,
   26MAR25) = -$110.60; the other 273 trades were net slightly positive. This was
   a tail-risk / risk-control failure.
2. **The live trader had no EV gate.** `crypto_combined_trader` passed
   `calibration_artifact=None`; the gate silently "treated as pass." So the live
   ML leg traded raw conjunction at any price ≤45c on a hardcoded 48.5% win
   assumption the data (38.55%) refutes. → Fixed: `require_calibration` fails
   closed on live (Phase 1).
3. **Risk state was in-memory + entry-only.** Daily-loss/consecutive counters
   reset on restart and nothing capped per-ticker exposure. → Fixed: persistent
   `TickerExposureTracker` (10 contracts / $15 per ticker) + persistent
   `RiskManager` (Phase 1).
4. **The ML leg never compared spot to the strike.** It bet up/down from
   momentum, but YES means "close ≥ open price." That mismatch is the core
   modeling error the value strategy fixes (Phase 2).

## The both-sides idea — verdict

**Taker "buy both sides" is mathematically impossible on one market.** From the
order book, `taker_pair_cost = (100 − best_no_bid) + (100 − best_yes_bid) =
200 − yes_bid − no_bid ≥ 100¢` always (since `yes_bid + no_bid ≤ 100` on any
uncrossed book). After fees it is strictly negative. Do **not** rebuild taker
arb. (Proven in `engine/pair_pricing.py`; the old feasibility replay measured
exactly this taker cost, which is why it never found an opportunity.)

**Maker spread-capture is the only real version.** Post resting limits at
`best_bid + 1` on both legs; profit = `100 − maker_pair_cost − fees` **only if
both legs fill**. Economics with the corrected fee model:

- Real Kalshi maker fee = **25% of taker** = `0.25 × 0.07 × P × (1−P)` per
  contract, ceil-to-cent (`engine/fees.py`). Near mid that's ~1c/leg → ~2c/pair
  (similar to the old flat 2.14c estimate — **fees were never the blocker**).
- The blocker is the **orphan rate**: on BTC/ETH (1–3c spreads) one leg fills and
  the other doesn't 80–100% of the time, leaving directional exposure that loses
  more than the spread captured. Net negative. Pairs are correctly **disabled**
  (`PAIR_ENABLED_SERIES = set()`).

**Go/no-go to enable maker pairs** (all required):
1. Wide-spread series only (≥ ~5c, e.g. HYPE/DOGE).
2. Ship the orphan-unwind state machine (`ORPHAN_UNWIND_STATE_MACHINE_SHIPPED`
   is `False`) — cancel/῾hedge the unfilled leg on a timeout or adverse move.
3. Measure, in **paper**, per-series orphan rate and maker-net-after-fees; enable
   live only where maker-net (incl. orphan losses) is positive.

Until 1–3 hold, the better expression of "buy the cheap side" is the **one-sided
value bet** below.

## The value strategy (Phase 2) — what to run instead

Price the binary directly and bet whichever side is cheap vs fair:

    P(YES) = Φ( ln(spot / strike) / (σ · √T) )      # drift ≈ 0 over 15 min

`spot` = live BTC, `strike` = `floor_strike` (window open), `T` = minutes to
close, `σ` = per-minute log-return vol from 1m bars. Take YES if
`fair_P·100 − yes_ask − taker_fee ≥ min_edge`, NO symmetrically; else pass.

- **Model is calibrated** (synthetic GBM backtest): Brier **0.15 vs 0.25**
  coin-flip, directional accuracy 0.77, reliability on the diagonal.
- **Live smoke test** found a real mispricing: spot $61,141 vs strike $60,941,
  7.9m left → fair P(YES)=0.928 while the book offered YES at 80c → **+10.8c/
  contract** net edge. The edge concentrates in the final minutes, where
  spot-vs-strike is nearly decided but the book lags.
- Tools: `scripts/backtest_value_strategy.py` (run `--bars` on the Windows data
  box to get the real-data Brier), `scripts/crypto_value_trader.py` (sim-first;
  every order funnels through the Phase-1 gate + per-ticker cap).

## 2026-06-11 — model fixes + adaptive strategy (validated on 90d real bars)

Backtest on 90d of real BTCUSDT 1m bars (Binance, 2026-03-13..06-10; 103,680
samples) surfaced three discrepancies that were depressing the numbers:

1. **T off-by-one (backtest)**: deciding at the close of bar `elapsed` means the
   price is at minute `elapsed+1`, so T = 14−elapsed, not 15−elapsed. Worst in
   the final minutes where the edge lives.
2. **Strike off-by-one (backtest)**: the market strike is the window OPEN
   (minute 0) = the first bar's `open`, not its close (minute 1).
3. **Vol estimator**: trailing 15-sample std is noisy and lags vol clustering;
   and raw 1m bar vol carries microstructure noise that doesn't propagate to
   settlement, making fair_p underconfident. → EWMA (λ=0.94) ×
   `BTC15M_VOL_SCALE = 0.90` (fitted on first 60d, validated on last 30d).

Result on identical decision points (true T = 13..2): **Brier 0.1641 → 0.1564,
directional accuracy 0.752 → 0.763**, reliability on the diagonal in every
decile. The live estimator (`calibrated_sigma_per_min`) now matches the
backtest exactly.

**Adaptive edge threshold** (the strategy improvement): per-T Brier runs 0.08
(T=2m) to 0.22 (T=13m), so a flat min-edge over-trades the fuzzy early window.
`required = base + 0.5c × minutes_to_close` (engine/value_strategy.py,
default-on). Out-of-sample (last 30d, one-entry-per-window replay, 2c spread,
1m-stale book): **+13.5c/trade vs +9.1c flat, win 65%, max drawdown ~5–6$**.

The old per-sample PnL sim books up to 12 trades/window — impossible live under
the per-ticker cap. Use `simulate_one_trade_per_window` ("HONEST replay" in the
report) for sizing expectations: ~93 windows/day traded, lag=1m conservative
assumption → ~+$12/day at 1 contract. Efficient-book sanity (lag=0) finds zero
trades — the strategy does not hallucinate edge.

## Validate-before-live checklist

1. `python -m scripts.backtest_value_strategy --bars` on Windows → confirm real
   BTC Brier beats 0.25 and reliability holds (real returns have fatter tails
   than GBM; if mid-bucket calibration drifts, widen `min_edge_cents`).
2. Run `crypto_value_trader` in sim for a session; confirm logged `fairP`/`ev`
   look sane and the gate/ticker-cap engage.
3. Graduate to `--mode live --contracts 1` with the hard per-ticker cap; compare
   realized win rate to the break-even implied by entry price + verified fees.

## Open follow-ups

- Crypto fee multiplier defaults to 0.07 (conservative); confirm vs the official
  PDF when reachable and change `CRYPTO_MULTIPLIER` in `engine/fees.py` only.
- Scorecard `trade_count=0` reconciliation (observability; shadow mode).
- Orphan-unwind state machine (prerequisite for any maker-pair revival).
