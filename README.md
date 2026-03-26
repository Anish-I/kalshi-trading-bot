# Kalshi Trading Bot

Automated trading system for [Kalshi](https://kalshi.com) prediction markets. Trades BTC 15-minute price direction and daily weather temperature markets using a multi-model voting system with adaptive weighting.

## Architecture

```
Coinbase WebSocket ──> Data Collector ──> 1m Bars ──> 32 Honest Features ──> 4-Model Voting ──> Kalshi Orders
     (real-time)        (5s/1m bars)     (parquet)    (candle-derived)       (weighted)          (API)

NWS + GFS Ensemble ──> Weather Forecast ──> Probability Model ──> Kalshi Weather Orders
  (31 members)           (27 cities)         (vs market price)       (edge > 15%)
```

## How It Works

### Crypto: 4-Model Voting System

Every 30 seconds, 4 independent models vote on BTC direction:

| Model | What It Sees | Type |
|---|---|---|
| **XGBoost** | 32 honest features from 1m bars | ML (retrained every 12h) |
| **Momentum** | 5m/10m returns, VWAP, Donchian, EMA slope | Rule-based |
| **Mean Reversion** | RSI/Stochastic/Donchian extremes | Rule-based |
| **Kalshi Consensus** | Orderbook depth + price velocity | Market signal |

**Voting rules:**
- Weighted score >= 2.0 (equivalent of 2 unweighted votes) → trade
- Opposing votes cancel out
- Confidence clamped to 95% max
- Bet sizing: score >= 3.0 → $15, score >= 2.0 → $7

**Adaptive weights** (Brier score + exponential decay):
- Models that predict well get higher vote weight (up to 2.0x)
- Models that predict poorly get demoted (down to 0.3x)
- Exponential decay (half-life ~14 trades) for regime adaptation
- Sample-size shrinkage prevents wild swings from few observations

### Weather: GFS Ensemble Forecasting

Trades daily high/low temperature markets for 27 city/temp combinations:
- 31-member GFS ensemble from Open-Meteo (free)
- NWS API for official forecasts
- P(above X) = count(members > X) / 31 — direct empirical probability
- Trades when model probability diverges from Kalshi market price by >15%
- Low-temp markets use `temperature_2m_min` ensemble (not max)

### 32 Honest Features

Trained on 13 days of Binance public historical data (18,720 1m bars). Only features we can honestly compute from OHLCV + buy/sell volume:

| Family | Count | Examples |
|---|---|---|
| Returns | 6 | 1m, 5m, 10m, 15m, 30m, 60m |
| EMA | 5 | EMA 9/21/50, slope, cross |
| Volatility | 4 | 5m/15m/60m realized vol, ratio |
| RSI/Stochastic | 3 | RSI-14, Stoch %K/%D |
| VWAP/ATR | 3 | VWAP deviation, ATR-14, ATR% |
| Donchian | 2 | Position, width |
| Volume | 3 | SMA ratio, taker imbalance, buy/sell ratio |
| Calendar | 6 | Sin/cos time, session flags |

### Trade Journal + Feedback Loop

Every decision is logged to `trade_journal.parquet`:
- All 4 model votes + confidences
- Feature snapshots at decision time
- Outcome tracking (win/loss/P&L)
- Adaptive weight recalculation on settlement

### Dashboard

Web UI served via Cloudflare Tunnel:
- Real-time 4-model vote display with colored indicators
- Balance, positions, recent trades with timestamps
- Bot controls (start/stop/restart)
- Trade notifications (browser push + in-page banner + sound)
- Auto-refreshes every 10 seconds

## Project Structure

```
kalshi-trading-bot/
├── config/settings.py              # All settings from .env
├── kalshi/                         # Kalshi API (RSA-PSS auth, REST client)
├── data/
│   ├── coinbase_auth_ws.py         # Authenticated WS (CDP key, EdDSA JWT)
│   ├── coinbase_ws.py              # Unauthenticated fallback + REST L2 polling
│   ├── bar_aggregator.py           # Trades → 5s/1m bars
│   └── storage.py                  # Parquet read/write
├── features/
│   ├── honest_features.py          # 32 real features from candles
│   └── pipeline.py                 # Legacy 83-feature pipeline
├── models/
│   ├── signal_models.py            # 4 models + weighted vote()
│   ├── trade_journal.py            # Persistent journal + Brier-score weights
│   └── xgboost_model.py            # XGBoost wrapper
├── weather/                        # Weather trading (27 cities)
│   ├── nws_client.py               # NWS API
│   ├── open_meteo_client.py        # 31-member GFS ensemble
│   ├── forecast_engine.py          # Probability distributions
│   ├── market_analyzer.py          # Find mispriced markets
│   └── trader.py                   # Execute weather trades
├── engine/                         # Risk management + position tracking
├── dashboard/                      # FastAPI + vanilla JS dashboard
├── scripts/
│   ├── crypto_ml_trader.py         # Main crypto trading bot (v3)
│   ├── collect_data.py             # Coinbase data collector
│   ├── weather_trade.py            # Weather trading bot
│   ├── download_binance_history.py # Fetch free historical CSVs
│   ├── train_honest.py             # Train on honest 32-feature set
│   ├── scheduled_retrain.py        # Auto-retrain every 12h
│   ├── optimize_thresholds.py      # Grid search for signal thresholds
│   └── start_dashboard.py          # Launch web dashboard
└── tests/                          # 36 tests (unit + integration + live API)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add: KALSHI_API_KEY_ID, RSA private key, Coinbase CDP key (optional)

# Download historical data + train model
python scripts/download_binance_history.py
python scripts/train_honest.py

# Start trading
python scripts/collect_data.py          # data collector (background)
python scripts/crypto_ml_trader.py      # crypto trader (background)
python scripts/weather_trade.py         # weather trader (background)
python scripts/start_dashboard.py       # web dashboard
cloudflared tunnel --url http://localhost:8050
```

## Key Design Decisions

1. **32 honest features, not 83 fake ones.** Dropped orderbook/L2/derivatives features that were computed from 1-level REST polling pretending to be real microstructure.

2. **Weighted voting, not single model.** XGBoost alone is 38% accurate (barely above random). The voting system adds value through selectivity — only trading when multiple uncorrelated signals agree.

3. **Brier-score weighting, not win-rate.** Models are weighted by calibrated probabilistic accuracy with exponential decay and sample-size shrinkage. Hot-hand bias is avoided.

4. **Settlement reconciliation.** Orders are only counted as positions when confirmed filled. Resting orders are polled. Overnight trades are reconciled before daily resets.

5. **Weather uses ensemble member counting.** P(above 58°F) = 28/31 members predict above 58°F = 90%. No Gaussian assumptions. Low-temp markets correctly use `temperature_2m_min`.

## Known Limitations

- XGBoost model accuracy is modest (38% on 3-class). Edge comes from voting selectivity + asymmetric payoffs, not model precision.
- No real L2 orderbook history. Coinbase L2 requires authenticated WebSocket with "View" scope.
- Binance is geo-blocked from the US. Historical data from data.binance.vision, live data from Coinbase.
- Weather backtests leak actual temperatures into forecasts (documented with warnings).
- 36 tests cover voting, journal, risk, and live API lifecycle, but not full end-to-end settlement math.

## Dependencies

- Python 3.11+, CUDA GPU (RTX 4070 Super for XGBoost training)
- Kalshi API key (RSA key pair)
- Coinbase CDP API key (optional, for authenticated WebSocket)
- cloudflared (for dashboard tunneling)
