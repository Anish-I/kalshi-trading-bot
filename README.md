# Kalshi Trading Bot

Automated trading system for [Kalshi](https://kalshi.com) prediction markets. Trades BTC 15-minute price direction and daily weather temperature markets using ML models, rule-based signals, and market microstructure analysis.

**Built from scratch in one session.** Started with $50, deployed crypto + weather strategies, built a web dashboard, and implemented a multi-model voting system.

## Architecture

```
Coinbase WebSocket ──> Data Collector ──> 83 Features ──> 4-Model Voting ──> Kalshi Orders
     (real-time)        (5s bars)         (parquet)        (consensus)       (API)

NWS + GFS Ensemble ──> Weather Forecast ──> Probability Model ──> Kalshi Weather Orders
  (31 members)           (27 cities)         (vs market price)       (edge > 15%)
```

## What It Does

### Crypto Trading (BTC 15-Minute Markets)

Every 30 seconds, the bot:

1. **Collects data** from Coinbase authenticated WebSocket (~21 trades/sec)
2. **Computes 83 features** across 8 families (microstructure, depth, trade flow, LOB dynamics, technical indicators, multi-timeframe, derivatives, calendar)
3. **Runs 4 independent models** that each vote UP, DOWN, or FLAT:
   - **XGBoost**: Gradient-boosted tree classifier trained on microstructure features (auto-retrains every 12h on GPU)
   - **Momentum**: Pure price action signals — 5m/10m returns, VWAP deviation, Donchian position, EMA slope
   - **Mean Reversion**: RSI/Stochastic/Donchian extremes that suggest bounces
   - **Kalshi Consensus**: The Kalshi orderbook depth imbalance + price velocity — what other real-money traders think
4. **Votes**: Only trades when 2+ models agree with 0 opposing
   - 3-4 agree → $15 bet (full conviction)
   - 2 agree, 0 oppose → $7 bet (moderate)
   - Anything else → no trade
5. **Places limit orders** on Kalshi's KXBTC15M binary markets (YES = BTC goes up, NO = BTC goes down)
6. **Tracks settlements** and updates running P&L

### Weather Trading (Daily High/Low Temperature Markets)

The weather bot:

1. **Fetches forecasts** from NWS API + Open-Meteo 31-member GFS ensemble for 27 city/temperature combinations (NYC, LA, Chicago, Miami, Phoenix, Atlanta, San Francisco, Las Vegas, Austin, Denver, Dallas, Seattle, DC, Minneapolis, Houston, San Antonio, Boston, New Orleans, Philadelphia, Oklahoma City — both highs and lows)
2. **Computes probability distributions** using ensemble member counting (e.g., 28/31 members predict above 58°F = 90% probability)
3. **Scans Kalshi weather markets** for mispricings — compares model probability to market price
4. **Trades when edge > 15%** — if our model says 90% but Kalshi prices it at 50%, that's a 40% edge
5. **Settles next morning** when NWS publishes the Daily Climate Report

## Model Details

### XGBoost (83 Features)

Trained on GPU (RTX 4070 Super) with walk-forward validation:

| Feature Family | Count | Examples |
|---|---|---|
| Microstructure | 8 | Mid-price, spread, microprice, top imbalance |
| Depth | 12 | Bid/ask depth at L5/L10/L20, book imbalance, slope |
| Trade Flow | 12 | Taker imbalance, cumulative volume delta, trade intensity |
| LOB Dynamics | 8 | Cancel rate, bid/ask retreat, spread volatility |
| Technical | 17 | EMA, RSI, VWAP, ATR, Donchian, Stochastic |
| Multi-Timeframe | 11 | RSI/EMA on 15m/1h/4h bars |
| Derivatives | 6 | Funding rate, basis proxy, open interest |
| Calendar | 8 | Sin/cos time-of-day, session flags |

Auto-retrains every 12 hours via Windows Scheduled Task. Model saved to `D:/kalshi-models/latest_model.json`.

### Signal Models (Rule-Based)

Thresholds optimized via grid search on collected data:

- **Momentum**: Requires 4/5 signals aligned (returns, VWAP, Donchian, EMA)
- **Mean Reversion**: RSI < 25 or > 80, Donchian position extremes, Stochastic < 10 or > 90
- **Kalshi Consensus**: Orderbook depth imbalance > 0.3, price velocity > 5c/min

### Weather Forecast Model

Uses the same approach as the [most profitable public Kalshi weather bot](https://github.com/suislanchez/polymarket-kalshi-weather-bot) ($1.3k profit):

- 31-member GFS ensemble from Open-Meteo (free, no API key)
- NWS API for official forecasts (matches Kalshi settlement source)
- Probability = fraction of ensemble members above/below threshold
- No Gaussian assumptions — direct empirical distribution

## Dashboard

Web-based dashboard served via Cloudflare Tunnel:

- Real-time balance, positions, recent trades
- **Model Thinking** section: shows all 4 model votes with colored indicators (green=UP, red=DOWN, gray=FLAT), agreement count, combined confidence
- Bot controls: Start/Stop/Restart buttons for each process
- Auto-refreshes every 10 seconds

Start: `python scripts/start_dashboard.py` then `cloudflared tunnel --url http://localhost:8050`

## Project Structure

```
kalshi-trading-bot/
├── config/settings.py           # All settings from .env
├── kalshi/                      # Kalshi API (RSA-PSS auth, REST client)
├── data/                        # Coinbase WebSocket + bar aggregation
│   ├── coinbase_auth_ws.py      #   Authenticated WS (CDP key, EdDSA JWT)
│   ├── coinbase_ws.py           #   Unauthenticated fallback
│   ├── bar_aggregator.py        #   Trades → 5s/1m bars
│   └── storage.py               #   Parquet read/write
├── features/                    # 83 microstructure features
├── models/
│   ├── signal_models.py         #   Momentum, MeanReversion, KalshiConsensus, vote()
│   ├── xgboost_model.py         #   XGBoost wrapper
│   └── transformer_model.py     #   TinyTransformer (unused, for future)
├── weather/                     # Weather trading (27 cities)
│   ├── nws_client.py            #   NWS API
│   ├── open_meteo_client.py     #   31-member GFS ensemble
│   ├── forecast_engine.py       #   Probability distributions
│   ├── market_analyzer.py       #   Find mispriced markets
│   └── trader.py                #   Execute weather trades
├── engine/                      # Risk management + position tracking
├── dashboard/                   # FastAPI + vanilla JS dashboard
├── monitoring/                  # Metrics + drift detection
└── scripts/
    ├── crypto_ml_trader.py      #   Main crypto trading bot (v3)
    ├── collect_data.py          #   Coinbase data collector
    ├── weather_trade.py         #   Weather trading bot
    ├── scheduled_retrain.py     #   Auto-retrain every 12h
    ├── optimize_thresholds.py   #   Grid search for signal thresholds
    ├── start_dashboard.py       #   Launch web dashboard
    └── backfill_train_backtest.py  # Historical backtest
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure .env (copy from .env.example)
cp .env.example .env
# Add: KALSHI_API_KEY_ID, Kalshi private key, Coinbase CDP key

# Start data collector
python scripts/collect_data.py

# Start crypto trader
python scripts/crypto_ml_trader.py

# Start weather trader
python scripts/weather_trade.py

# Start dashboard
python scripts/start_dashboard.py
cloudflared tunnel --url http://localhost:8050
```

## Key Learnings

1. **Binance is geo-blocked from the US** (HTTP 451). Use Coinbase instead.
2. **Coinbase L2 WebSocket requires API key with "View" scope** on Advanced Trade. Without it, you only get ticker + trades.
3. **Kalshi weather markets settle on NWS Daily Climate Report** from specific stations (Central Park for NYC, LAX for LA, Midway for Chicago).
4. **Kalshi market tickers**: `KXBTC15M-26MAR251000-00` (BTC 15-min), `KXHIGHNY-26MAR25-B55.5` (NYC high temp bracket 55-56F), `KXHIGHNY-26MAR25-T58` (above 58F).
5. **Kalshi API auth**: RSA-PSS signing with SHA256. Sign `{timestamp_ms}{METHOD}{path}`.
6. **The ML model alone (48% accuracy) is not the edge.** The edge comes from combining weak signals + asymmetric payoffs + selectivity.
7. **Weather trading has the best edge** on Kalshi. NWS forecasts are genuinely more accurate than market prices. The crypto 15-min market is much more efficient.

## Dependencies

- Python 3.11+
- CUDA GPU (RTX 4070 Super used for XGBoost training)
- Kalshi API key (RSA key pair)
- Coinbase CDP API key (optional, for authenticated WebSocket)
- cloudflared (for dashboard tunneling)
