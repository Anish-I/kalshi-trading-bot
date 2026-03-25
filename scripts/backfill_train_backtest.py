"""
End-to-end pipeline: Backfill historical data → Compute features → Train XGBoost →
Walk-forward backtest with 1000+ instances → Simulate Kalshi P&L.

Usage: python scripts/backfill_train_backtest.py
"""

import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from config.settings import settings
from features.technical import TechnicalFeatures
from features.calendar import CalendarFeatures
from features.labeler import Labeler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# 1. BACKFILL — download 14 days of 1m klines from Binance via ccxt
# ---------------------------------------------------------------------------

def fetch_all_klines(days: int = 14) -> pd.DataFrame:
    """Fetch `days` of 1-minute BTC/USD klines from CryptoCompare (no geo-block)."""
    import httpx

    now = datetime.now(timezone.utc)
    chunks: list[pd.DataFrame] = []
    total = 0

    # CryptoCompare histominute: max 2000 per request, paginate backwards from `toTs`
    to_ts = int(now.timestamp())
    target_start = int((now - timedelta(days=days)).timestamp())

    while to_ts > target_start:
        log.info(
            "Fetching klines ending at %s  (%d rows so far)",
            datetime.fromtimestamp(to_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
            total,
        )
        try:
            resp = httpx.get(
                "https://min-api.cryptocompare.com/data/v2/histominute",
                params={"fsym": "BTC", "tsym": "USD", "limit": 2000, "toTs": to_ts},
                timeout=30,
            )
            data = resp.json()
        except Exception as e:
            log.warning("Fetch error: %s — retrying in 2s", str(e)[:80])
            time.sleep(2)
            continue

        if data.get("Response") != "Success" or not data.get("Data", {}).get("Data"):
            log.warning("CryptoCompare error: %s", data.get("Message", "unknown"))
            break

        bars = data["Data"]["Data"]
        df = pd.DataFrame(bars)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"volumefrom": "volume"})
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        # Remove zero-volume bars (empty periods)
        df = df[df["volume"] > 0]

        if df.empty:
            break

        chunks.append(df)
        total += len(df)

        # Move backwards
        to_ts = int(df["timestamp"].iloc[0].timestamp()) - 1

        if to_ts <= target_start:
            break

        time.sleep(0.3)

    if not chunks:
        return pd.DataFrame()

    result = (
        pd.concat(chunks, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    log.info("Backfill complete: %d 1-minute bars over %d days", len(result), days)
    return result


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING — vectorized over the full history
# ---------------------------------------------------------------------------

def compute_all_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute ~40 features from 1m bars using vectorized pandas operations.

    We skip orderbook/microstructure features (no historical L2 data)
    and focus on technical + calendar + momentum features that can be
    computed from candles alone.
    """
    df = bars.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    ts = df["timestamp"]

    # --- EMAs ---
    df["ema_9"] = close.ewm(span=9, adjust=False).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["ema_9_slope"] = df["ema_9"].pct_change()
    df["ema_21_slope"] = df["ema_21"].pct_change()
    df["ema_cross"] = df["ema_9"] - df["ema_21"]
    df["ema_cross_pct"] = df["ema_cross"] / close

    # --- VWAP ---
    cum_vol = volume.cumsum()
    cum_pv = (close * volume).cumsum()
    df["vwap"] = cum_pv / cum_vol.replace(0, np.nan)
    df["vwap_deviation"] = (close - df["vwap"]) / df["vwap"]

    # --- ATR ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / close

    # --- RSI ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss_s = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss_s.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # --- Stochastic ---
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    denom = (high_14 - low_14).replace(0, np.nan)
    df["stoch_k"] = 100 * (close - low_14) / denom
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # --- Bollinger Bands ---
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_position"] = (close - bb_lower) / bb_width
    df["bb_width_pct"] = bb_width / sma_20

    # --- Donchian ---
    df["donchian_high_20"] = high.rolling(20).max()
    df["donchian_low_20"] = low.rolling(20).min()
    dc_range = (df["donchian_high_20"] - df["donchian_low_20"]).replace(0, np.nan)
    df["donchian_position"] = (close - df["donchian_low_20"]) / dc_range

    # --- Volume features ---
    df["volume_sma_ratio"] = volume / volume.rolling(20).mean().replace(0, np.nan)
    df["volume_std_ratio"] = volume / volume.rolling(20).std().replace(0, np.nan)

    # --- Momentum / returns ---
    df["ret_1m"] = close.pct_change(1)
    df["ret_5m"] = close.pct_change(5)
    df["ret_10m"] = close.pct_change(10)
    df["ret_15m"] = close.pct_change(15)
    df["ret_30m"] = close.pct_change(30)
    df["ret_60m"] = close.pct_change(60)

    # --- Volatility ---
    df["volatility_5m"] = df["ret_1m"].rolling(5).std()
    df["volatility_15m"] = df["ret_1m"].rolling(15).std()
    df["volatility_60m"] = df["ret_1m"].rolling(60).std()
    df["vol_ratio_5_60"] = df["volatility_5m"] / df["volatility_60m"].replace(0, np.nan)

    # --- Multi-timeframe RSI ---
    close_15m = close.rolling(15).apply(lambda x: x.iloc[-1], raw=False)
    delta_15m = close_15m.diff()
    gain_15m = delta_15m.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss_15m = (-delta_15m).clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs_15m = gain_15m / loss_15m.replace(0, np.nan)
    df["rsi_15m"] = 100 - 100 / (1 + rs_15m)

    # --- Rate of change ---
    df["roc_5"] = close / close.shift(5) - 1
    df["roc_10"] = close / close.shift(10) - 1
    df["roc_20"] = close / close.shift(20) - 1

    # --- Calendar features (vectorized) ---
    hour = ts.dt.hour + ts.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    df["is_us_session"] = ((ts.dt.hour >= 14) & (ts.dt.hour < 21)).astype(float)
    df["is_asia_session"] = (ts.dt.hour < 7).astype(float)

    # --- Higher TF trend ---
    ema_60 = close.ewm(span=60, adjust=False).mean()
    ema_240 = close.ewm(span=240, adjust=False).mean()
    df["trend_1h"] = np.sign(ema_60.diff())
    df["trend_4h"] = np.sign(ema_240.diff())
    df["higher_tf_trend"] = df["trend_1h"] + df["trend_4h"]

    # Identify feature columns (exclude raw OHLCV + timestamp)
    raw_cols = {"timestamp", "open", "high", "low", "close", "volume", "vwap"}
    feature_cols = [c for c in df.columns if c not in raw_cols]

    log.info("Computed %d features", len(feature_cols))
    return df


# ---------------------------------------------------------------------------
# 3. LABELING
# ---------------------------------------------------------------------------

def label_data(bars: pd.DataFrame, horizon: int = 15, threshold_bps: float = 3.0) -> pd.Series:
    """Create 3-class direction labels: +1 (up), 0 (flat), -1 (down)."""
    close = bars["close"].astype(float)
    fwd_ret = np.log(close.shift(-horizon) / close)
    thr = threshold_bps * 1e-4
    labels = pd.Series(
        np.where(fwd_ret > thr, 1, np.where(fwd_ret < -thr, -1, 0)),
        index=bars.index,
    )
    # Last `horizon` rows have no label
    labels.iloc[-horizon:] = np.nan
    return labels


# ---------------------------------------------------------------------------
# 4. TRAINING + WALK-FORWARD BACKTEST
# ---------------------------------------------------------------------------

def run_training_and_backtest(
    features: pd.DataFrame,
    labels: pd.Series,
    feature_cols: list[str],
    n_splits: int = 5,
    purge_bars: int = 15,
) -> dict:
    """Walk-forward train/test with purge gap. Returns detailed results."""
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report, log_loss, accuracy_score

    X = features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = labels.astype(int)
    y_mc = y + 1  # {-1,0,1} -> {0,1,2}

    n = len(X)
    fold_size = n // (n_splits + 1)

    all_test_preds = []
    all_test_proba = []
    all_test_true = []
    all_test_idx = []
    fold_reports = []

    for fold in range(n_splits):
        train_end = fold_size * (fold + 2)
        test_start = train_end + purge_bars
        test_end = min(train_end + fold_size, n)

        if test_start >= test_end:
            continue

        X_tr = X.iloc[:train_end]
        y_tr = y_mc.iloc[:train_end]
        X_te = X.iloc[test_start:test_end]
        y_te = y_mc.iloc[test_start:test_end]

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            max_depth=6,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=50,
            tree_method="hist",
            device="cuda",
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        proba = model.predict_proba(X_te)

        acc = accuracy_score(y_te, preds)
        try:
            ll = log_loss(y_te, proba, labels=[0, 1, 2])
        except ValueError:
            ll = float("nan")

        fold_reports.append({
            "fold": fold,
            "train_size": len(X_tr),
            "test_size": len(X_te),
            "accuracy": acc,
            "log_loss": ll,
        })
        log.info(
            "Fold %d: train=%d test=%d accuracy=%.4f log_loss=%.4f",
            fold, len(X_tr), len(X_te), acc, ll,
        )

        all_test_preds.extend(preds)
        all_test_proba.extend(proba)
        all_test_true.extend(y_te.values)
        all_test_idx.extend(range(test_start, test_end))

    all_test_preds = np.array(all_test_preds)
    all_test_proba = np.array(all_test_proba)
    all_test_true = np.array(all_test_true)

    # Overall classification report
    report = classification_report(
        all_test_true, all_test_preds,
        target_names=["down", "flat", "up"],
        digits=4,
    )

    # Directional accuracy (excluding flat predictions)
    directional_mask = all_test_preds != 1  # not flat
    if directional_mask.sum() > 0:
        dir_acc = accuracy_score(all_test_true[directional_mask], all_test_preds[directional_mask])
        dir_count = directional_mask.sum()
    else:
        dir_acc = 0.0
        dir_count = 0

    # Train final model on all data
    log.info("Training final model on full dataset (%d samples)...", len(X))
    final_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        tree_method="hist",
        device="cuda",
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    final_model.fit(X, y_mc)

    # Feature importance
    importance = pd.Series(
        final_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    return {
        "fold_reports": fold_reports,
        "classification_report": report,
        "overall_accuracy": accuracy_score(all_test_true, all_test_preds),
        "directional_accuracy": dir_acc,
        "directional_trades": dir_count,
        "total_test_instances": len(all_test_true),
        "test_preds": all_test_preds,
        "test_proba": all_test_proba,
        "test_true": all_test_true,
        "test_idx": all_test_idx,
        "feature_importance": importance,
        "final_model": final_model,
    }


# ---------------------------------------------------------------------------
# 5. KALSHI P&L SIMULATION
# ---------------------------------------------------------------------------

def simulate_kalshi_pnl(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_proba: np.ndarray,
    confidence_threshold: float = 0.55,
    fee_cents: int = 0,
) -> dict:
    """Simulate Kalshi binary contract trading P&L.

    For each prediction:
    - If model predicts UP with confidence > threshold:
      Buy YES at (1 - confidence) * 100 cents
      If correct: profit = 100 - entry_price - fee
      If wrong: loss = -entry_price - fee
    - Same logic for DOWN (buy NO)
    """
    trades = []
    running_pnl = 0
    peak_pnl = 0
    max_dd = 0

    for i in range(len(pred_labels)):
        pred_class = pred_labels[i]  # 0=down, 1=flat, 2=up
        true_class = true_labels[i]

        # Skip flat predictions
        if pred_class == 1:
            continue

        # Get confidence for the predicted direction
        confidence = pred_proba[i][pred_class]
        if confidence < confidence_threshold:
            continue

        # Entry price (in cents) — we pay the implied probability
        entry_cents = int(confidence * 100)

        # Determine if correct
        if pred_class == 2:  # predicted UP
            correct = true_class == 2  # actual UP
        else:  # pred_class == 0, predicted DOWN
            correct = true_class == 0  # actual DOWN

        if correct:
            pnl = 100 - entry_cents - fee_cents
        else:
            pnl = -entry_cents - fee_cents

        running_pnl += pnl
        peak_pnl = max(peak_pnl, running_pnl)
        dd = peak_pnl - running_pnl
        max_dd = max(max_dd, dd)

        trades.append({
            "index": i,
            "pred": "UP" if pred_class == 2 else "DOWN",
            "true": ["DOWN", "FLAT", "UP"][true_class],
            "confidence": confidence,
            "entry_cents": entry_cents,
            "pnl_cents": pnl,
            "running_pnl": running_pnl,
            "correct": correct,
        })

    if not trades:
        return {"n_trades": 0, "total_pnl": 0}

    trades_df = pd.DataFrame(trades)
    wins = trades_df["correct"].sum()
    n = len(trades_df)

    returns = trades_df["pnl_cents"].values.astype(float)
    sharpe = (returns.mean() / returns.std() * np.sqrt(96)) if returns.std() > 0 else 0.0

    return {
        "n_trades": n,
        "wins": int(wins),
        "losses": n - int(wins),
        "win_rate": wins / n,
        "total_pnl_cents": int(running_pnl),
        "total_pnl_dollars": running_pnl / 100,
        "avg_pnl_per_trade_cents": returns.mean(),
        "max_drawdown_cents": int(max_dd),
        "sharpe_ratio": sharpe,
        "trades_df": trades_df,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("BACKFILL → TRAIN → BACKTEST PIPELINE")
    log.info("=" * 60)

    # --- Step 1: Backfill ---
    log.info("\n[1/5] Downloading 14 days of BTC/USDT 1m klines from Binance...")
    bars = fetch_all_klines(days=14)
    if bars.empty or len(bars) < 1000:
        log.error("Not enough data fetched (%d bars). Aborting.", len(bars))
        return
    log.info("Got %d bars from %s to %s", len(bars), bars["timestamp"].iloc[0], bars["timestamp"].iloc[-1])

    # --- Step 2: Features ---
    log.info("\n[2/5] Computing features...")
    featured = compute_all_features(bars)

    raw_cols = {"timestamp", "open", "high", "low", "close", "volume", "vwap"}
    feature_cols = [c for c in featured.columns if c not in raw_cols]
    log.info("Feature columns: %d", len(feature_cols))

    # --- Step 3: Labels ---
    log.info("\n[3/5] Labeling (15-min horizon, 3bps threshold)...")
    labels = label_data(bars, horizon=15, threshold_bps=3.0)

    # Drop warmup rows (first 240 for indicator convergence + last 15 for labels)
    warmup = 240
    valid_mask = (~labels.isna()) & (featured.index >= warmup)
    featured_valid = featured.loc[valid_mask].reset_index(drop=True)
    labels_valid = labels.loc[valid_mask].reset_index(drop=True)

    log.info(
        "Valid samples: %d\nLabel distribution:\n  UP (+1): %d\n  FLAT (0): %d\n  DOWN (-1): %d",
        len(labels_valid),
        (labels_valid == 1).sum(),
        (labels_valid == 0).sum(),
        (labels_valid == -1).sum(),
    )

    # --- Step 4: Train + Walk-Forward ---
    log.info("\n[4/5] Walk-forward training (5 folds, 15-bar purge, GPU)...")
    results = run_training_and_backtest(
        featured_valid, labels_valid, feature_cols, n_splits=5, purge_bars=15,
    )

    log.info("\n--- WALK-FORWARD RESULTS ---")
    log.info("Total test instances: %d", results["total_test_instances"])
    log.info("Overall accuracy: %.4f", results["overall_accuracy"])
    log.info("Directional accuracy: %.4f (%d directional trades)", results["directional_accuracy"], results["directional_trades"])
    log.info("\nClassification Report:\n%s", results["classification_report"])
    log.info("\nTop 20 features:\n%s", results["feature_importance"].head(20).to_string())

    # --- Step 5: Kalshi P&L simulation ---
    log.info("\n[5/5] Simulating Kalshi P&L at various confidence thresholds...")

    for threshold in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        sim = simulate_kalshi_pnl(
            results["test_true"],
            results["test_preds"],
            results["test_proba"],
            confidence_threshold=threshold,
            fee_cents=0,
        )
        if sim["n_trades"] > 0:
            log.info(
                "  Threshold=%.0f%%: trades=%d win_rate=%.1f%% pnl=$%.2f avg=%.1fc sharpe=%.2f dd=$%.2f",
                threshold * 100,
                sim["n_trades"],
                sim["win_rate"] * 100,
                sim["total_pnl_dollars"],
                sim["avg_pnl_per_trade_cents"],
                sim["sharpe_ratio"],
                sim["max_drawdown_cents"] / 100,
            )
        else:
            log.info("  Threshold=%.0f%%: no trades", threshold * 100)

    # Best threshold analysis
    best_sim = simulate_kalshi_pnl(
        results["test_true"],
        results["test_preds"],
        results["test_proba"],
        confidence_threshold=0.50,
        fee_cents=0,
    )

    if best_sim["n_trades"] > 0:
        log.info("\n--- DETAILED BACKTEST (50%% threshold) ---")
        log.info("Total trades: %d", best_sim["n_trades"])
        log.info("Wins: %d | Losses: %d", best_sim["wins"], best_sim["losses"])
        log.info("Win rate: %.1f%%", best_sim["win_rate"] * 100)
        log.info("Total P&L: $%.2f", best_sim["total_pnl_dollars"])
        log.info("Avg P&L per trade: %.1f cents", best_sim["avg_pnl_per_trade_cents"])
        log.info("Max drawdown: $%.2f", best_sim["max_drawdown_cents"] / 100)
        log.info("Sharpe ratio: %.2f", best_sim["sharpe_ratio"])

        # Show last 20 trades
        trades_df = best_sim["trades_df"]
        log.info("\nLast 20 trades:")
        for _, t in trades_df.tail(20).iterrows():
            emoji = "W" if t["correct"] else "L"
            log.info(
                "  [%s] pred=%s true=%s conf=%.1f%% entry=%dc pnl=%+dc running=$%.2f",
                emoji, t["pred"], t["true"], t["confidence"] * 100,
                t["entry_cents"], t["pnl_cents"], t["running_pnl"] / 100,
            )

    # --- Save model ---
    model_dir = Path(settings.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"xgb_{ts}.json"
    results["final_model"].save_model(str(model_path))
    log.info("\nFinal model saved to %s", model_path)

    # Save features for reference
    data_dir = Path(settings.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    featured_valid["label"] = labels_valid.values
    featured_valid.to_parquet(data_dir / "features_labeled.parquet", index=False)
    log.info("Feature data saved to %s", data_dir / "features_labeled.parquet")

    log.info("\n" + "=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
