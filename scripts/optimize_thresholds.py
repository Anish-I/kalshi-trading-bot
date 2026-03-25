"""Grid search to optimize Momentum and Mean Reversion model thresholds."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path

# Load data
feat_dir = Path("D:/kalshi-data/features")
df = pd.concat([pd.read_parquet(f) for f in sorted(feat_dir.glob("*.parquet"))]).sort_values("timestamp").reset_index(drop=True)
print(f"Total rows: {len(df)}")

# Create labels
price = df["btc_price"].astype(float)
horizon = 180
fwd_ret = np.log(price.shift(-horizon) / price)
thr = 3e-4
labels = np.where(fwd_ret > thr, 1, np.where(fwd_ret < -thr, -1, 0))
labels[-horizon:] = 99

valid = labels != 99
X = df[valid].reset_index(drop=True)
y = labels[valid]
print(f"Valid: {len(X)} | UP={int((y==1).sum())} FLAT={int((y==0).sum())} DOWN={int((y==-1).sum())}")

split = int(len(X) * 0.8)
X_opt, y_opt = X.iloc[:split], y[:split]
X_val, y_val = X.iloc[split:], y[split:]
print(f"Opt: {len(X_opt)} | Val: {len(X_val)}")

# ============================================================
# MOMENTUM
# ============================================================
print("\n" + "=" * 60)
print("MOMENTUM MODEL THRESHOLD OPTIMIZATION")
print("=" * 60)

best_mom = {"score": -999}

for m5_thr in [0.0003, 0.0005, 0.001, 0.0015, 0.002]:
    for m10_thr in [0.0005, 0.001, 0.002, 0.003]:
        for vwap_thr in [0.0003, 0.0005, 0.001, 0.002]:
            for donch_hi in [0.6, 0.7, 0.8]:
                for ema_thr in [0.00005, 0.0001, 0.0002]:
                    sigs = np.zeros(len(X_opt))
                    m5 = X_opt["momentum_5m"].fillna(0).values
                    m10 = X_opt["momentum_10m"].fillna(0).values
                    vwap = X_opt["vwap_deviation"].fillna(0).values
                    donch = X_opt["donchian_position"].fillna(0.5).values
                    ema_s = X_opt["ema_9_slope"].fillna(0).values

                    sigs += np.where(m5 > m5_thr, 1, np.where(m5 < -m5_thr, -1, 0))
                    sigs += np.where(m10 > m10_thr, 1, np.where(m10 < -m10_thr, -1, 0))
                    sigs += np.where(vwap > vwap_thr, 1, np.where(vwap < -vwap_thr, -1, 0))
                    sigs += np.where(donch > donch_hi, 1, np.where(donch < (1 - donch_hi), -1, 0))
                    sigs += np.where(ema_s > ema_thr, 1, np.where(ema_s < -ema_thr, -1, 0))

                    preds = np.where(sigs >= 3, 1, np.where(sigs <= -3, -1, 0))
                    directional = preds != 0
                    if directional.sum() < 100:
                        continue
                    acc = (preds[directional] == y_opt[directional]).mean()
                    n = int(directional.sum())
                    score = acc * min(1.0, n / 1000)
                    if score > best_mom["score"]:
                        best_mom = {"score": score, "acc": acc, "n": n,
                                    "m5": m5_thr, "m10": m10_thr, "vwap": vwap_thr,
                                    "donch_hi": donch_hi, "ema": ema_thr}

print(f"Best opt: acc={best_mom['acc']:.1%} trades={best_mom['n']}")
print(f"  m5={best_mom['m5']} m10={best_mom['m10']} vwap={best_mom['vwap']} donch={best_mom['donch_hi']} ema={best_mom['ema']}")

# Validate
p = best_mom
sigs = np.zeros(len(X_val))
sigs += np.where(X_val["momentum_5m"].fillna(0).values > p["m5"], 1, np.where(X_val["momentum_5m"].fillna(0).values < -p["m5"], -1, 0))
sigs += np.where(X_val["momentum_10m"].fillna(0).values > p["m10"], 1, np.where(X_val["momentum_10m"].fillna(0).values < -p["m10"], -1, 0))
sigs += np.where(X_val["vwap_deviation"].fillna(0).values > p["vwap"], 1, np.where(X_val["vwap_deviation"].fillna(0).values < -p["vwap"], -1, 0))
sigs += np.where(X_val["donchian_position"].fillna(0.5).values > p["donch_hi"], 1, np.where(X_val["donchian_position"].fillna(0.5).values < (1 - p["donch_hi"]), -1, 0))
sigs += np.where(X_val["ema_9_slope"].fillna(0).values > p["ema"], 1, np.where(X_val["ema_9_slope"].fillna(0).values < -p["ema"], -1, 0))
pv = np.where(sigs >= 3, 1, np.where(sigs <= -3, -1, 0))
dv = pv != 0
print(f"Validation: acc={((pv[dv] == y_val[dv]).mean() if dv.sum() > 0 else 0):.1%} trades={dv.sum()}")

# ============================================================
# MEAN REVERSION
# ============================================================
print("\n" + "=" * 60)
print("MEAN REVERSION MODEL THRESHOLD OPTIMIZATION")
print("=" * 60)

best_mr = {"score": -999}

for rsi_lo in [20, 25, 30, 35]:
    for rsi_hi in [65, 70, 75, 80]:
        for donch_lo in [0.1, 0.15, 0.2, 0.25]:
            for stoch_lo in [10, 15, 20, 25]:
                sigs = np.zeros(len(X_opt))
                rsi = X_opt["rsi_14"].fillna(50).values
                donch = X_opt["donchian_position"].fillna(0.5).values
                stoch = X_opt["stoch_k"].fillna(50).values

                sigs += np.where(rsi < rsi_lo, 2, np.where(rsi < rsi_lo + 10, 1,
                         np.where(rsi > rsi_hi, -2, np.where(rsi > rsi_hi - 10, -1, 0))))
                sigs += np.where(donch < donch_lo, 2, np.where(donch < donch_lo + 0.1, 1,
                         np.where(donch > 1 - donch_lo, -2, np.where(donch > 1 - donch_lo - 0.1, -1, 0))))
                sigs += np.where(stoch < stoch_lo, 1, np.where(stoch > 100 - stoch_lo, -1, 0))

                preds = np.where(sigs >= 3, 1, np.where(sigs <= -3, -1, 0))
                directional = preds != 0
                if directional.sum() < 50:
                    continue
                acc = (preds[directional] == y_opt[directional]).mean()
                n = int(directional.sum())
                score = acc * min(1.0, n / 500)
                if score > best_mr["score"]:
                    best_mr = {"score": score, "acc": acc, "n": n,
                               "rsi_lo": rsi_lo, "rsi_hi": rsi_hi,
                               "donch_lo": donch_lo, "stoch_lo": stoch_lo}

print(f"Best opt: acc={best_mr['acc']:.1%} trades={best_mr['n']}")
print(f"  rsi_lo={best_mr['rsi_lo']} rsi_hi={best_mr['rsi_hi']} donch_lo={best_mr['donch_lo']} stoch_lo={best_mr['stoch_lo']}")

# Validate
p = best_mr
sigs = np.zeros(len(X_val))
rsi = X_val["rsi_14"].fillna(50).values
donch = X_val["donchian_position"].fillna(0.5).values
stoch = X_val["stoch_k"].fillna(50).values
sigs += np.where(rsi < p["rsi_lo"], 2, np.where(rsi < p["rsi_lo"] + 10, 1,
         np.where(rsi > p["rsi_hi"], -2, np.where(rsi > p["rsi_hi"] - 10, -1, 0))))
sigs += np.where(donch < p["donch_lo"], 2, np.where(donch < p["donch_lo"] + 0.1, 1,
         np.where(donch > 1 - p["donch_lo"], -2, np.where(donch > 1 - p["donch_lo"] - 0.1, -1, 0))))
sigs += np.where(stoch < p["stoch_lo"], 1, np.where(stoch > 100 - p["stoch_lo"], -1, 0))
pv = np.where(sigs >= 3, 1, np.where(sigs <= -3, -1, 0))
dv = pv != 0
print(f"Validation: acc={((pv[dv] == y_val[dv]).mean() if dv.sum() > 0 else 0):.1%} trades={dv.sum()}")

# ============================================================
# COMPARISON: Old vs Optimized
# ============================================================
print("\n" + "=" * 60)
print("OLD vs OPTIMIZED THRESHOLDS")
print("=" * 60)
print(f"Momentum:")
print(f"  Old:  m5=0.001 m10=0.002 vwap=0.001 donch=0.7 ema=0.0001")
print(f"  New:  m5={best_mom['m5']} m10={best_mom['m10']} vwap={best_mom['vwap']} donch={best_mom['donch_hi']} ema={best_mom['ema']}")
print(f"Mean Reversion:")
print(f"  Old:  rsi_lo=30 rsi_hi=70 donch_lo=0.1 stoch_lo=20")
print(f"  New:  rsi_lo={best_mr['rsi_lo']} rsi_hi={best_mr['rsi_hi']} donch_lo={best_mr['donch_lo']} stoch_lo={best_mr['stoch_lo']}")
