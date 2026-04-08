"""Train both feature sets, build both calibrations, print a comparison report.

This script does NOT modify MACRO_FEATURES_ENABLED in config/settings.py.
Each subprocess toggles the flag in-process only via a context manager.
Promotion of honest_plus_macro to live trading requires human review.
"""
import subprocess
import sys
import json
from pathlib import Path


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit(r.returncode)
    return r.stdout


def latest(prefix: str) -> Path | None:
    models = sorted(Path("D:/kalshi-models").glob(f"{prefix}*.schema.json"))
    return models[-1] if models else None


def load_metrics(schema_path: Path | None) -> dict:
    if not schema_path or not schema_path.exists():
        return {}
    return json.loads(schema_path.read_text())


if __name__ == "__main__":
    run([sys.executable, "scripts/train_honest.py", "--features", "honest"])
    run([sys.executable, "scripts/train_honest.py", "--features", "honest_plus_macro"])
    run([sys.executable, "scripts/build_crypto_calibration.py", "--features", "honest"])
    run([sys.executable, "scripts/build_crypto_calibration.py", "--features", "honest_plus_macro"])

    h = load_metrics(latest("xgb_honest_2"))
    hm = load_metrics(latest("xgb_honest_plus_macro_"))

    print("\n=== ABLATION REPORT ===")
    print(f"{'metric':<30} {'honest':>15} {'honest+macro':>15} {'delta':>10}")
    for k in ("cv_accuracy", "cv_log_loss", "training_samples", "n_features"):
        a = h.get(k)
        b = hm.get(k)
        if a is None or b is None:
            continue
        delta = b - a if isinstance(a, (int, float)) else "n/a"
        print(f"{k:<30} {a:>15} {b:>15} {delta:>10}")

    print("\n=== CRITERION ===")
    print("Promote honest_plus_macro to live ONLY IF:")
    print("  - cv_accuracy improves, AND")
    print("  - calibrated EV improves (run build_crypto_calibration to compare), AND")
    print("  - tradable bucket count does NOT shrink")
    print("\nFlag MACRO_FEATURES_ENABLED stays False until human review.")
