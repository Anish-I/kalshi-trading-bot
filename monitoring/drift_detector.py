"""Feature drift detection for model monitoring."""

from collections import deque

import pandas as pd


class DriftDetector:
    """Detects distribution shift between training and live features."""

    def __init__(self, feature_names: list[str], window: int = 100):
        self.feature_names = feature_names
        self.baseline_means: dict[str, float] = {}
        self.baseline_stds: dict[str, float] = {}
        self.recent_values: dict[str, deque] = {
            name: deque(maxlen=window) for name in feature_names
        }
        self.is_calibrated: bool = False

    def calibrate(self, feature_matrix: pd.DataFrame) -> None:
        """Compute baseline statistics from training data.

        Args:
            feature_matrix: DataFrame with columns matching feature_names.
        """
        for name in self.feature_names:
            self.baseline_means[name] = float(feature_matrix[name].mean())
            self.baseline_stds[name] = float(feature_matrix[name].std())
        self.is_calibrated = True

    def update(self, features: dict[str, float]) -> None:
        """Append a single observation for each feature.

        Args:
            features: Mapping of feature name to current value.
        """
        for name in self.feature_names:
            if name in features:
                self.recent_values[name].append(features[name])

    def check_drift(self) -> list[dict]:
        """Compare recent feature means to baseline using z-scores.

        Returns:
            List of dicts for features with |z| > 2, each containing:
            feature, z_score, recent_mean, baseline_mean.
        """
        if not self.is_calibrated:
            return []

        alerts: list[dict] = []

        for name in self.feature_names:
            values = self.recent_values[name]
            if not values:
                continue

            baseline_std = self.baseline_stds[name]
            if baseline_std == 0:
                continue

            recent_mean = sum(values) / len(values)
            baseline_mean = self.baseline_means[name]
            z = abs(recent_mean - baseline_mean) / baseline_std

            if z > 2:
                alerts.append(
                    {
                        "feature": name,
                        "z_score": round(z, 3),
                        "recent_mean": round(recent_mean, 6),
                        "baseline_mean": round(baseline_mean, 6),
                    }
                )

        return alerts

    def get_drift_summary(self) -> dict:
        """Return a high-level drift summary.

        Returns:
            Dict with n_drifted, total_features, drifted_features,
            worst_z_score.
        """
        alerts = self.check_drift()
        drifted_names = [a["feature"] for a in alerts]
        worst_z = max((a["z_score"] for a in alerts), default=0.0)

        return {
            "n_drifted": len(alerts),
            "total_features": len(self.feature_names),
            "drifted_features": drifted_names,
            "worst_z_score": worst_z,
        }
