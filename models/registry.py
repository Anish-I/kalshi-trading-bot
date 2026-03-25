"""Model registry for saving, loading, and listing trained models."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from models.xgboost_model import XGBoostDirectionModel
from models.transformer_model import TransformerDirectionModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages persistence of trained models with metadata."""

    def __init__(self, model_dir: str | None = None):
        self.model_dir = Path(model_dir or settings.MODEL_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, model_type: str, metadata: dict | None = None) -> str:
        """Save a model with timestamp-based filename.

        Args:
            model: An XGBoostDirectionModel or TransformerDirectionModel instance.
            model_type: One of 'xgboost' or 'transformer'.
            metadata: Optional dict of extra metadata to persist.

        Returns:
            Path to the saved model file.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_name = f"{model_type}_{timestamp}"

        if model_type == "xgboost":
            model_path = str(self.model_dir / f"{base_name}.json")
        elif model_type == "transformer":
            model_path = str(self.model_dir / f"{base_name}.pt")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.save(model_path)

        # Save metadata alongside
        meta = {
            "model_type": model_type,
            "timestamp": timestamp,
            "path": model_path,
            **(metadata or {}),
        }
        meta_path = str(self.model_dir / f"{base_name}.meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info("Model saved to registry: %s", model_path)
        return model_path

    def load_latest(self, model_type: str = "xgboost"):
        """Load the most recently saved model of the given type.

        Returns:
            An XGBoostDirectionModel or TransformerDirectionModel instance, or None.
        """
        if model_type == "xgboost":
            pattern = "xgboost_*.json"
        elif model_type == "transformer":
            pattern = "transformer_*.pt"
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        candidates = sorted(self.model_dir.glob(pattern))
        # Filter out .features.json and .meta.json files
        candidates = [c for c in candidates if not c.name.endswith(".features.json") and not c.name.endswith(".meta.json")]

        if not candidates:
            logger.warning("No %s models found in %s", model_type, self.model_dir)
            return None

        latest = candidates[-1]  # sorted alphabetically, timestamps sort correctly

        if model_type == "xgboost":
            model = XGBoostDirectionModel()
            model.load(str(latest))
        else:
            model = TransformerDirectionModel(d_in=1)  # d_in will be overwritten by load
            model.load(str(latest))

        logger.info("Loaded latest %s model: %s", model_type, latest.name)
        return model

    def list_models(self) -> list[dict]:
        """List all saved models with their metadata."""
        results = []
        for meta_file in sorted(self.model_dir.glob("*.meta.json")):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                results.append(meta)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read metadata %s: %s", meta_file, e)
        return results
