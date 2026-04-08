"""Smoke tests for scripts/train_honest.py CLI surface.

We don't run a real training (it requires a 90d binance parquet and is slow).
Instead we verify:
  - the module imports cleanly
  - the argparse --features flag exists and accepts both choices
  - the in-process macro toggle context manager restores state
"""
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_train_honest_imports():
    mod = importlib.import_module("scripts.train_honest")
    assert hasattr(mod, "main")
    assert hasattr(mod, "_parse_args")
    assert hasattr(mod, "_macro_features_enabled")


def test_parse_args_features_flag(monkeypatch):
    mod = importlib.import_module("scripts.train_honest")
    monkeypatch.setattr(sys, "argv", ["train_honest.py"])
    ns = mod._parse_args()
    assert ns.features == "honest"

    monkeypatch.setattr(sys, "argv", ["train_honest.py", "--features", "honest_plus_macro"])
    ns = mod._parse_args()
    assert ns.features == "honest_plus_macro"


def test_macro_features_context_manager_restores_flag():
    mod = importlib.import_module("scripts.train_honest")
    from config import settings as settings_module
    s = settings_module.settings
    prev = getattr(s, "MACRO_FEATURES_ENABLED", False)
    assert prev is False  # project invariant: flag off on disk

    with mod._macro_features_enabled(True):
        assert getattr(s, "MACRO_FEATURES_ENABLED") is True

    # must be restored after the context exits
    assert getattr(s, "MACRO_FEATURES_ENABLED") == prev
