"""Tests for the weighted voting system."""
import sys
sys.path.insert(0, ".")

from models.signal_models import vote, MomentumModel, MeanReversionModel


def test_unanimous_up():
    results = [("up", 0.70), ("up", 0.65), ("up", 0.60), ("up", 0.55)]
    direction, conf, score = vote(results)
    assert direction == "up"
    assert conf <= 0.95
    assert score >= 4.0


def test_unanimous_down():
    results = [("down", 0.80), ("down", 0.70), ("down", 0.60), ("down", 0.55)]
    direction, conf, score = vote(results)
    assert direction == "down"
    assert conf <= 0.95


def test_split_vote_no_trade():
    results = [("up", 0.70), ("down", 0.70), ("flat", 0.50), ("flat", 0.50)]
    direction, conf, score = vote(results)
    assert direction == "flat"
    assert score == 0.0


def test_two_up_no_opposition():
    results = [("up", 0.65), ("up", 0.60), ("flat", 0.50), ("flat", 0.50)]
    direction, conf, score = vote(results)
    assert direction == "up"
    assert score >= 2.0


def test_two_up_one_down_rejected():
    """2 up + 1 down: down_score=1 < up_score=2, should still pass since up > down."""
    results = [("up", 0.65), ("up", 0.60), ("down", 0.70), ("flat", 0.50)]
    direction, conf, score = vote(results)
    assert direction == "up"  # up_score=2 > down_score=1


def test_confidence_clamped():
    """Even with high individual confidences, output must be <= 0.95."""
    results = [("up", 0.99), ("up", 0.98), ("up", 0.97), ("up", 0.96)]
    direction, conf, score = vote(results)
    assert conf <= 0.95


def test_weighted_boost():
    """A 2x-weighted model should make a 1-vote direction reach threshold."""
    results = [("up", 0.70), ("flat", 0.50), ("flat", 0.50), ("flat", 0.50)]
    # Without weights: up_score=1.0 < 2.0 threshold → flat
    direction, conf, score = vote(results)
    assert direction == "flat"

    # With 2.5x weight on the UP model: up_score=2.5 >= 2.0 → up
    direction, conf, score = vote(results, weights=[2.5, 1.0, 1.0, 1.0])
    assert direction == "up"
    assert score >= 2.0
    assert conf <= 0.95


def test_weighted_override():
    """Higher-weighted down should beat lower-weighted up."""
    results = [("up", 0.70), ("up", 0.65), ("down", 0.80), ("flat", 0.50)]
    # Default weights: up=2.0, down=1.0 → up wins
    direction, _, _ = vote(results)
    assert direction == "up"

    # Give down model 3x weight: down=3.0 > up=2.0
    direction, conf, score = vote(results, weights=[1.0, 1.0, 3.0, 1.0])
    assert direction == "down"
    assert conf <= 0.95


def test_default_weights():
    """Without weights, should behave like equal 1.0 weights."""
    results = [("down", 0.70), ("down", 0.65), ("flat", 0.50), ("flat", 0.50)]
    d1, c1, s1 = vote(results)
    d2, c2, s2 = vote(results, weights=[1.0, 1.0, 1.0, 1.0])
    assert d1 == d2
    assert abs(c1 - c2) < 0.001
    assert abs(s1 - s2) < 0.001


def test_momentum_model_returns_valid():
    m = MomentumModel()
    features = {"momentum_5m": 0.003, "momentum_10m": 0.004,
                "vwap_deviation": 0.002, "donchian_position": 0.9,
                "ema_9_slope": 0.001}
    direction, conf = m.score(features)
    assert direction in ("up", "down", "flat")
    assert 0 <= conf <= 1.0


def test_meanrev_model_returns_valid():
    m = MeanReversionModel()
    features = {"rsi_14": 20, "donchian_position": 0.1, "stoch_k": 5,
                "volume_sma_ratio": 1.0}
    direction, conf = m.score(features)
    assert direction in ("up", "down", "flat")
    assert 0 <= conf <= 1.0
