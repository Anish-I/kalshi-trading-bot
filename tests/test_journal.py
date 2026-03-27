"""Tests for TradeJournal adaptive weighting."""
import sys
import tempfile

sys.path.insert(0, ".")

from models.trade_journal import TradeJournal


def _make_journal():
    tmp = tempfile.mkdtemp()
    return TradeJournal(data_dir=tmp)


def test_initial_weights_equal():
    j = _make_journal()
    w = j.get_model_weights()
    assert all(v == 1.0 for v in w.values())


def test_log_decision():
    j = _make_journal()
    j.log_decision(
        ticker="TEST-123",
        btc_price=70000,
        models={"xgboost": {"vote": "DOWN", "confidence": 80}},
        vote_result="DOWN",
        agreement=3,
        action="pending",
        side="no",
        edge=0.15,
    )
    assert len(j.entries) == 1
    assert j.entries[0]["action"] == "pending"


def test_log_outcome_updates_entry():
    j = _make_journal()
    j.log_decision(
        ticker="TEST-123",
        btc_price=70000,
        models={},
        vote_result="DOWN",
        agreement=3,
        action="trading",
        side="no",
    )
    j.log_outcome("TEST-123", won=True, pnl_cents=48)
    assert j.entries[0]["won"] is True
    assert j.entries[0]["pnl_cents"] == 48
    assert j.entries[0]["settled"] is True


def test_weights_clamp():
    """Weights must stay in [0.3, 2.5] range."""
    j = _make_journal()
    w = j.get_model_weights()
    for v in w.values():
        assert 0.3 <= v <= 2.5


def test_weights_adapt_after_outcomes():
    j = _make_journal()

    # Simulate 10 trades where momentum is always right, xgboost always wrong.
    for i in range(10):
        j.log_decision(
            ticker=f"T-{i}",
            btc_price=70000,
            models={
                "xgboost": {"vote": "UP", "confidence": 70},
                "momentum": {"vote": "DOWN", "confidence": 65},
                "mean_reversion": {"vote": "FLAT", "confidence": 50},
                "kalshi_consensus": {"vote": "FLAT", "confidence": 50},
            },
            vote_result="DOWN",
            agreement=2,
            action="trading",
            side="no",
        )
        j.log_outcome(f"T-{i}", won=True, pnl_cents=50)

    w = j.get_model_weights()
    assert w["momentum"] > 1.0, f"Momentum should be boosted, got {w['momentum']}"
    assert w["xgboost"] < 1.0, f"XGBoost should be penalized, got {w['xgboost']}"


def test_save_and_load():
    j = _make_journal()
    j.log_decision(
        ticker="T-1",
        btc_price=70000,
        models={},
        vote_result="UP",
        agreement=3,
        action="executed",
        side="yes",
    )
    j.log_outcome("T-1", won=True, pnl_cents=30)
    j.save()

    j2 = TradeJournal(data_dir=j.journal_path.parent)
    assert len(j2.entries) == 1
    assert j2.entries[0]["won"] in (True, 1)


def test_get_stats():
    j = _make_journal()
    for i in range(5):
        j.log_decision(
            ticker=f"T-{i}",
            btc_price=70000,
            models={},
            vote_result="DOWN",
            agreement=3,
            action="trading",
            side="no",
        )
        j.log_outcome(f"T-{i}", won=(i < 3), pnl_cents=50 if i < 3 else -50)

    stats = j.get_stats()
    assert stats["settled_trades"] == 5
    assert stats["wins"] == 3
    assert stats["losses"] == 2
    assert stats["win_rate"] == 0.6


def test_phantom_trade_not_in_settled():
    """Failed orders should not appear in settled stats."""
    j = _make_journal()
    j.log_decision(
        ticker="FAIL-1",
        btc_price=70000,
        models={},
        vote_result="UP",
        agreement=3,
        action="failed",
        side="yes",
    )
    stats = j.get_stats()
    assert stats["settled_trades"] == 0


def test_log_decision_records_calibration_fields_and_honest_features():
    j = _make_journal()
    j.log_decision(
        ticker="CAL-1",
        btc_price=70000,
        models={"xgboost": {"vote": "UP", "confidence": 61}},
        vote_result="UP",
        agreement=2,
        action="pending",
        side="yes",
        entry_price=45,
        contracts=10,
        bet_dollars=4.5,
        edge=0.035,
        features_snapshot={"rsi_14": 33.0, "ret_5m": 0.001, "ret_10m": 0.002},
        session_tag="us_core",
        effective_simulate=True,
        rule_name="xgb+mom calibrated_ev",
        entry_side="yes",
        entry_price_cents=45,
        price_bucket="45c",
        calibrated_p_win=0.485,
        gross_ev_cents_per_contract=3.5,
        net_ev_cents_per_contract=1.5,
        bucket_trade_count=60,
        calibration_version="artifact-123",
    )

    entry = j.entries[0]
    assert entry["session_tag"] == "us_core"
    assert entry["effective_simulate"] is True
    assert entry["price_bucket"] == "45c"
    assert entry["calibrated_p_win"] == 0.485
    assert entry["bucket_trade_count"] == 60
    assert entry["calibration_version"] == "artifact-123"
    assert entry["feat_ret_5m"] == 0.001
    assert entry["feat_ret_10m"] == 0.002
