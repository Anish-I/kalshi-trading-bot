"""Tests for pair state machine."""
import pytest
from engine.pair_state import PairTrade, PairTracker


def test_pair_state_transitions():
    pair = PairTrade(ticker="TEST-001")
    assert pair.state == "idle"
    assert not pair.is_orphan
    assert not pair.is_complete

    pair.transition("quoting_both")
    assert pair.state == "quoting_both"

    pair.record_yes_fill(40, 1)
    assert pair.state == "yes_only_filled"
    assert pair.is_orphan
    assert pair.yes_filled
    assert not pair.no_filled

    pair.record_no_fill(50, 1)
    assert pair.state == "pair_completed"
    assert pair.is_complete
    assert pair.pair_cost_cents == 90
    assert pair.pair_profit_cents == 10


def test_both_fill_simultaneously():
    pair = PairTrade(ticker="TEST-002", state="quoting_both",
                     yes_submitted_price=30, no_submitted_price=60)
    pair.record_no_fill(60, 1)
    assert pair.state == "no_only_filled"

    pair.record_yes_fill(30, 1)
    assert pair.state == "pair_completed"
    assert pair.pair_cost_cents == 90


def test_tracker_stats():
    tracker = PairTracker()
    p1 = tracker.start_pair("T1", 40, 50)
    p1.record_yes_fill(40, 1)
    p1.record_no_fill(50, 1)
    tracker.complete_pair("T1")

    stats = tracker.get_stats()
    assert stats["completed_pairs"] == 1
    assert stats["total_profit_cents"] == 10


def test_orphan_resolve():
    tracker = PairTracker()
    p = tracker.start_pair("T2", 45, 45)
    p.record_yes_fill(45, 1)
    assert p.is_orphan

    resolved = tracker.resolve_orphan("T2", "timeout")
    assert resolved is not None
    assert resolved.state == "resolved"
    assert len(tracker.active) == 0
    assert len(tracker.completed) == 1
