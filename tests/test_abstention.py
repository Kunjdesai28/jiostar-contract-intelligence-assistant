"""Unit tests for the abstention engine."""

import pytest
from src.retrieval.abstention import AbstentionEngine


def test_abstention_empty_candidates():
    """Verify that an empty candidate list results in abstention."""
    engine = AbstentionEngine(enabled=True, threshold=5.0)
    should_abstain, candidates, rationale = engine.evaluate([])
    assert should_abstain is True
    assert candidates == []
    assert "No matching candidates" in rationale


def test_abstention_score_below_threshold():
    """Verify abstention when top candidate score is below threshold."""
    candidates = [{"chunk_id": "c1", "score": 3.5}, {"chunk_id": "c2", "score": 2.1}]
    engine = AbstentionEngine(enabled=True, threshold=5.0)
    should_abstain, filtered, rationale = engine.evaluate(candidates)

    assert should_abstain is True
    assert filtered == []
    assert "did not exceed threshold" in rationale


def test_abstention_score_equal_to_threshold():
    """Verify that score <= threshold boundary triggers abstention (inclusive boundary)."""
    candidates = [{"chunk_id": "c1", "score": 5.0}]
    engine = AbstentionEngine(enabled=True, threshold=5.0)
    should_abstain, filtered, rationale = engine.evaluate(candidates)

    assert should_abstain is True
    assert filtered == []


def test_abstention_score_above_threshold():
    """Verify that score > threshold does not abstain and returns candidates."""
    candidates = [{"chunk_id": "c1", "score": 7.2}, {"chunk_id": "c2", "score": 5.1}]
    engine = AbstentionEngine(enabled=True, threshold=5.0)
    should_abstain, filtered, rationale = engine.evaluate(candidates)

    assert should_abstain is False
    assert len(filtered) == 2
    assert "exceeded threshold" in rationale


def test_abstention_disabled():
    """Verify that when disabled, candidates are returned regardless of scores."""
    candidates = [{"chunk_id": "c1", "score": 0.0}]
    engine = AbstentionEngine(enabled=False, threshold=10.0)
    should_abstain, filtered, rationale = engine.evaluate(candidates)

    assert should_abstain is False
    assert len(filtered) == 1
    assert "Abstention disabled" in rationale
