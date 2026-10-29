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


def test_abstention_normalized_score_low():
    """Verify abstention when normalized BM25 score per content word is below threshold."""
    engine = AbstentionEngine(
        enabled=True,
        strategy="normalized_score_coverage",
        min_norm_score=2.4,
    )
    # Query with 4 content words: "broadcast", "syndication", "rights", "matches"
    query = "What are the broadcast syndication rights granted for cricket matches?"
    # Score 6.0 / 4 content words = 1.5 < 2.4 -> should abstain
    candidates = [{"chunk_id": "c1", "score": 6.0, "text": "This agreement defines broadcast terms."}]
    should_abstain, filtered, rationale = engine.evaluate(candidates, query=query)

    assert should_abstain is True
    assert filtered == []
    assert "Normalized relevance score" in rationale


def test_abstention_normalized_score_high():
    """Verify non-abstention when normalized score is sufficient and coverage is good."""
    engine = AbstentionEngine(
        enabled=True,
        strategy="normalized_score_coverage",
        min_norm_score=2.4,
    )
    query = "What is the expiration date?"
    # 2 content words: "expiration", "date". Score 8.0 / 2 = 4.0 >= 2.4
    candidates = [{
        "chunk_id": "c1",
        "score": 8.0,
        "text": "The expiration date of this contract shall be December 31."
    }]
    should_abstain, filtered, rationale = engine.evaluate(candidates, query=query)

    assert should_abstain is False
    assert len(filtered) == 1
    assert "Candidate passed normalized relevance" in rationale


def test_abstention_coverage_guardrail():
    """Verify abstention when query has >=5 content words but only <=1 appears in passage."""
    engine = AbstentionEngine(
        enabled=True,
        strategy="normalized_score_coverage",
        min_norm_score=2.0,
        min_words_for_guard=5,
        max_matched_for_guard=1,
    )
    # 6 content words: "average", "annual", "rainfall", "amazon", "rainforest", "basin"
    query = "What is the average annual rainfall in the Amazon rainforest basin?"
    # Top score 18.0 / 6 = 3.0 (passes min_norm_score 2.0), but text only contains "average" (1 match)
    candidates = [{
        "chunk_id": "c1",
        "score": 18.0,
        "text": "The average daily volume of corporate shares shall be reported annually."
    }]
    should_abstain, filtered, rationale = engine.evaluate(candidates, query=query)

    assert should_abstain is True
    assert filtered == []
    assert "Insufficient evidence coverage" in rationale
