"""Tests for Streamlit UI helpers, evaluation history loader, and exact offset verification."""

from pathlib import Path
import pytest

from app.streamlit_app import (
    get_cached_pipeline,
    load_iteration_log,
    load_official_test_metrics,
)


def test_load_iteration_log_structure():
    """Verify that iteration log loader parses all 5 iterations with correct schema."""
    iterations = load_iteration_log()
    assert len(iterations) == 5

    for idx, it in enumerate(iterations, start=1):
        assert it["iteration"] == idx
        assert "candidate_config_id" in it
        assert "decision" in it
        assert it["decision"] in ("accepted", "rejected")
        assert "metrics_before" in it
        assert "metrics_after" in it

    # Check rejection of iteration 4
    assert iterations[3]["decision"] == "rejected"
    assert iterations[3]["candidate_config_id"] == "candidate_004"

    # Check acceptance of iteration 5
    assert iterations[4]["decision"] == "accepted"
    assert iterations[4]["candidate_config_id"] == "candidate_005"


def test_load_official_test_metrics_values():
    """Verify that official test metrics are loaded without re-evaluating."""
    metrics = load_official_test_metrics()
    assert metrics.get("split") == "test"

    char_metrics = metrics.get("character_level_retrieval_metrics", {})
    assert round(char_metrics.get("recall@5", 0.0), 4) == 0.2961
    assert round(char_metrics.get("precision@5", 0.0), 4) == 0.0429

    abs_metrics = metrics.get("abstention_metrics", {})
    assert abs_metrics.get("correct_negative_abstentions") == "6/13"
    assert abs_metrics.get("false_positive_abstentions") == "0/54"


def test_pipeline_execution_and_exact_offsets():
    """Verify that default pipeline executes search with 100% exact substring match."""
    query = "Highlight the parts related to Exclusivity."
    top_k = 3

    pipeline = get_cached_pipeline("configs/final.yaml")
    result = pipeline.search(query=query, top_k=top_k)

    assert result["status"] == "results_found"
    assert not result["abstained"]
    assert len(result["passages"]) == top_k

    for passage in result["passages"]:
        start = passage["start_char"]
        end = passage["end_char"]
        doc_file = passage["file_path"]
        raw_text = pipeline.contracts[doc_file]["text"]
        assert raw_text[start:end] == passage["text"]
