"""Tests for Streamlit UI helpers, comparative retrieval, and evaluation history."""

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


def test_parallel_pipeline_comparison_execution():
    """Verify that baseline and final pipelines execute side-by-side with exact offsets."""
    query = "Highlight provisions concerning confidentiality."
    top_k = 3

    base_p = get_cached_pipeline("configs/baseline.yaml")
    final_p = get_cached_pipeline("configs/final.yaml")

    base_res = base_p.search(query=query, top_k=top_k)
    final_res = final_p.search(query=query, top_k=top_k)

    # Both must return valid schema
    for res, p in [(base_res, base_p), (final_res, final_p)]:
        assert "status" in res
        assert "abstained" in res
        assert "passages" in res
        assert "abstention_rationale" in res

        # Verify exact substring offsets
        for passage in res["passages"]:
            start = passage["start_char"]
            end = passage["end_char"]
            doc_file = passage["file_path"]
            raw_text = p.contracts[doc_file]["text"]
            assert raw_text[start:end] == passage["text"]
