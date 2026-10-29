"""Unit tests for character-level retrieval metrics and abstention evaluation."""

import pytest
from src.evaluation.metrics import (
    compute_abstention_metrics,
    compute_corpus_retrieval_metrics,
    compute_query_character_metrics,
)


def test_compute_query_character_metrics_perfect_match():
    """Verify precision=1.0, recall=1.0 on exact span match."""
    gt = [{"file_path": "c1.txt", "span": [100, 200]}]  # 100 chars
    ret = [{"file_path": "c1.txt", "start_char": 100, "end_char": 200}]

    res = compute_query_character_metrics(gt, ret)
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["overlap_chars"] == 100
    assert res["gt_chars"] == 100
    assert res["retrieved_chars"] == 100


def test_compute_query_character_metrics_partial_overlap():
    """Verify metrics on partial character span overlap."""
    gt = [{"file_path": "c1.txt", "span": [100, 200]}]  # 100 chars
    ret = [{"file_path": "c1.txt", "start_char": 150, "end_char": 250}]  # 100 chars, overlap [150:200] = 50 chars

    res = compute_query_character_metrics(gt, ret)
    assert res["overlap_chars"] == 50
    assert res["gt_chars"] == 100
    assert res["retrieved_chars"] == 100
    assert res["precision"] == 0.5
    assert res["recall"] == 0.5


def test_compute_query_character_metrics_zero_overlap():
    """Verify zero overlap when documents match but ranges disjoint."""
    gt = [{"file_path": "c1.txt", "span": [100, 200]}]
    ret = [{"file_path": "c1.txt", "start_char": 300, "end_char": 400}]

    res = compute_query_character_metrics(gt, ret)
    assert res["overlap_chars"] == 0
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0


def test_compute_query_character_metrics_different_documents():
    """Verify that identical character ranges in different documents are never merged or matched."""
    gt = [{"file_path": "contract_A.txt", "span": [100, 200]}]
    ret = [{"file_path": "contract_B.txt", "start_char": 100, "end_char": 200}]

    res = compute_query_character_metrics(gt, ret)
    assert res["overlap_chars"] == 0
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0


def test_compute_query_character_metrics_overlapping_retrieved_passages():
    """Verify that overlapping retrieved passages in same document do not double-count retrieved characters."""
    gt = [{"file_path": "c1.txt", "span": [100, 200]}]  # 100 chars
    # Two retrieved chunks: [100, 160] (60 chars) and [140, 200] (60 chars)
    # Union is [100:200] = 100 chars (not 120 chars)
    ret = [
        {"file_path": "c1.txt", "start_char": 100, "end_char": 160},
        {"file_path": "c1.txt", "start_char": 140, "end_char": 200},
    ]

    res = compute_query_character_metrics(gt, ret)
    assert res["overlap_chars"] == 100
    assert res["retrieved_chars"] == 100
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0


def test_compute_corpus_retrieval_metrics_cutoffs():
    """Verify aggregation across multiple queries at various K cutoffs."""
    eval_recs = [
        {
            "ground_truth_snippets": [{"file_path": "doc.txt", "span": [0, 100]}],
            "ranked_passages": [
                {"file_path": "doc.txt", "start_char": 0, "end_char": 100},  # Rank 1: perfect
                {"file_path": "doc.txt", "start_char": 100, "end_char": 200},  # Rank 2
            ],
        },
        {
            "ground_truth_snippets": [{"file_path": "doc.txt", "span": [500, 600]}],
            "ranked_passages": [
                {"file_path": "doc.txt", "start_char": 0, "end_char": 100},  # Rank 1: no overlap
                {"file_path": "doc.txt", "start_char": 500, "end_char": 600},  # Rank 2: perfect
            ],
        },
    ]
    metrics = compute_corpus_retrieval_metrics(eval_recs, k_values=[1, 2])
    # Query 1: Precision@1 = 1.0, Recall@1 = 1.0
    # Query 2: Precision@1 = 0.0, Recall@1 = 0.0
    # Mean Recall@1 = (1.0 + 0.0)/2 = 0.5
    assert metrics["recall@1"] == 0.5
    # At K=2: both queries have the correct passage within top 2
    assert metrics["recall@2"] > metrics["recall@1"]


def test_compute_abstention_metrics():
    """Verify calculation of correct abstention, false abstention, and coverage rates."""
    unans_results = [
        {"abstained": True},
        {"abstained": True},
        {"abstained": False},  # 2 correct, 1 incorrect
    ]
    ans_results = [
        {"abstained": False},
        {"abstained": False},
        {"abstained": True},  # 1 false abstention, 2 covered
    ]

    metrics = compute_abstention_metrics(unans_results, ans_results)
    u = metrics["unanswerable_queries"]
    a = metrics["answerable_queries"]

    assert u["total"] == 3
    assert u["correct_abstentions"] == 2
    assert u["correct_abstention_rate"] == round(2 / 3, 4)

    assert a["total"] == 3
    assert a["false_abstentions"] == 1
    assert a["false_abstention_rate"] == round(1 / 3, 4)
    assert a["covered_queries"] == 2
    assert a["coverage_rate"] == round(2 / 3, 4)
