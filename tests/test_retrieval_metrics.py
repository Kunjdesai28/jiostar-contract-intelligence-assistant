"""Unit tests for character-level retrieval metrics and abstention evaluation.

Includes the 10 mandatory synthetic verification cases specified in Phase 2.5 audit:
  1. Perfect 100-character match.
  2. 50-character overlap.
  3. Zero overlap in same document.
  4. Two gold spans in one document.
  5. One predicted span covering two gold spans.
  6. Overlapping retrieved spans (merged without double counting).
  7. Identical offsets in different documents (isolated per document).
  8. Empty retrieved list.
  9. Multiple predictions at different ranks across multiple documents.
  10. Duplicate predicted spans in retrieved list.
"""

import pytest
from src.evaluation.metrics import (
    compute_abstention_metrics,
    compute_corpus_retrieval_metrics,
    compute_query_character_metrics,
)


def test_synthetic_case_1_perfect_100_char_match():
    """Case 1: Exact 100-character span match in the same document.
    Expected: overlap=100, ret=100, gt=100 -> Precision=1.0, Recall=1.0
    """
    gt = [{"file_path": "c1.txt", "span": [100, 200]}]
    ret = [{"file_path": "c1.txt", "start_char": 100, "end_char": 200}]

    res = compute_query_character_metrics(gt, ret)
    assert res["overlap_chars"] == 100
    assert res["retrieved_chars"] == 100
    assert res["gt_chars"] == 100
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0


def test_synthetic_case_2_partial_50_char_overlap():
    """Case 2: 50-character overlap between retrieved passage and gold span.
    Expected: overlap [150:200] = 50 chars. gt=100, ret=100 -> Precision=0.5, Recall=0.5
    """
    gt = [{"file_path": "c1.txt", "span": [100, 200]}]  # 100 chars
    ret = [{"file_path": "c1.txt", "start_char": 150, "end_char": 250}]  # 100 chars

    res = compute_query_character_metrics(gt, ret)
    assert res["overlap_chars"] == 50
    assert res["retrieved_chars"] == 100
    assert res["gt_chars"] == 100
    assert res["precision"] == 0.5
    assert res["recall"] == 0.5


def test_synthetic_case_3_zero_overlap_same_document():
    """Case 3: Same document, but non-overlapping character ranges.
    Expected: overlap=0 -> Precision=0.0, Recall=0.0
    """
    gt = [{"file_path": "c1.txt", "span": [100, 200]}]
    ret = [{"file_path": "c1.txt", "start_char": 300, "end_char": 400}]

    res = compute_query_character_metrics(gt, ret)
    assert res["overlap_chars"] == 0
    assert res["retrieved_chars"] == 100
    assert res["gt_chars"] == 100
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0


def test_synthetic_case_4_two_gold_spans_one_document():
    """Case 4: Two gold spans in one document, only one retrieved.
    Gold: [100, 200] (100 chars) + [300, 400] (100 chars) -> Total GT = 200 chars.
    Pred: [100, 200] (100 chars).
    Expected: overlap=100. Precision=100/100=1.0, Recall=100/200=0.5
    """
    gt = [
        {"file_path": "c1.txt", "span": [100, 200]},
        {"file_path": "c1.txt", "span": [300, 400]},
    ]
    ret = [{"file_path": "c1.txt", "start_char": 100, "end_char": 200}]

    res = compute_query_character_metrics(gt, ret)
    assert res["overlap_chars"] == 100
    assert res["gt_chars"] == 200
    assert res["retrieved_chars"] == 100
    assert res["precision"] == 1.0
    assert res["recall"] == 0.5


def test_synthetic_case_5_one_predicted_span_covering_two_gold_spans():
    """Case 5: A single large retrieved span covering both gold spans.
    Gold: [100, 200] (100 chars) + [300, 400] (100 chars) -> Total GT = 200 chars.
    Pred: [50, 450] (400 chars).
    Expected: overlap = 100 + 100 = 200 chars.
    Precision = 200 / 400 = 0.5, Recall = 200 / 200 = 1.0
    """
    gt = [
        {"file_path": "c1.txt", "span": [100, 200]},
        {"file_path": "c1.txt", "span": [300, 400]},
    ]
    ret = [{"file_path": "c1.txt", "start_char": 50, "end_char": 450}]

    res = compute_query_character_metrics(gt, ret)
    assert res["overlap_chars"] == 200
    assert res["gt_chars"] == 200
    assert res["retrieved_chars"] == 400
    assert res["precision"] == 0.5
    assert res["recall"] == 1.0


def test_synthetic_case_6_overlapping_retrieved_spans():
    """Case 6: Overlapping retrieved spans in the same document must be merged.
    Gold: [100, 200] (100 chars).
    Ret: [100, 160] (60 chars) and [140, 220] (80 chars).
    Union of retrieved chars: [100, 220] = 120 chars (NOT 60+80=140).
    Overlap: [100, 200] = 100 chars.
    Expected: Precision = 100 / 120 = 0.8333, Recall = 100 / 100 = 1.0
    """
    gt = [{"file_path": "c1.txt", "span": [100, 200]}]
    ret = [
        {"file_path": "c1.txt", "start_char": 100, "end_char": 160},
        {"file_path": "c1.txt", "start_char": 140, "end_char": 220},
    ]

    res = compute_query_character_metrics(gt, ret)
    assert res["overlap_chars"] == 100
    assert res["retrieved_chars"] == 120
    assert res["gt_chars"] == 100
    assert round(res["precision"], 4) == 0.8333
    assert res["recall"] == 1.0


def test_synthetic_case_7_identical_offsets_different_documents():
    """Case 7: Same character offsets in different documents must NEVER match.
    Gold: [100, 200] in 'doc_A.txt'
    Pred: [100, 200] in 'doc_B.txt'
    Expected: overlap = 0. Precision = 0.0, Recall = 0.0
    """
    gt = [{"file_path": "doc_A.txt", "span": [100, 200]}]
    ret = [{"file_path": "doc_B.txt", "start_char": 100, "end_char": 200}]

    res = compute_query_character_metrics(gt, ret)
    assert res["overlap_chars"] == 0
    assert res["retrieved_chars"] == 100
    assert res["gt_chars"] == 100
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0


def test_synthetic_case_8_empty_retrieved_list():
    """Case 8: Empty predictions (abstention or zero matches).
    Expected: overlap = 0, retrieved_chars = 0 -> Precision = 0.0, Recall = 0.0
    """
    gt = [{"file_path": "c1.txt", "span": [100, 200]}]
    ret = []

    res = compute_query_character_metrics(gt, ret)
    assert res["overlap_chars"] == 0
    assert res["retrieved_chars"] == 0
    assert res["gt_chars"] == 100
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0


def test_synthetic_case_9_multiple_predictions_different_ranks():
    """Case 9: Multiple predictions at different ranks across multiple documents.
    Gold: [0, 100] in doc1 (100 chars) + [0, 100] in doc2 (100 chars) -> Total GT = 200 chars.
    Rank 1: doc1 [0, 100] (100 chars)
    Rank 2: doc3 [0, 500] (500 chars, distractor)
    Rank 3: doc2 [0, 50] (50 chars)
    
    At K=1: overlap=100, ret=100 -> Prec@1 = 100/100 = 1.0, Rec@1 = 100/200 = 0.5
    At K=2: overlap=100, ret=600 -> Prec@2 = 100/600 = 0.1667, Rec@2 = 100/200 = 0.5
    At K=3: overlap=150, ret=650 -> Prec@3 = 150/650 = 0.2308, Rec@3 = 150/200 = 0.75
    """
    eval_rec = {
        "ground_truth_snippets": [
            {"file_path": "doc1.txt", "span": [0, 100]},
            {"file_path": "doc2.txt", "span": [0, 100]},
        ],
        "ranked_passages": [
            {"file_path": "doc1.txt", "start_char": 0, "end_char": 100},
            {"file_path": "doc3.txt", "start_char": 0, "end_char": 500},
            {"file_path": "doc2.txt", "start_char": 0, "end_char": 50},
        ],
    }

    metrics = compute_corpus_retrieval_metrics([eval_rec], k_values=[1, 2, 3])
    assert metrics["precision@1"] == 1.0
    assert metrics["recall@1"] == 0.5

    assert round(metrics["precision@2"], 4) == 0.1667
    assert metrics["recall@2"] == 0.5

    assert round(metrics["precision@3"], 4) == 0.2308
    assert metrics["recall@3"] == 0.75


def test_synthetic_case_10_duplicate_predicted_spans():
    """Case 10: Duplicate predicted spans in retrieved list must be merged without inflating denominator.
    Gold: [100, 200] (100 chars) in c1.txt.
    Retrieved: [100, 200] and duplicate [100, 200] in c1.txt.
    Expected: union of ret chars is 100 chars. Precision=1.0, Recall=1.0
    """
    gt = [{"file_path": "c1.txt", "span": [100, 200]}]
    ret = [
        {"file_path": "c1.txt", "start_char": 100, "end_char": 200},
        {"file_path": "c1.txt", "start_char": 100, "end_char": 200},
    ]

    res = compute_query_character_metrics(gt, ret)
    assert res["retrieved_chars"] == 100
    assert res["overlap_chars"] == 100
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0


def test_abstention_metrics_accuracy():
    """Verify abstention metrics calculation on unanswerable and answerable query results."""
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
