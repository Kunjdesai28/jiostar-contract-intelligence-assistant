"""Character-level retrieval metrics and abstention evaluation for LegalBench-RAG."""

from typing import Any, Dict, List, Set, Tuple


def _compute_character_set(spans: List[Tuple[int, int]]) -> Set[int]:
    """Compute the union of integer character indices for a list of (start, end) intervals."""
    char_set: Set[int] = set()
    for start, end in spans:
        if end > start and start >= 0:
            char_set.update(range(start, end))
    return char_set


def compute_query_character_metrics(
    ground_truth_snippets: List[Dict[str, Any]],
    retrieved_passages: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute character-level precision and recall for a single query against ground truth.

    Definitions follow LegalBench-RAG character-level metrics:
      - Overlapping retrieved passages in the same document are merged (no double-counting).
      - Multiple ground-truth snippets in the same document are merged.
      - Different documents are kept strictly separate.
      - Precision = |Overlap| / |Retrieved Chars| (0.0 if Retrieved Chars == 0)
      - Recall = |Overlap| / |Ground Truth Chars| (0.0 if Ground Truth Chars == 0)

    Args:
        ground_truth_snippets: List of snippet dicts with 'file_path' and 'span' [start, end].
        retrieved_passages: List of passage dicts with 'file_path' (or 'document'), 'start_char', 'end_char'.

    Returns:
        Dict with 'precision', 'recall', 'overlap_chars', 'gt_chars', 'retrieved_chars'.
    """
    # 1. Group ground truth character intervals by document
    gt_by_doc: Dict[str, List[Tuple[int, int]]] = {}
    for s in ground_truth_snippets:
        doc = s.get("file_path", "")
        span = s.get("span", [])
        if doc and len(span) == 2:
            gt_by_doc.setdefault(doc, []).append((int(span[0]), int(span[1])))

    # 2. Group retrieved character intervals by document
    ret_by_doc: Dict[str, List[Tuple[int, int]]] = {}
    for p in retrieved_passages:
        doc = p.get("file_path") or p.get("document", "")
        start = int(p.get("start_char", 0))
        end = int(p.get("end_char", 0))
        if doc and end > start:
            ret_by_doc.setdefault(doc, []).append((start, end))

    # 3. Compute character sets and overlaps per document
    all_docs = set(gt_by_doc.keys()).union(set(ret_by_doc.keys()))
    total_overlap_chars = 0
    total_gt_chars = 0
    total_ret_chars = 0

    for doc in all_docs:
        gt_chars = _compute_character_set(gt_by_doc.get(doc, []))
        ret_chars = _compute_character_set(ret_by_doc.get(doc, []))

        total_gt_chars += len(gt_chars)
        total_ret_chars += len(ret_chars)
        total_overlap_chars += len(gt_chars.intersection(ret_chars))

    precision = (total_overlap_chars / total_ret_chars) if total_ret_chars > 0 else 0.0
    recall = (total_overlap_chars / total_gt_chars) if total_gt_chars > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "overlap_chars": total_overlap_chars,
        "gt_chars": total_gt_chars,
        "retrieved_chars": total_ret_chars,
    }


def compute_corpus_retrieval_metrics(
    eval_records: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Compute mean character-level Precision@K and Recall@K across a query dataset.

    Args:
        eval_records: List of dicts, each with:
          - 'ground_truth_snippets': List[Dict]
          - 'ranked_passages': List[Dict] (at least max(k_values) passages)
        k_values: List of cutoffs (e.g. [1, 3, 5, 10]).

    Returns:
        Dictionary mapping metric names (e.g. 'precision@1', 'recall@5') to mean float values.
    """
    if not eval_records:
        return {f"precision@{k}": 0.0 for k in k_values} | {f"recall@{k}": 0.0 for k in k_values}

    num_queries = len(eval_records)
    metric_accumulators = {f"precision@{k}": 0.0 for k in k_values}
    metric_accumulators.update({f"recall@{k}": 0.0 for k in k_values})

    for rec in eval_records:
        gt = rec.get("ground_truth_snippets", [])
        passages = rec.get("ranked_passages", [])

        for k in k_values:
            top_k_passages = passages[:k]
            res = compute_query_character_metrics(gt, top_k_passages)
            metric_accumulators[f"precision@{k}"] += res["precision"]
            metric_accumulators[f"recall@{k}"] += res["recall"]

    # Compute macro-average over queries
    return {
        metric: round(total / num_queries, 4)
        for metric, total in metric_accumulators.items()
    }


def compute_abstention_metrics(
    unanswerable_results: List[Dict[str, Any]],
    answerable_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate abstention performance on unanswerable vs answerable queries.

    Args:
        unanswerable_results: List of search result dicts for unanswerable queries.
        answerable_results: List of search result dicts for answerable queries.

    Returns:
        Detailed abstention metric dictionary.
    """
    total_unanswerable = len(unanswerable_results)
    total_answerable = len(answerable_results)

    # Unanswerable queries: desired action is abstained == True
    correct_abstentions = sum(1 for r in unanswerable_results if r.get("abstained") is True)
    incorrect_returns = total_unanswerable - correct_abstentions

    # Answerable queries: desired action is abstained == False
    false_abstentions = sum(1 for r in answerable_results if r.get("abstained") is True)
    answerable_coverage = total_answerable - false_abstentions

    correct_abstention_rate = (
        round(correct_abstentions / total_unanswerable, 4) if total_unanswerable > 0 else 0.0
    )
    false_abstention_rate = (
        round(false_abstentions / total_answerable, 4) if total_answerable > 0 else 0.0
    )
    coverage_rate = (
        round(answerable_coverage / total_answerable, 4) if total_answerable > 0 else 0.0
    )

    return {
        "unanswerable_queries": {
            "total": total_unanswerable,
            "correct_abstentions": correct_abstentions,
            "incorrect_returns": incorrect_returns,
            "correct_abstention_rate": correct_abstention_rate,
            "correct_abstention_percentage": f"{correct_abstention_rate * 100:.1f}%",
        },
        "answerable_queries": {
            "total": total_answerable,
            "false_abstentions": false_abstentions,
            "covered_queries": answerable_coverage,
            "false_abstention_rate": false_abstention_rate,
            "false_abstention_percentage": f"{false_abstention_rate * 100:.1f}%",
            "coverage_rate": coverage_rate,
            "coverage_percentage": f"{coverage_rate * 100:.1f}%",
        },
    }
