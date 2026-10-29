"""Automated Failure Diagnostics for Legal Clause Retrieval.

Offline evaluation component classifying failure modes per query:
  - wrong_document: No retrieved passage is from a document containing a gold answer.
  - wrong_section: Correct document retrieved, but 0 passages overlap gold answer span.
  - partial_span: Retrieved passage overlaps gold answer, but fails to cover complete span.
  - incorrect_abstention: System abstains despite the query being answerable.
  - failed_abstention: System returns evidence for a labeled unanswerable query.
  - successful_retrieval: At least one returned passage overlaps the annotated answer span.
"""

from typing import Any, Dict, List, Optional


def diagnose_query_result(
    query: str,
    ground_truth_snippets: List[Dict[str, Any]],
    ranked_passages: List[Dict[str, Any]],
    abstained: bool = False,
    is_unanswerable: bool = False,
) -> Dict[str, Any]:
    """Diagnose the primary failure category for a single query.

    Precedence order for answerable queries:
      1. incorrect_abstention (if abstained)
      2. wrong_document (no overlap with target documents)
      3. wrong_section (correct document found, but 0 character overlap with gold span)
      4. partial_span (character overlap exists, but complete span not covered)
      5. successful_retrieval (complete or substantial span covered)

    For unanswerable queries:
      - successful_retrieval (if abstained)
      - failed_abstention (if not abstained)

    Args:
        query: Query string.
        ground_truth_snippets: List of gold snippet dicts with 'file_path', 'start_char', 'end_char', 'text'.
        ranked_passages: Top-K retrieved passages.
        abstained: Whether the pipeline abstained.
        is_unanswerable: Whether the query is labeled unanswerable.

    Returns:
        Dict with primary category, details, and overlap statistics.
    """
    if is_unanswerable:
        if abstained or len(ranked_passages) == 0:
            return {
                "query": query,
                "category": "successful_retrieval",
                "is_unanswerable": True,
                "details": "Correctly abstained on unanswerable query.",
            }
        else:
            return {
                "query": query,
                "category": "failed_abstention",
                "is_unanswerable": True,
                "details": f"Failed to abstain on unanswerable query; returned {len(ranked_passages)} passages.",
            }

    # Answerable query
    if abstained:
        return {
            "query": query,
            "category": "incorrect_abstention",
            "is_unanswerable": False,
            "details": "Pipeline incorrectly abstained on an answerable query.",
        }

    if not ranked_passages:
        return {
            "query": query,
            "category": "wrong_document",
            "is_unanswerable": False,
            "details": "No passages returned.",
        }

    gold_docs = set()
    for s in ground_truth_snippets:
        doc = s.get("file_path") or s.get("document_id") or s.get("doc_id")
        if doc:
            gold_docs.add(doc)

    retrieved_docs = {p.get("file_path") or p.get("doc_id") for p in ranked_passages}

    # Check wrong_document
    doc_match = bool(gold_docs.intersection(retrieved_docs))
    if not doc_match and gold_docs:
        return {
            "query": query,
            "category": "wrong_document",
            "is_unanswerable": False,
            "details": f"Retrieved docs {retrieved_docs} did not intersect gold docs {gold_docs}.",
        }

    # Check span overlaps
    total_gold_chars = 0
    overlapped_chars = 0
    full_span_covered = False

    for s in ground_truth_snippets:
        s_doc = s.get("file_path") or s.get("doc_id")
        if "span" in s and len(s["span"]) == 2:
            s_start = int(s["span"][0])
            s_end = int(s["span"][1])
        else:
            s_start = int(s.get("start_char", 0))
            s_end = int(s.get("end_char", 0))

        s_text = s.get("answer") or s.get("text", "")
        total_gold_chars += (s_end - s_start) if (s_end > s_start) else len(s_text)

        for p in ranked_passages:
            p_doc = p.get("file_path") or p.get("doc_id")
            if s_doc and p_doc and s_doc != p_doc:
                continue

            p_start = p.get("start_char", 0)
            p_end = p.get("end_char", 0)

            inter_start = max(s_start, p_start)
            inter_end = min(s_end, p_end)
            if inter_end > inter_start:
                inter_len = inter_end - inter_start
                overlapped_chars += inter_len
                if p_start <= s_start and p_end >= s_end:
                    full_span_covered = True

    if overlapped_chars == 0:
        return {
            "query": query,
            "category": "wrong_section",
            "is_unanswerable": False,
            "details": "Correct document retrieved, but 0 returned passages overlap the gold answer span.",
        }

    if not full_span_covered or (total_gold_chars > 0 and overlapped_chars < total_gold_chars):
        return {
            "query": query,
            "category": "partial_span",
            "is_unanswerable": False,
            "details": f"Passage overlaps gold span ({overlapped_chars}/{total_gold_chars} chars) but does not fully cover it.",
        }

    return {
        "query": query,
        "category": "successful_retrieval",
        "is_unanswerable": False,
        "details": "Retrieved passage successfully and completely covers the annotated answer span.",
    }


def run_failure_diagnostics(
    answerable_eval_records: List[Dict[str, Any]],
    unanswerable_search_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Aggregate diagnostics across all training queries.

    Args:
        answerable_eval_records: List of eval records containing query, ground_truth_snippets, ranked_passages.
        unanswerable_search_results: Optional list of pipeline search results for unanswerable queries.

    Returns:
        Structured diagnostics report.
    """
    category_counts = {
        "wrong_document": 0,
        "wrong_section": 0,
        "partial_span": 0,
        "incorrect_abstention": 0,
        "failed_abstention": 0,
        "successful_retrieval": 0,
    }

    per_query_diagnostics = []

    for r in answerable_eval_records:
        query = r["query"]
        gt_snippets = r.get("ground_truth_snippets", [])
        passages = r.get("ranked_passages", [])
        abstained = r.get("abstained", False)

        diag = diagnose_query_result(
            query=query,
            ground_truth_snippets=gt_snippets,
            ranked_passages=passages,
            abstained=abstained,
            is_unanswerable=False,
        )
        cat = diag["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        per_query_diagnostics.append(diag)

    if unanswerable_search_results:
        for u in unanswerable_search_results:
            query = u["query"]
            abstained = u.get("abstained", False)
            passages = u.get("passages", [])
            diag = diagnose_query_result(
                query=query,
                ground_truth_snippets=[],
                ranked_passages=passages,
                abstained=abstained,
                is_unanswerable=True,
            )
            cat = diag["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
            per_query_diagnostics.append(diag)

    total_queries = len(per_query_diagnostics)
    total_errors = total_queries - category_counts["successful_retrieval"]

    # Determine dominant failure mode among errors
    error_counts = {k: v for k, v in category_counts.items() if k != "successful_retrieval"}
    dominant_failure = max(error_counts.items(), key=lambda kv: kv[1])[0] if any(error_counts.values()) else "none"

    return {
        "total_queries_evaluated": total_queries,
        "total_errors": total_errors,
        "category_counts": category_counts,
        "dominant_failure": dominant_failure,
        "wrong_document_rate": round(category_counts["wrong_document"] / max(1, total_queries), 4),
        "wrong_section_rate": round(category_counts["wrong_section"] / max(1, total_queries), 4),
        "partial_span_rate": round(category_counts["partial_span"] / max(1, total_queries), 4),
        "incorrect_abstention_rate": round(category_counts["incorrect_abstention"] / max(1, total_queries), 4),
        "failed_abstention_rate": round(category_counts["failed_abstention"] / max(1, total_queries), 4),
        "per_query_diagnostics": per_query_diagnostics,
    }
