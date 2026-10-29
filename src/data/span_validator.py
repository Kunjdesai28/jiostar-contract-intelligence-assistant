"""Ground-truth character offset validator for LegalBench-RAG CUAD snippets."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.data.loader import resolve_safe_corpus_path

logger = logging.getLogger(__name__)


def validate_ground_truth_spans(
    tests: List[Dict[str, Any]],
    corpus_root: Path | str,
    summary_output_path: Optional[Path | str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate every ground-truth snippet span against the exact document UTF-8 substring.

    Strict validation checks:
      1. Source document exists and is readable UTF-8.
      2. Snippet contains 'span' list [start, end] and 'answer' string.
      3. start and end are integers.
      4. 0 <= start < end <= len(document).
      5. document[start:end] == expected_answer_text (exact match without stripping or normalization).

    Args:
        tests: List of benchmark test records.
        corpus_root: Corpus root directory.
        summary_output_path: Optional path to write validation summary JSON.

    Returns:
        Tuple of (is_valid_overall, validation_summary_dict).
    """
    c_root = Path(corpus_root).resolve()

    total_queries = len(tests)
    total_snippets = 0
    valid_snippets = 0
    invalid_snippets = 0
    missing_documents: List[str] = []
    offset_mismatches: List[Dict[str, Any]] = []

    # Cache loaded contract texts in memory to avoid redundant re-reads
    doc_cache: Dict[str, str] = {}

    for t_idx, test_item in enumerate(tests):
        query = test_item.get("query", f"<query_idx_{t_idx}>")
        snippets = test_item.get("snippets", [])

        if not isinstance(snippets, list) or not snippets:
            invalid_snippets += 1
            offset_mismatches.append({
                "query_index": t_idx,
                "query": query,
                "error_type": "empty_snippets_list",
                "details": "Record has no snippets or snippets is not a list.",
            })
            continue

        for s_idx, snippet in enumerate(snippets):
            total_snippets += 1
            file_path = snippet.get("file_path")
            span = snippet.get("span")
            expected_answer = snippet.get("answer")

            if not file_path:
                invalid_snippets += 1
                offset_mismatches.append({
                    "query_index": t_idx,
                    "snippet_index": s_idx,
                    "query": query,
                    "error_type": "missing_file_path",
                    "details": "Snippet does not specify file_path.",
                })
                continue

            # Load document text
            if file_path not in doc_cache:
                try:
                    resolved_file = resolve_safe_corpus_path(c_root, file_path)
                    if not resolved_file.exists():
                        missing_documents.append(file_path)
                        invalid_snippets += 1
                        offset_mismatches.append({
                            "query_index": t_idx,
                            "snippet_index": s_idx,
                            "query": query,
                            "file_path": file_path,
                            "error_type": "missing_document",
                            "details": f"File not found: {resolved_file}",
                        })
                        continue

                    with open(resolved_file, "r", encoding="utf-8") as f:
                        doc_cache[file_path] = f.read()
                except Exception as e:
                    invalid_snippets += 1
                    offset_mismatches.append({
                        "query_index": t_idx,
                        "snippet_index": s_idx,
                        "query": query,
                        "file_path": file_path,
                        "error_type": "document_read_failure",
                        "details": str(e),
                    })
                    continue

            doc_text = doc_cache[file_path]
            doc_len = len(doc_text)

            # Validate span format
            if (
                not isinstance(span, (list, tuple))
                or len(span) != 2
                or not isinstance(span[0], int)
                or not isinstance(span[1], int)
                or isinstance(span[0], bool)
                or isinstance(span[1], bool)
            ):
                invalid_snippets += 1
                offset_mismatches.append({
                    "query_index": t_idx,
                    "snippet_index": s_idx,
                    "query": query,
                    "file_path": file_path,
                    "error_type": "invalid_span_format",
                    "details": f"Span {span} must be a 2-element integer list/tuple.",
                })
                continue

            start, end = span

            # Validate offset bounds
            if start < 0:
                invalid_snippets += 1
                offset_mismatches.append({
                    "query_index": t_idx,
                    "snippet_index": s_idx,
                    "query": query,
                    "file_path": file_path,
                    "span": [start, end],
                    "error_type": "negative_start_offset",
                    "details": f"Start offset {start} is negative.",
                })
                continue

            if end <= start:
                invalid_snippets += 1
                offset_mismatches.append({
                    "query_index": t_idx,
                    "snippet_index": s_idx,
                    "query": query,
                    "file_path": file_path,
                    "span": [start, end],
                    "error_type": "empty_or_inverted_span",
                    "details": f"End offset {end} is <= start offset {start}.",
                })
                continue

            if end > doc_len:
                invalid_snippets += 1
                offset_mismatches.append({
                    "query_index": t_idx,
                    "snippet_index": s_idx,
                    "query": query,
                    "file_path": file_path,
                    "span": [start, end],
                    "document_length": doc_len,
                    "error_type": "end_offset_out_of_bounds",
                    "details": f"End offset {end} exceeds document character length {doc_len}.",
                })
                continue

            # Validate answer text existence
            if expected_answer is None or not isinstance(expected_answer, str):
                invalid_snippets += 1
                offset_mismatches.append({
                    "query_index": t_idx,
                    "snippet_index": s_idx,
                    "query": query,
                    "file_path": file_path,
                    "error_type": "missing_answer_text",
                    "details": "Expected answer text is missing or not a string.",
                })
                continue

            # Extract substring and compare strictly
            actual_substring = doc_text[start:end]
            if actual_substring != expected_answer:
                invalid_snippets += 1
                offset_mismatches.append({
                    "query_index": t_idx,
                    "snippet_index": s_idx,
                    "query": query,
                    "file_path": file_path,
                    "span": [start, end],
                    "error_type": "text_mismatch",
                    "details": {
                        "expected_length": len(expected_answer),
                        "actual_length": len(actual_substring),
                        "expected_preview": expected_answer[:60],
                        "actual_preview": actual_substring[:60],
                    },
                })
                continue

            # If all checks pass
            valid_snippets += 1

    is_valid_overall = (invalid_snippets == 0 and len(missing_documents) == 0)

    summary = {
        "is_valid_overall": is_valid_overall,
        "total_benchmark_queries": total_queries,
        "total_annotated_snippets": total_snippets,
        "valid_snippets": valid_snippets,
        "invalid_snippets": invalid_snippets,
        "unique_documents_checked": len(doc_cache),
        "missing_documents_count": len(set(missing_documents)),
        "missing_documents": sorted(list(set(missing_documents))),
        "offset_mismatches_count": len(offset_mismatches),
        "offset_mismatches": offset_mismatches,
    }

    if is_valid_overall:
        logger.info(f"Span validation PASSED: all {valid_snippets}/{total_snippets} snippets match exact character offsets.")
    else:
        logger.error(f"Span validation FAILED: {invalid_snippets} invalid snippets, {len(missing_documents)} missing documents.")

    if summary_output_path:
        out_p = Path(summary_output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved span validation summary to {out_p}")

    return is_valid_overall, summary
