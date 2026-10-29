"""Deterministic selection of the official LegalBench-RAG CUAD mini benchmark."""

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def mini(tests: List[Dict[str, Any]], n: int = 194) -> List[Dict[str, Any]]:
    """Prescribed deterministic selection function for LegalBench-RAG-mini CUAD.

    DO NOT ALTER THIS FUNCTION OR ITS SEEDING LOGIC.

    Args:
        tests: List of benchmark test records.
        n: Number of records to select (default 194).

    Returns:
        Sorted and sliced list of n test records.
    """
    if len(tests) <= n:
        return tests

    def key(t: Dict[str, Any]) -> float:
        # Validate that snippet structure exists
        if "snippets" not in t or not t["snippets"]:
            raise ValueError(f"Test record missing required snippets: {t.get('query', '<unknown>')}")
        first_snippet = t["snippets"][0]
        if "file_path" not in first_snippet:
            raise ValueError(f"Snippet missing 'file_path': {first_snippet}")

        random.seed(first_snippet["file_path"])
        return random.random()

    return sorted(tests, key=key)[:n]


def select_and_validate_mini_benchmark(
    raw_benchmark_path: Path | str,
    output_path: Optional[Path | str] = None,
    expected_size: int = 194,
) -> Tuple[List[Dict[str, Any]], str]:
    """Load the full CUAD benchmark, select exactly 194 mini queries, validate and save.

    Args:
        raw_benchmark_path: Path to the raw benchmarks/cuad.json file.
        output_path: Optional path to save data/processed/cuad_mini.json.
        expected_size: Required number of selected queries (must be 194).

    Returns:
        Tuple of (selected_tests, sha256_checksum).
    """
    path = Path(raw_benchmark_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw benchmark file does not exist: {path}")

    raw_text = path.read_text(encoding="utf-8")
    benchmark_data = json.loads(raw_text)

    if "tests" not in benchmark_data or not isinstance(benchmark_data["tests"], list):
        raise ValueError("Invalid benchmark structure: missing top-level 'tests' list.")

    all_tests = benchmark_data["tests"]
    logger.info(f"Loaded {len(all_tests)} tests from {path}")

    # Validate all input records have valid schema before selection
    for idx, t in enumerate(all_tests):
        if "query" not in t or not isinstance(t["query"], str) or not t["query"].strip():
            raise ValueError(f"Record at index {idx} has invalid query: {t}")
        if "snippets" not in t or not isinstance(t["snippets"], list) or not t["snippets"]:
            raise ValueError(f"Record '{t.get('query')}' has empty or invalid snippets list.")
        for s_idx, s in enumerate(t["snippets"]):
            if "file_path" not in s or "span" not in s or "answer" not in s:
                raise ValueError(f"Record '{t['query']}' snippet {s_idx} missing required fields (file_path, span, answer).")

    # Perform deterministic selection
    selected = mini(all_tests, n=expected_size)

    if len(selected) != expected_size:
        raise ValueError(f"Selected mini benchmark size mismatch: expected {expected_size}, got {len(selected)}.")

    # Check for duplicate queries
    query_list = [t["query"] for t in selected]
    unique_queries = set(query_list)
    if len(query_list) != len(unique_queries):
        dup_count = len(query_list) - len(unique_queries)
        logger.warning(f"Benchmark contains {dup_count} duplicate query strings among selected {expected_size}.")
    else:
        logger.info(f"All {len(selected)} selected queries are unique.")

    # Serialize deterministically
    serialized_payload = json.dumps({"tests": selected}, indent=2, ensure_ascii=False)
    checksum = hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(serialized_payload, encoding="utf-8")
        logger.info(f"Saved mini benchmark to {out} (SHA-256: {checksum})")

    return selected, checksum
