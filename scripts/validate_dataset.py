#!/usr/bin/env python3
"""Dataset validation script to verify all ground-truth spans, splits, and negative queries."""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import get_resolved_paths, load_yaml_config
from src.data.loader import resolve_safe_corpus_path
from src.data.negative_queries import validate_negative_queries
from src.data.span_validator import validate_ground_truth_spans
from src.data.splitter import assign_split


def main():
    print("=" * 70)
    print("JioStar Contract Intelligence Assistant - Dataset Validation")
    print("=" * 70)

    config = load_yaml_config()
    paths = get_resolved_paths(config)

    failures = []

    # 1. Validate mini benchmark file
    mini_p = paths["mini_benchmark_output"]
    if not mini_p.exists():
        print(f"[FAIL] Mini benchmark file does not exist: {mini_p}")
        return 1

    mini_data = json.loads(mini_p.read_text(encoding="utf-8"))
    tests = mini_data.get("tests", [])
    if len(tests) != 194:
        failures.append(f"Mini benchmark expected 194 queries, found {len(tests)}")
    else:
        print(f"[PASS] Mini benchmark contains exactly 194 queries.")

    # 2. Validate Ground-Truth Spans
    corpus_root = paths["raw_dir"] / "corpus"
    is_valid_spans, span_summary = validate_ground_truth_spans(tests, corpus_root)
    if not is_valid_spans or span_summary["invalid_snippets"] > 0:
        failures.append(f"Span validation failed: {span_summary['invalid_snippets']} invalid snippets.")
    else:
        print(f"[PASS] All {span_summary['valid_snippets']}/{span_summary['total_annotated_snippets']} ground-truth spans match document text exactly.")

    # 3. Validate Train/Test Splits
    train_p = paths["train_queries_output"]
    test_p = paths["test_queries_output"]
    if not train_p.exists() or not test_p.exists():
        failures.append("Train or test query output file missing.")
    else:
        train_data = json.loads(train_p.read_text(encoding="utf-8")).get("tests", [])
        test_data = json.loads(test_p.read_text(encoding="utf-8")).get("tests", [])
        total_split = len(train_data) + len(test_data)
        if total_split != 194:
            failures.append(f"Train+Test count mismatch: {total_split} != 194")
        else:
            print(f"[PASS] Split counts: {len(train_data)} train / {len(test_data)} test (Total: 194).")

        # Verify each record split hash
        for t in train_data:
            if assign_split(t["query"]) != "train":
                failures.append(f"Query assigned to train split hashes to test: '{t['query']}'")
        for t in test_data:
            if assign_split(t["query"]) != "test":
                failures.append(f"Query assigned to test split hashes to train: '{t['query']}'")

    # 4. Validate Contract Manifest
    manifest_p = paths["contract_manifest_output"]
    if not manifest_p.exists():
        failures.append("Contract manifest file missing.")
    else:
        manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
        contracts = manifest_data.get("contracts", [])
        print(f"[PASS] Contract manifest contains {len(contracts)} unique referenced contracts.")
        for c in contracts:
            cpath = Path(c["local_file_path"])
            if not cpath.exists():
                failures.append(f"Contract in manifest does not exist on disk: {cpath}")

    # 5. Validate Negative Queries
    neg_p = paths["unanswerable_queries_file"]
    if not neg_p.exists():
        failures.append("Unanswerable queries file missing.")
    else:
        is_neg_valid, neg_report = validate_negative_queries(neg_p)
        if not is_neg_valid:
            failures.append(f"Negative queries failed validation: {neg_report['errors']}")
        else:
            print(f"[PASS] Unanswerable queries: {neg_report['total_queries']} questions valid ({neg_report['train_count']} train / {neg_report['test_count']} test).")

    print("-" * 70)
    if failures:
        print(f"[RESULT] Validation FAILED with {len(failures)} issue(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("[RESULT] All validation checks PASSED successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
