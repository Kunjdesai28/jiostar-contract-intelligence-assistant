#!/usr/bin/env python3
"""Dataset inspection utility to report live metrics computed from actual files."""

import hashlib
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import get_resolved_paths, load_yaml_config


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file if it exists."""
    if not path.exists():
        return "N/A (file not found)"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    print("=" * 75)
    print("JioStar Contract Intelligence Assistant - Dataset Inspection Report")
    print("=" * 75)

    config = load_yaml_config()
    paths = get_resolved_paths(config)

    # 1. Dataset Provenance
    prov_path = paths["raw_dir"] / "provenance.json"
    if prov_path.exists():
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        print("\n1. DATASET PROVENANCE")
        print(f"   Source Repository:      {prov.get('dataset_source')}")
        print(f"   Configured Revision:    {prov.get('revision_configured')}")
        print(f"   Verified Commit SHA:    {prov.get('commit_sha')}")
        print(f"   Acquired UTC:           {prov.get('acquired_at_utc')}")
        print(f"   Raw Corpus Files:       {prov.get('corpus_file_count')}")
    else:
        print("\n1. DATASET PROVENANCE: Not found (run scripts/download_dataset.py)")

    # 2. Original CUAD Benchmark
    raw_benchmark = paths["raw_dir"] / config["paths"]["benchmark_file"]
    if raw_benchmark.exists():
        raw_b_data = json.loads(raw_benchmark.read_text(encoding="utf-8"))
        raw_tests = raw_b_data.get("tests", [])
        print("\n2. ORIGINAL CUAD BENCHMARK")
        print(f"   Total Test Records:     {len(raw_tests)}")
        print(f"   Raw File SHA-256:       {compute_file_sha256(raw_benchmark)}")
    else:
        print("\n2. ORIGINAL CUAD BENCHMARK: Raw benchmark missing.")

    # 3. Mini Benchmark & Splits
    mini_path = paths["mini_benchmark_output"]
    train_path = paths["train_queries_output"]
    test_path = paths["test_queries_output"]

    if mini_path.exists():
        mini_data = json.loads(mini_path.read_text(encoding="utf-8"))
        mini_tests = mini_data.get("tests", [])
        print("\n3. MINI BENCHMARK (LEGALBENCH-RAG)")
        print(f"   Selected Mini Size:     {len(mini_tests)} (Expected: 194)")
        print(f"   Mini File SHA-256:      {compute_file_sha256(mini_path)}")

        if train_path.exists() and test_path.exists():
            train_tests = json.loads(train_path.read_text(encoding="utf-8")).get("tests", [])
            test_tests = json.loads(test_path.read_text(encoding="utf-8")).get("tests", [])
            print(f"   Train Split Queries:    {len(train_tests)} ({len(train_tests)/len(mini_tests)*100:.1f}%)")
            print(f"   Test Split Queries:     {len(test_tests)} ({len(test_tests)/len(mini_tests)*100:.1f}%)")
            print(f"   Train File SHA-256:     {compute_file_sha256(train_path)}")
            print(f"   Test File SHA-256:      {compute_file_sha256(test_path)}")
    else:
        print("\n3. MINI BENCHMARK: Not prepared yet (run scripts/prepare_dataset.py)")

    # 4. Referenced Contract Scope
    manifest_path = paths["contract_manifest_output"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contracts = manifest.get("contracts", [])
        total_chars = sum(c.get("char_length", 0) for c in contracts)
        print("\n4. INDEXED CONTRACT SCOPE")
        print(f"   Unique Contracts:       {len(contracts)}")
        print(f"   Total Corpus Chars:     {total_chars:,} characters")
        print(f"   Manifest SHA-256:       {compute_file_sha256(manifest_path)}")
    else:
        print("\n4. INDEXED CONTRACT SCOPE: Manifest missing.")

    # 5. Span Validation Summary
    val_path = paths["validation_summary_output"]
    if val_path.exists():
        val = json.loads(val_path.read_text(encoding="utf-8"))
        print("\n5. GROUND-TRUTH SPAN VALIDATION")
        print(f"   Overall Validity:       {'PASSED (100% exact substring match)' if val.get('is_valid_overall') else 'FAILED'}")
        print(f"   Total Annotated Spans:  {val.get('total_annotated_snippets')}")
        print(f"   Valid Spans:            {val.get('valid_snippets')}")
        print(f"   Invalid Spans:          {val.get('invalid_snippets')}")
        print(f"   Missing Documents:      {val.get('missing_documents_count')}")
    else:
        print("\n5. GROUND-TRUTH SPAN VALIDATION: Validation summary missing.")

    # 6. Unanswerable Negative Queries
    neg_path = paths["unanswerable_queries_file"]
    if neg_path.exists():
        neg_data = json.loads(neg_path.read_text(encoding="utf-8"))
        train_neg = sum(1 for q in neg_data if q.get("split") == "train")
        test_neg = sum(1 for q in neg_data if q.get("split") == "test")
        categories = {}
        for q in neg_data:
            c = q.get("category", "unknown")
            categories[c] = categories.get(c, 0) + 1

        print("\n6. UNANSWERABLE QUERIES BENCHMARK")
        print(f"   Total Negative Queries: {len(neg_data)} (>= 30 required)")
        print(f"   Train Negative Queries: {train_neg}")
        print(f"   Test Negative Queries:  {test_neg}")
        print(f"   Negative Set SHA-256:   {compute_file_sha256(neg_path)}")
        print("   Category Breakdown:")
        for cat, cnt in sorted(categories.items()):
            print(f"     - {cat:28s}: {cnt}")
    else:
        print("\n6. UNANSWERABLE QUERIES BENCHMARK: File missing.")

    print("\n" + "=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
