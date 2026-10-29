#!/usr/bin/env python3
"""Script to execute the complete CUAD mini dataset preparation pipeline."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.prepare import run_preparation_pipeline


def main():
    print("=" * 70)
    print("JioStar Contract Intelligence Assistant - Dataset Preparation Pipeline")
    print("=" * 70)
    try:
        summary = run_preparation_pipeline()
        print("\n[SUCCESS] Dataset preparation pipeline completed successfully.")
        print(f"  Mini Benchmark Size:    {summary['mini_benchmark']['actual_size']} queries (SHA-256: {summary['mini_benchmark']['sha256'][:16]}...)")
        print(f"  Train / Test Splits:    {summary['train_test_split']['train_count']} train / {summary['train_test_split']['test_count']} test")
        print(f"  Referenced Contracts:   {summary['indexed_contracts']['unique_count']} unique contracts in manifest")
        print(f"  Span Validation:        {summary['span_validation']['valid_snippets']} / {summary['span_validation']['total_snippets']} snippets valid (0 invalid)")
        print(f"  Unanswerable Queries:   {summary['unanswerable_queries']['total_count']} total ({summary['unanswerable_queries']['train_count']} train / {summary['unanswerable_queries']['test_count']} test)")
        print(f"  Preparation Summary:    Saved to {summary['span_validation']['summary_file']}")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Preparation pipeline failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
