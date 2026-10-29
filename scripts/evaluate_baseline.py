#!/usr/bin/env python3
"""Run baseline retrieval evaluation on the train split and record scientific metrics."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation.baseline import run_baseline_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate BM25 Baseline Retrieval System on Train Split")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to baseline YAML config (default: configs/baseline.yaml)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train"],
        help="Evaluation split. MANDATORY GUARD: Only 'train' is permitted in Phase 2.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/baseline_train_metrics.json",
        help="Output metrics JSON file path.",
    )

    args = parser.parse_args()

    print("=" * 75)
    print("JioStar Contract Intelligence Assistant - Baseline Retrieval Evaluation")
    print("=" * 75)

    if args.split != "train":
        print(f"[FATAL GUARD ERROR] Split '{args.split}' is strictly blocked in Phase 2!", file=sys.stderr)
        return 1

    try:
        results = run_baseline_evaluation(
            config_path=args.config,
            output_path=args.output,
            split=args.split,
        )

        counts = results["dataset_counts"]
        ret = results["retrieval_metrics_character_level"]
        abst = results["abstention_metrics"]

        print(f"\n[SUCCESS] Baseline evaluation completed in {results['elapsed_seconds']}s.")
        print(f"  Evaluated Split:              {results['split_evaluated']}")
        print(f"  Indexed Contracts / Chunks:   {counts['indexed_contracts']} contracts / {counts['indexed_chunks']} chunks")
        print(f"  Answerable Train Queries:     {counts['answerable_train_queries']}")
        print(f"  Unanswerable Train Queries:   {counts['unanswerable_train_queries']}")
        print(f"  Total Queries Evaluated:      {counts['total_evaluated_queries']}")

        print("\n--- Character-Level Retrieval Metrics ---")
        for k in [1, 3, 5, 10]:
            print(f"  Precision@{k:<2d}: {ret.get(f'precision@{k}', 0.0):.4f}   |   Recall@{k:<2d}: {ret.get(f'recall@{k}', 0.0):.4f}")

        print("\n--- Abstention Performance ---")
        u_info = abst["unanswerable_queries"]
        a_info = abst["answerable_queries"]
        print(f"  Correct Abstention on Negatives: {u_info['correct_abstentions']}/{u_info['total']} ({u_info['correct_abstention_percentage']})")
        print(f"  False Abstention on Positives:   {a_info['false_abstentions']}/{a_info['total']} ({a_info['false_abstention_percentage']})")
        print(f"  Answerable Query Coverage:       {a_info['covered_queries']}/{a_info['total']} ({a_info['coverage_percentage']})")

        print(f"\nMetrics written to: {args.output}")
        print("=" * 75)
        return 0

    except Exception as e:
        print(f"\n[ERROR] Baseline evaluation failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
