"""Utility script to compare experiment iterations from improvement logs."""

import json
from pathlib import Path
import sys

def main() -> int:
    summary_path = Path("experiments/improvement_summary.json")
    if not summary_path.exists():
        print(f"Error: {summary_path} does not exist. Run scripts/run_improvement_loop.py first.")
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("\n" + "=" * 90)
    print("JIOSTAR CONTRACT RETRIEVAL — EXPERIMENT ITERATION COMPARISON")
    print("=" * 90)

    perf = summary.get("performance_comparison_training", {})
    rec5 = perf.get("character_recall_at_5", {})
    prec5 = perf.get("character_precision_at_5", {})

    print(f"\nBaseline Recall@5:    {rec5.get('baseline'):.4f}")
    print(f"Final Recall@5:       {rec5.get('final'):.4f} (Relative Gain: +{rec5.get('relative_gain_percent', 0):.2f}%)")
    print(f"Baseline Precision@5: {prec5.get('baseline'):.4f}")
    print(f"Final Precision@5:    {prec5.get('final'):.4f}")

    print("\nITERATION LOG TABLE:")
    header = f"{'Iter':<5} | {'Candidate ID':<14} | {'Change Parameter':<25} | {'Rec@5':<8} | {'Prec@5':<8} | {'Decision':<10} | {'Duration':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for it in summary.get("iterations", []):
        it_num = it.get("iteration")
        cand_id = it.get("candidate_config_id")
        param = str(it.get("proposed_change", {}).get("parameter", ""))[:25]
        rec = it.get("metrics_after", {}).get("recall_at_5", 0.0)
        prec = it.get("metrics_after", {}).get("precision_at_5", 0.0)
        dec = it.get("decision", "").upper()
        dur = f"{it.get('duration_seconds', 0.0):.1f}s"
        print(f"{it_num:<5} | {cand_id:<14} | {param:<25} | {rec:<8.4f} | {prec:<8.4f} | {dec:<10} | {dur:<8}")

    print("-" * len(header))
    return 0

if __name__ == "__main__":
    sys.exit(main())
