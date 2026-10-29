"""Predeclared Experimental Objective and Guardrails Specification.

The experimental objective and guardrails are frozen prior to running candidate experiments:
  Primary Objective:
    - Maximize macro-averaged character Recall@5 on the 140 answerable training queries.

  Guardrails:
    1. Recall@5 improvement: candidate_recall_at_5 > accepted_recall_at_5 + 1e-6
    2. Precision@5 relative tolerance: candidate_precision_at_5 >= accepted_precision_at_5 * 0.80 (max 20% relative drop)
    3. Correct negative abstention rate: candidate_correct_abstention >= accepted_correct_abstention - 0.10 (max 10pp drop)
    4. False abstention rate: candidate_false_abstention <= accepted_false_abstention + 0.05 (max 5pp increase)
    5. Mean query latency: candidate_latency <= baseline_latency * 3.0 (max 3x baseline latency)
    6. System validity: Complete execution without runtime exceptions or missing metrics.
"""

from typing import Any, Dict


def get_default_objective_spec() -> Dict[str, Any]:
    """Return the frozen primary objective and guardrails definition."""
    return {
        "primary_objective": {
            "name": "character_recall_at_5",
            "direction": "maximize",
            "split": "train",
            "evaluated_queries": 140,
            "min_delta": 1e-6,
            "description": "Macro-averaged character Recall@5 on 140 answerable training queries.",
        },
        "secondary_metrics": [
            "character_precision_at_5",
            "character_recall_at_1",
            "character_recall_at_3",
            "character_recall_at_10",
            "correct_abstention_rate",
            "false_abstention_rate",
            "mean_latency_ms",
            "indexed_chunks_count",
        ],
        "guardrails": {
            "max_precision_at_5_relative_drop": 0.20,
            "max_correct_abstention_drop_pp": 0.10,
            "max_false_abstention_increase_pp": 0.05,
            "max_latency_multiplier": 3.0,
            "require_valid_execution": True,
        },
    }
