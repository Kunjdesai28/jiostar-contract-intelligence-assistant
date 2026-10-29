"""Append-only experiment logger and summary generator for retrieval self-improvement."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Logs improvement iterations to append-only JSONL and writes run summaries."""

    def __init__(
        self,
        log_path: Optional[Path | str] = None,
        summary_path: Optional[Path | str] = None,
    ):
        """Initialize experiment logger.

        Args:
            log_path: Path to improvement_log.jsonl.
            summary_path: Path to improvement_summary.json.
        """
        self.log_path = Path(log_path) if log_path else Path("experiments/improvement_log.jsonl")
        self.summary_path = Path(summary_path) if summary_path else Path("experiments/improvement_summary.json")

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)

    def log_iteration(self, record: Dict[str, Any]) -> None:
        """Append an iteration record to improvement_log.jsonl.

        Args:
            record: Structured experiment record dictionary.
        """
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        serialized = json.dumps(record, ensure_ascii=False)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(serialized + "\n")
        logger.info(
            f"Logged iteration {record.get('iteration')} ({record.get('decision')}) to {self.log_path}"
        )

    def write_summary(
        self,
        baseline_metrics: Dict[str, Any],
        final_metrics: Dict[str, Any],
        iteration_records: List[Dict[str, Any]],
        objective_spec: Dict[str, Any],
        total_duration_seconds: float,
    ) -> Dict[str, Any]:
        """Generate and save end-to-end experiment summary to JSON.

        Args:
            baseline_metrics: Metrics from original baseline.
            final_metrics: Metrics from the winning accepted configuration.
            iteration_records: List of all iteration results.
            objective_spec: Predeclared objective and guardrails.
            total_duration_seconds: Total loop execution wall-clock time.

        Returns:
            Summary dictionary.
        """
        accepted_count = sum(1 for r in iteration_records if r.get("decision") == "accepted")
        rejected_count = sum(1 for r in iteration_records if r.get("decision") == "rejected")
        failed_count = sum(1 for r in iteration_records if r.get("decision") == "failed")

        base_k = baseline_metrics.get("retrieval_metrics_character_level", {})
        final_k = final_metrics.get("retrieval_metrics_character_level", {})

        summary = {
            "title": "JioStar Contract Retrieval Self-Improvement Loop Summary",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_duration_seconds": round(total_duration_seconds, 2),
            "total_iterations_executed": len(iteration_records),
            "iterations_accepted": accepted_count,
            "iterations_rejected": rejected_count,
            "iterations_failed": failed_count,
            "predeclared_objective": objective_spec,
            "baseline_configuration": baseline_metrics.get("configuration"),
            "final_selected_configuration": final_metrics.get("configuration"),
            "performance_comparison_training": {
                "character_precision_at_1": {
                    "baseline": base_k.get("precision@1"),
                    "final": final_k.get("precision@1"),
                    "delta": round(final_k.get("precision@1", 0) - base_k.get("precision@1", 0), 4),
                },
                "character_recall_at_1": {
                    "baseline": base_k.get("recall@1"),
                    "final": final_k.get("recall@1"),
                    "delta": round(final_k.get("recall@1", 0) - base_k.get("recall@1", 0), 4),
                },
                "character_precision_at_5": {
                    "baseline": base_k.get("precision@5"),
                    "final": final_k.get("precision@5"),
                    "delta": round(final_k.get("precision@5", 0) - base_k.get("precision@5", 0), 4),
                },
                "character_recall_at_5": {
                    "baseline": base_k.get("recall@5"),
                    "final": final_k.get("recall@5"),
                    "delta": round(final_k.get("recall@5", 0) - base_k.get("recall@5", 0), 4),
                    "relative_gain_percent": round(
                        ((final_k.get("recall@5", 0) - base_k.get("recall@5", 0)) / max(1e-6, base_k.get("recall@5", 0))) * 100,
                        2,
                    ),
                },
                "character_recall_at_10": {
                    "baseline": base_k.get("recall@10"),
                    "final": final_k.get("recall@10"),
                    "delta": round(final_k.get("recall@10", 0) - base_k.get("recall@10", 0), 4),
                },
                "correct_abstention_rate": {
                    "baseline": baseline_metrics.get("abstention_metrics", {}).get("correct_abstention_rate")
                    or baseline_metrics.get("abstention_metrics", {}).get("unanswerable_queries", {}).get("correct_abstention_rate", 0.0),
                    "final": final_metrics.get("abstention_metrics", {}).get("correct_abstention_rate")
                    or final_metrics.get("abstention_metrics", {}).get("unanswerable_queries", {}).get("correct_abstention_rate", 0.0),
                },
                "false_abstention_rate": {
                    "baseline": baseline_metrics.get("abstention_metrics", {}).get("false_abstention_rate")
                    or baseline_metrics.get("abstention_metrics", {}).get("answerable_queries", {}).get("false_abstention_rate", 0.0),
                    "final": final_metrics.get("abstention_metrics", {}).get("false_abstention_rate")
                    or final_metrics.get("abstention_metrics", {}).get("answerable_queries", {}).get("false_abstention_rate", 0.0),
                },
            },
            "iterations": iteration_records,
        }

        self.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved experiment summary to {self.summary_path}")
        return summary
