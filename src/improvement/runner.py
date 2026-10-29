"""Automated Self-Improvement Loop Runner.

Executes iterative retrieval self-improvement:
  1. Verifies dataset checksums and baseline integrity.
  2. Diagnoses failure patterns on training data.
  3. Proposes controlled, single-variable configuration modifications.
  4. Evaluates candidates on training data (140 answerable + 19 unanswerable).
  5. Applies predeclared acceptance/rejection rules against frozen guardrails.
  6. Updates accepted reference configuration dynamically upon acceptance.
  7. Logs all iterations to append-only experiments/improvement_log.jsonl.
  8. Emits final winning configuration to configs/final.yaml.
"""

import copy
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.config import get_project_root, get_resolved_paths, load_yaml_config
from src.evaluation.diagnostics import run_failure_diagnostics
from src.evaluation.metrics import compute_abstention_metrics, compute_corpus_retrieval_metrics
from src.improvement.decision import evaluate_experiment_decision
from src.improvement.experiment_logger import ExperimentLogger
from src.improvement.objectives import get_default_objective_spec
from src.improvement.proposer import ExperimentProposer
from src.retrieval.pipeline import ContractRetrievalPipeline

logger = logging.getLogger(__name__)

FROZEN_NEGATIVE_QUERY_CHECKSUM = "7eaf304f4a22a4567100fb283a53f1bfc7b458eaa2f7feb7d78ba66db3d539d3"


def evaluate_pipeline_on_training_split(
    pipeline: ContractRetrievalPipeline,
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, Any]:
    """Evaluate a configured retrieval pipeline exclusively on the training split.

    Strict guardrail: Evaluates ONLY the 140 answerable train queries and
    19 unanswerable train queries. Never touches held-out test data.

    Args:
        pipeline: Initialized ContractRetrievalPipeline.
        k_values: Cutoff ranks for evaluation.

    Returns:
        Structured evaluation metrics dict including character metrics, abstention, and diagnostics.
    """
    start_time = time.time()
    paths = get_resolved_paths()
    max_k = max(k_values)

    # 1. Load answerable training queries
    train_queries_path = paths["train_queries_output"]
    train_data = json.loads(train_queries_path.read_text(encoding="utf-8"))
    answerable_train_tests: List[Dict[str, Any]] = train_data.get("tests", [])

    # 2. Load unanswerable training queries
    unanswerable_path = paths["unanswerable_queries_file"]
    all_neg_data = json.loads(unanswerable_path.read_text(encoding="utf-8"))
    unanswerable_train_tests = [q for q in all_neg_data if q.get("split") == "train"]

    # 3. Build index
    total_chunks = pipeline.build_index()

    # 4. Search answerable queries
    eval_records: List[Dict[str, Any]] = []
    answerable_search_results: List[Dict[str, Any]] = []

    for test_item in answerable_train_tests:
        query = test_item["query"]
        gt_snippets = test_item.get("snippets", [])

        # Retrieve raw candidates for retrieval recall/precision
        raw_candidates = pipeline.search_candidates(query=query, top_k=max_k)
        eval_records.append({
            "query": query,
            "ground_truth_snippets": gt_snippets,
            "ranked_passages": raw_candidates,
        })

        # Pipeline search for abstention/coverage measurement
        pipe_res = pipeline.search(query=query, top_k=max_k)
        answerable_search_results.append(pipe_res)

    retrieval_metrics = compute_corpus_retrieval_metrics(eval_records, k_values=k_values)

    # 5. Search unanswerable queries
    unanswerable_search_results: List[Dict[str, Any]] = []
    for neg_item in unanswerable_train_tests:
        query = neg_item["query"]
        pipe_res = pipeline.search(query=query, top_k=max_k)
        unanswerable_search_results.append(pipe_res)

    abstention_metrics = compute_abstention_metrics(
        unanswerable_results=unanswerable_search_results,
        answerable_results=answerable_search_results,
    )

    # 6. Run diagnostics
    diagnostics = run_failure_diagnostics(
        answerable_eval_records=eval_records,
        unanswerable_search_results=unanswerable_search_results,
    )

    elapsed_time = round(time.time() - start_time, 2)

    return {
        "elapsed_seconds": elapsed_time,
        "split_evaluated": "train",
        "configuration": pipeline.config,
        "dataset_counts": {
            "indexed_contracts": pipeline.contracts_count,
            "indexed_chunks": total_chunks,
            "answerable_train_queries": len(answerable_train_tests),
            "unanswerable_train_queries": len(unanswerable_train_tests),
            "total_evaluated_queries": len(answerable_train_tests) + len(unanswerable_train_tests),
        },
        "retrieval_metrics_character_level": retrieval_metrics,
        "abstention_metrics": abstention_metrics,
        "diagnostics": diagnostics,
    }


class ImprovementLoopRunner:
    """Orchestrates the automated retrieval self-improvement loop."""

    def __init__(
        self,
        config_path: Optional[Path | str] = None,
        iterations: int = 5,
        log_path: Optional[Path | str] = None,
        summary_path: Optional[Path | str] = None,
        final_config_path: Optional[Path | str] = None,
    ):
        """Initialize ImprovementLoopRunner.

        Args:
            config_path: Path to improvement_search.yaml or baseline.yaml.
            iterations: Number of candidate iterations to execute (minimum 5).
            log_path: Path for append-only JSONL experiment logs.
            summary_path: Path for JSON experiment summary.
            final_config_path: Output path for winning configuration.
        """
        root = get_project_root()
        self.root = root
        self.iterations = max(5, int(iterations))
        self.config_path = Path(config_path) if config_path else root / "configs" / "improvement_search.yaml"
        self.final_config_path = Path(final_config_path) if final_config_path else root / "configs" / "final.yaml"

        self.logger = ExperimentLogger(log_path=log_path, summary_path=summary_path)
        self.proposer = ExperimentProposer()
        self.objective_spec = get_default_objective_spec()

    def verify_integrity(self) -> None:
        """Verify immutable dataset artifacts and checksums prior to running."""
        paths = get_resolved_paths()
        unans_file = paths["unanswerable_queries_file"]
        if not unans_file.exists():
            raise FileNotFoundError(f"Unanswerable queries file missing at {unans_file}")

        file_bytes = unans_file.read_bytes()
        actual_sha = hashlib.sha256(file_bytes).hexdigest()
        if actual_sha != FROZEN_NEGATIVE_QUERY_CHECKSUM:
            raise ValueError(
                f"Data integrity violation: Negative-query checksum mismatch! Expected {FROZEN_NEGATIVE_QUERY_CHECKSUM}, got {actual_sha}"
            )
        logger.info("Data integrity verified: negative query SHA-256 checksum confirmed.")

    def run(self) -> Dict[str, Any]:
        """Execute the automated self-improvement loop.

        Returns:
            End-to-end experiment summary dictionary.
        """
        loop_start_time = time.time()
        self.verify_integrity()

        # Step 1: Load baseline configuration
        baseline_cfg_path = self.root / "configs" / "baseline.yaml"
        baseline_config = load_yaml_config(baseline_cfg_path)
        logger.info(f"Loaded baseline configuration from {baseline_cfg_path}")

        # Step 2: Evaluate baseline on training data to establish immutable reference
        logger.info("Evaluating BM25 baseline on training split...")
        baseline_pipeline = ContractRetrievalPipeline(config=baseline_config)
        baseline_metrics = evaluate_pipeline_on_training_split(baseline_pipeline)
        base_rec5 = baseline_metrics["retrieval_metrics_character_level"]["recall@5"]
        base_prec5 = baseline_metrics["retrieval_metrics_character_level"]["precision@5"]
        logger.info(f"Baseline training metrics: Recall@5={base_rec5:.4f}, Precision@5={base_prec5:.4f}")

        # Step 3: Initialize current accepted configuration and state
        current_accepted_config = copy.deepcopy(baseline_config)
        current_accepted_metrics = copy.deepcopy(baseline_metrics)
        current_accepted_id = "bm25_baseline"

        iteration_records: List[Dict[str, Any]] = []

        # Step 4: Execute iterations 1..N
        for it in range(1, self.iterations + 1):
            it_start_time = time.time()
            logger.info(f"\n{'='*70}\nSTARTING IMPROVEMENT ITERATION {it}/{self.iterations}\n{'='*70}")

            # 4a. Diagnose failures on current accepted state
            diagnostics = current_accepted_metrics.get("diagnostics", {})
            dom_failure = diagnostics.get("dominant_failure", "wrong_section")
            logger.info(f"Iteration {it}: Current accepted ID: '{current_accepted_id}', Dominant failure: '{dom_failure}'")

            # 4b. Propose configuration candidate
            proposal = self.proposer.propose(
                iteration=it,
                parent_config=current_accepted_config,
                diagnostics=diagnostics,
            )
            cand_config = proposal["candidate_config"]
            cand_id = proposal["candidate_config_id"]
            logger.info(f"Proposed candidate {cand_id}: {proposal['proposed_change']}")

            # 4c. Build candidate pipeline & evaluate on training split
            try:
                candidate_pipeline = ContractRetrievalPipeline(config=cand_config)
                candidate_metrics = evaluate_pipeline_on_training_split(candidate_pipeline)
                c_rec5 = candidate_metrics["retrieval_metrics_character_level"]["recall@5"]
                c_prec5 = candidate_metrics["retrieval_metrics_character_level"]["precision@5"]
                logger.info(f"Evaluated candidate {cand_id}: Recall@5={c_rec5:.4f}, Precision@5={c_prec5:.4f}")

                # 4d. Apply predeclared acceptance/rejection rule
                decision, reason, guardrail_results = evaluate_experiment_decision(
                    accepted_metrics=current_accepted_metrics,
                    candidate_metrics=candidate_metrics,
                    baseline_metrics=baseline_metrics,
                    spec=self.objective_spec,
                )
                eval_status = "evaluated"

            except Exception as e:
                logger.error(f"Error evaluating candidate {cand_id}: {e}", exc_info=True)
                decision = "failed"
                reason = f"Runtime evaluation error: {str(e)}"
                guardrail_results = {"error": str(e)}
                candidate_metrics = {}
                eval_status = "failed"

            it_duration = round(time.time() - it_start_time, 2)
            logger.info(f"DECISION for {cand_id}: {decision.upper()} - {reason}")

            # Assemble iteration record
            before_abs = current_accepted_metrics["abstention_metrics"]
            rec_before = {
                "recall_at_5": current_accepted_metrics["retrieval_metrics_character_level"]["recall@5"],
                "precision_at_5": current_accepted_metrics["retrieval_metrics_character_level"]["precision@5"],
                "recall_at_1": current_accepted_metrics["retrieval_metrics_character_level"].get("recall@1"),
                "recall_at_10": current_accepted_metrics["retrieval_metrics_character_level"].get("recall@10"),
                "correct_abstention_rate": before_abs.get("correct_abstention_rate") or before_abs.get("unanswerable_queries", {}).get("correct_abstention_rate", 0.0),
                "false_abstention_rate": before_abs.get("false_abstention_rate") or before_abs.get("answerable_queries", {}).get("false_abstention_rate", 0.0),
            }
            after_abs = candidate_metrics.get("abstention_metrics", {})
            rec_after = (
                {
                    "recall_at_5": candidate_metrics["retrieval_metrics_character_level"]["recall@5"],
                    "precision_at_5": candidate_metrics["retrieval_metrics_character_level"]["precision@5"],
                    "recall_at_1": candidate_metrics["retrieval_metrics_character_level"].get("recall@1"),
                    "recall_at_10": candidate_metrics["retrieval_metrics_character_level"].get("recall@10"),
                    "correct_abstention_rate": after_abs.get("correct_abstention_rate") or after_abs.get("unanswerable_queries", {}).get("correct_abstention_rate", 0.0),
                    "false_abstention_rate": after_abs.get("false_abstention_rate") or after_abs.get("answerable_queries", {}).get("false_abstention_rate", 0.0),
                }
                if eval_status == "evaluated"
                else {}
            )

            record = {
                "iteration": it,
                "proposer": "rule_based",
                "parent_config_id": current_accepted_id,
                "candidate_config_id": cand_id,
                "proposed_change": proposal["proposed_change"],
                "scientific_rationale": proposal["scientific_rationale"],
                "expected_direction": proposal["expected_direction"],
                "diagnosis": {
                    "dominant_failure": dom_failure,
                    "reason": f"Dominant failure mode measured on training split ({diagnostics.get('total_errors', 0)} total errors)",
                },
                "metrics_before": rec_before,
                "metrics_after": rec_after,
                "guardrail_results": guardrail_results,
                "decision": decision,
                "decision_reason": reason,
                "evaluation_split": "train",
                "duration_seconds": it_duration,
                "candidate_config": cand_config,
            }

            self.logger.log_iteration(record)
            iteration_records.append(record)

            # Step 5: State update if accepted
            if decision == "accepted":
                logger.info(f"UPDATING ACCEPTED STATE: {current_accepted_id} -> {cand_id}")
                current_accepted_config = copy.deepcopy(cand_config)
                current_accepted_metrics = copy.deepcopy(candidate_metrics)
                current_accepted_id = cand_id
            else:
                logger.info(f"RETAINING PREVIOUS ACCEPTED STATE: {current_accepted_id}")

        total_loop_time = time.time() - loop_start_time

        # Step 6: Save final accepted configuration to configs/final.yaml
        self.final_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.final_config_path, "w", encoding="utf-8") as f:
            yaml.dump(current_accepted_config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved final selected configuration to {self.final_config_path}")

        # Step 7: Write full experiment summary
        summary = self.logger.write_summary(
            baseline_metrics=baseline_metrics,
            final_metrics=current_accepted_metrics,
            iteration_records=iteration_records,
            objective_spec=self.objective_spec,
            total_duration_seconds=total_loop_time,
        )

        return summary
