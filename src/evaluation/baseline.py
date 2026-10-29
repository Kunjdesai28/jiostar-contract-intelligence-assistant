"""Baseline retrieval evaluation runner evaluating exclusively on the training split."""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import get_project_root, get_resolved_paths, load_yaml_config
from src.evaluation.metrics import compute_abstention_metrics, compute_corpus_retrieval_metrics
from src.retrieval.pipeline import ContractRetrievalPipeline

logger = logging.getLogger(__name__)


def run_baseline_evaluation(
    config_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    split: str = "train",
) -> Dict[str, Any]:
    """Execute evaluation of the BM25 baseline retrieval system.

    STRICT GUARD: Phase 2 permits evaluation ONLY on the 'train' split.
    Attempting to evaluate on 'test' will raise a PermissionError.

    Args:
        config_path: Path to baseline YAML configuration.
        output_path: Destination for metrics JSON (defaults to experiments/baseline_train_metrics.json).
        split: Must be 'train'.

    Returns:
        Structured evaluation metrics report dictionary.
    """
    if split != "train":
        raise PermissionError(
            f"Evaluation guardrail violation: Split '{split}' is blocked in Phase 2! "
            "Phase 2 permits evaluation exclusively on the 'train' split to prevent test-set leakage."
        )

    start_time = time.time()
    root = get_project_root()
    paths = get_resolved_paths()

    # Load baseline configuration
    c_path = Path(config_path) if config_path else root / "configs" / "baseline.yaml"
    config = load_yaml_config(c_path)

    k_values = config.get("evaluation", {}).get("k_values", [1, 3, 5, 10])
    max_k = max(k_values)

    logger.info("Initializing retrieval pipeline for baseline evaluation...")
    pipeline = ContractRetrievalPipeline(config=config)
    total_indexed_chunks = pipeline.build_index()

    # 1. Load answerable training queries
    train_queries_path = paths["train_queries_output"]
    if not train_queries_path.exists():
        raise FileNotFoundError(f"Train queries missing at {train_queries_path}. Run prepare_dataset.py first.")

    train_data = json.loads(train_queries_path.read_text(encoding="utf-8"))
    answerable_train_tests: List[Dict[str, Any]] = train_data.get("tests", [])
    logger.info(f"Loaded {len(answerable_train_tests)} answerable training queries.")

    # 2. Load unanswerable training queries
    unanswerable_path = paths["unanswerable_queries_file"]
    if not unanswerable_path.exists():
        raise FileNotFoundError(f"Unanswerable queries missing at {unanswerable_path}.")

    all_neg_data = json.loads(unanswerable_path.read_text(encoding="utf-8"))
    unanswerable_train_tests = [q for q in all_neg_data if q.get("split") == "train"]
    logger.info(f"Loaded {len(unanswerable_train_tests)} unanswerable training queries (from {len(all_neg_data)} total).")

    # 3. Evaluate retrieval on answerable training queries
    eval_records: List[Dict[str, Any]] = []
    answerable_search_results: List[Dict[str, Any]] = []

    logger.info(f"Running retrieval search on {len(answerable_train_tests)} answerable queries at top_k={max_k}...")
    for idx, test_item in enumerate(answerable_train_tests):
        query = test_item["query"]
        gt_snippets = test_item.get("snippets", [])

        # Execute search without abstention filtering for raw retrieval recall/precision calculation
        raw_candidates = pipeline.search_candidates(query=query, top_k=max_k)
        eval_records.append({
            "query": query,
            "ground_truth_snippets": gt_snippets,
            "ranked_passages": raw_candidates,
        })

        # Also execute full pipeline search (with abstention engine) to measure coverage/false abstention
        pipeline_res = pipeline.search(query=query, top_k=max_k)
        answerable_search_results.append(pipeline_res)

    retrieval_metrics = compute_corpus_retrieval_metrics(eval_records, k_values=k_values)
    logger.info(f"Retrieval metrics computed: {retrieval_metrics}")

    # 4. Evaluate abstention on unanswerable training queries
    unanswerable_search_results: List[Dict[str, Any]] = []
    logger.info(f"Running retrieval search on {len(unanswerable_train_tests)} unanswerable queries...")
    for neg_item in unanswerable_train_tests:
        query = neg_item["query"]
        res = pipeline.search(query=query, top_k=max_k)
        unanswerable_search_results.append(res)

    abstention_metrics = compute_abstention_metrics(
        unanswerable_results=unanswerable_search_results,
        answerable_results=answerable_search_results,
    )
    logger.info(f"Abstention metrics computed: {abstention_metrics}")

    elapsed_time = round(time.time() - start_time, 2)

    # 5. Provenance & Artifact assembly
    prov_file = paths["raw_dir"] / "provenance.json"
    provenance_info = json.loads(prov_file.read_text(encoding="utf-8")) if prov_file.exists() else {}

    results_report = {
        "evaluation_name": "BM25 Baseline Retrieval Evaluation (Train Split)",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_time,
        "split_evaluated": split,
        "configuration": config,
        "provenance": provenance_info,
        "dataset_counts": {
            "indexed_contracts": pipeline.contracts_count,
            "indexed_chunks": total_indexed_chunks,
            "answerable_train_queries": len(answerable_train_tests),
            "unanswerable_train_queries": len(unanswerable_train_tests),
            "total_evaluated_queries": len(answerable_train_tests) + len(unanswerable_train_tests),
        },
        "retrieval_metrics_character_level": retrieval_metrics,
        "abstention_metrics": abstention_metrics,
        "metric_definitions": {
            "precision@k": "Mean character-level precision: Overlap characters between retrieved passages and ground-truth divided by total retrieved characters at cutoff K.",
            "recall@k": "Mean character-level recall: Overlap characters between retrieved passages and ground-truth divided by total ground-truth characters at cutoff K.",
            "correct_abstention_rate": "Fraction of unanswerable queries for which the system successfully returned no passages (abstained).",
            "false_abstention_rate": "Fraction of answerable queries where the system incorrectly abstained.",
            "coverage_rate": "Fraction of answerable queries where the system returned retrieved evidence.",
        },
    }

    # Save to experiments/baseline_train_metrics.json
    out_target = Path(output_path) if output_path else root / "experiments" / "baseline_train_metrics.json"
    out_target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(results_report, indent=2, ensure_ascii=False)
    out_target.write_text(serialized, encoding="utf-8")
    results_report["metrics_file_sha256"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    logger.info(f"Saved baseline evaluation report to {out_target}")

    return results_report
