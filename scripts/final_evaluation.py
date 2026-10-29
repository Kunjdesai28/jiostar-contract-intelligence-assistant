"""One-time official held-out test evaluation with strict scientific safeguards.

Usage:
    python scripts/final_evaluation.py --config configs/final.yaml --split test --confirm-final
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_resolved_paths
from src.evaluation.metrics import compute_query_character_metrics
from src.retrieval.pipeline import ContractRetrievalPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("final_evaluation")

FROZEN_NEGATIVE_QUERY_CHECKSUM = "7eaf304f4a22a4567100fb283a53f1bfc7b458eaa2f7feb7d78ba66db3d539d3"
FROZEN_CONFIG_CHECKSUM = "6448998c99d0c236dd0644dd3abd6c62b679a0749187b3ce8f14ff210caa7718"
OUTPUT_FILE = Path("experiments/final_test_metrics.json")
FREEZE_MANIFEST_FILE = Path("docs/PHASE4_FREEZE.md")


def verify_safeguards(config_path: Path, split: str, confirm_final: bool):
    """Enforce strict pre-flight checks before evaluation can execute."""
    if not confirm_final:
        raise ValueError(
            "Evaluation aborted: --confirm-final flag must be explicitly passed to confirm "
            "that this is the single, irreversible official held-out test execution."
        )

    if split != "test":
        raise ValueError(f"Evaluation aborted: --split must be 'test', received '{split}'.")

    if OUTPUT_FILE.exists():
        raise RuntimeError(
            f"Evaluation aborted: Official test result artifact already exists at {OUTPUT_FILE}. "
            "To maintain scientific integrity, the held-out test set cannot be evaluated more than once."
        )

    if not FREEZE_MANIFEST_FILE.exists():
        raise RuntimeError(
            f"Evaluation aborted: Freeze manifest missing at {FREEZE_MANIFEST_FILE}. "
            "Stage C freeze must be completed before evaluating on test data."
        )

    # Verify configuration checksum
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cfg_bytes = config_path.read_bytes()
    cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()
    if cfg_hash != FROZEN_CONFIG_CHECKSUM:
        raise ValueError(
            f"Evaluation aborted: Configuration checksum mismatch!\n"
            f"Expected: {FROZEN_CONFIG_CHECKSUM}\n"
            f"Actual:   {cfg_hash}\n"
            "The configuration has been modified after freezing!"
        )

    # Verify frozen unanswerable query checksum
    paths = get_resolved_paths()
    unans_path = paths["unanswerable_queries_file"]
    if not unans_path.exists():
        raise FileNotFoundError(f"Unanswerable queries file not found: {unans_path}")

    unans_bytes = unans_path.read_bytes()
    unans_hash = hashlib.sha256(unans_bytes).hexdigest()
    if unans_hash != FROZEN_NEGATIVE_QUERY_CHECKSUM:
        raise ValueError(
            f"Evaluation aborted: Unanswerable queries checksum mismatch!\n"
            f"Expected: {FROZEN_NEGATIVE_QUERY_CHECKSUM}\n"
            f"Actual:   {unans_hash}"
        )

    logger.info("All pre-flight integrity safeguards and checksums VERIFIED.")


def main():
    parser = argparse.ArgumentParser(description="One-time official held-out test evaluation.")
    parser.add_argument("--config", type=str, default="configs/final.yaml", help="Path to frozen configuration.")
    parser.add_argument("--split", type=str, required=True, choices=["test"], help="Dataset split (must be 'test').")
    parser.add_argument("--confirm-final", action="store_true", help="Explicit confirmation for final test run.")
    args = parser.parse_args()

    config_path = Path(args.config)
    verify_safeguards(config_path, args.split, args.confirm_final)

    paths = get_resolved_paths()

    # Load test queries
    test_queries_path = paths["test_queries_output"]
    if not test_queries_path.exists():
        raise FileNotFoundError(f"Test queries dataset missing at {test_queries_path}")

    test_pos_data = json.loads(test_queries_path.read_text(encoding="utf-8"))["tests"]
    all_neg_data = json.loads(paths["unanswerable_queries_file"].read_text(encoding="utf-8"))
    test_neg_data = [q for q in all_neg_data if q.get("split") == "test"]

    n_pos = len(test_pos_data)
    n_neg = len(test_neg_data)

    if n_pos != 54:
        raise ValueError(f"Expected exactly 54 answerable test queries, found {n_pos}.")
    if n_neg != 13:
        raise ValueError(f"Expected exactly 13 unanswerable test queries, found {n_neg}.")

    logger.info(f"Loaded held-out test split: {n_pos} answerable queries, {n_neg} unanswerable queries (Total: {n_pos + n_neg}).")

    # Initialize retrieval pipeline
    pipeline = ContractRetrievalPipeline(config_path=str(config_path))
    pipeline.build_index()

    k_values = [1, 3, 5, 10]
    rec_accum = {k: 0.0 for k in k_values}
    prec_accum = {k: 0.0 for k in k_values}
    doc_hit_1_count = 0
    doc_hit_5_count = 0
    false_pos_abstentions = 0
    latencies = []

    pos_eval_records = []

    logger.info(f"Evaluating {n_pos} answerable test queries...")
    for idx, q_item in enumerate(test_pos_data):
        query = q_item["query"]
        snippets = q_item.get("snippets", [])
        gold_docs = {s["file_path"] for s in snippets if "file_path" in s}

        t0 = time.perf_counter()
        result = pipeline.search(query=query, top_k=max(k_values))
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        abstained = result.get("abstained", False)
        passages = result.get("passages", [])

        if abstained:
            false_pos_abstentions += 1
            query_metrics = {f"precision@{k}": 0.0 for k in k_values}
            query_metrics.update({f"recall@{k}": 0.0 for k in k_values})
            doc_hit_1 = False
            doc_hit_5 = False
        else:
            # Document hit evaluation
            ret_docs_1 = [p.get("document", "") for p in passages[:1]]
            ret_docs_5 = [p.get("document", "") for p in passages[:5]]
            doc_hit_1 = any(d in gold_docs for d in ret_docs_1)
            doc_hit_5 = any(d in gold_docs for d in ret_docs_5)

            if doc_hit_1:
                doc_hit_1_count += 1
            if doc_hit_5:
                doc_hit_5_count += 1

            query_metrics = {}
            for k in k_values:
                m = compute_query_character_metrics(snippets, passages[:k])
                prec_accum[k] += m["precision"]
                rec_accum[k] += m["recall"]
                query_metrics[f"precision@{k}"] = round(m["precision"], 4)
                query_metrics[f"recall@{k}"] = round(m["recall"], 4)

        pos_eval_records.append({
            "query_index": idx + 1,
            "query": query,
            "abstained": abstained,
            "abstention_rationale": result.get("abstention_rationale", ""),
            "doc_hit_at_1": doc_hit_1,
            "doc_hit_at_5": doc_hit_5,
            "latency_ms": round(lat_ms, 2),
            "metrics": query_metrics,
        })

    # Evaluate unanswerable negative queries
    logger.info(f"Evaluating {n_neg} unanswerable negative test queries...")
    correct_neg_abstentions = 0
    neg_eval_records = []

    for idx, q_item in enumerate(test_neg_data):
        query = q_item["query"]
        q_id = q_item.get("id", f"TEST_NEG_{idx+1:03d}")
        category = q_item.get("category", "unspecified")

        t0 = time.perf_counter()
        result = pipeline.search(query=query, top_k=5)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        abstained = result.get("abstained", False)
        if abstained:
            correct_neg_abstentions += 1

        neg_eval_records.append({
            "id": q_id,
            "category": category,
            "query": query,
            "abstained": abstained,
            "abstention_rationale": result.get("abstention_rationale", ""),
            "returned_count": len(result.get("passages", [])),
            "top1_score": result.get("passages", [{}])[0].get("score", 0.0) if result.get("passages") else 0.0,
            "latency_ms": round(lat_ms, 2),
        })

    # Aggregate metrics
    mean_precision = {f"precision@{k}": round(prec_accum[k] / n_pos, 4) for k in k_values}
    mean_recall = {f"recall@{k}": round(rec_accum[k] / n_pos, 4) for k in k_values}

    doc_hit_1_rate = round(doc_hit_1_count / n_pos, 4)
    doc_hit_5_rate = round(doc_hit_5_count / n_pos, 4)

    neg_abstention_rate = round(correct_neg_abstentions / n_neg, 4)
    incorrect_neg_responses = n_neg - correct_neg_abstentions
    false_pos_rate = round(false_pos_abstentions / n_pos, 4)
    answerable_coverage = round((n_pos - false_pos_abstentions) / n_pos, 4)

    mean_lat = round(float(np.mean(latencies)), 2)
    median_lat = round(float(np.median(latencies)), 2)
    p95_lat = round(float(np.percentile(latencies, 95)), 2)

    timestamp_utc = datetime.now(timezone.utc).isoformat()

    final_results = {
        "evaluation_name": "official_held_out_test_evaluation",
        "timestamp_utc": timestamp_utc,
        "split": "test",
        "configuration_path": str(config_path),
        "configuration_sha256": FROZEN_CONFIG_CHECKSUM,
        "negative_query_checksum": FROZEN_NEGATIVE_QUERY_CHECKSUM,
        "sample_counts": {
            "answerable_test_queries": n_pos,
            "unanswerable_test_queries": n_neg,
            "total_test_queries": n_pos + n_neg,
        },
        "character_level_retrieval_metrics": {
            **mean_precision,
            **mean_recall,
        },
        "document_hit_metrics": {
            "document_hit_at_1": doc_hit_1_rate,
            "document_hit_at_5": doc_hit_5_rate,
            "document_hit_at_1_count": f"{doc_hit_1_count}/{n_pos}",
            "document_hit_at_5_count": f"{doc_hit_5_count}/{n_pos}",
        },
        "abstention_metrics": {
            "correct_negative_abstentions": f"{correct_neg_abstentions}/{n_neg}",
            "correct_negative_abstention_rate": neg_abstention_rate,
            "incorrect_negative_responses": f"{incorrect_neg_responses}/{n_neg}",
            "incorrect_negative_response_rate": round(incorrect_neg_responses / n_neg, 4),
            "false_positive_abstentions": f"{false_pos_abstentions}/{n_pos}",
            "false_positive_abstention_rate": false_pos_rate,
            "answerable_query_coverage": answerable_coverage,
        },
        "latency_metrics_ms": {
            "mean": mean_lat,
            "median": median_lat,
            "p95": p95_lat,
        },
        "negative_query_breakdown": neg_eval_records,
    }

    # Write output atomically
    temp_output = OUTPUT_FILE.with_suffix(".tmp")
    temp_output.write_text(json.dumps(final_results, indent=2), encoding="utf-8")
    os.replace(temp_output, OUTPUT_FILE)

    logger.info(f"Official test evaluation saved to {OUTPUT_FILE}.")
    print("\n" + "=" * 60)
    print("OFFICIAL HELD-OUT TEST EVALUATION RESULTS")
    print("=" * 60)
    print(json.dumps(final_results["character_level_retrieval_metrics"], indent=2))
    print(f"Document Hit@1: {doc_hit_1_rate:.1%} ({doc_hit_1_count}/{n_pos})")
    print(f"Document Hit@5: {doc_hit_5_rate:.1%} ({doc_hit_5_count}/{n_pos})")
    print(f"Correct Negative Abstentions: {correct_neg_abstentions}/{n_neg} ({neg_abstention_rate:.1%})")
    print(f"False Positive Abstentions:   {false_pos_abstentions}/{n_pos} ({false_pos_rate:.1%})")
    print(f"Answerable Coverage:          {answerable_coverage:.1%}")
    print(f"Mean Latency: {mean_lat} ms | Median: {median_lat} ms | P95: {p95_lat} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
