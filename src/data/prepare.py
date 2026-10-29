"""Unified data preparation pipeline for JioStar Contract Intelligence Assistant."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import get_project_root, get_resolved_paths, load_yaml_config
from src.data.loader import load_indexed_contracts
from src.data.mini_selector import select_and_validate_mini_benchmark
from src.data.negative_queries import freeze_negative_queries, validate_negative_queries
from src.data.span_validator import validate_ground_truth_spans
from src.data.splitter import split_benchmark

logger = logging.getLogger(__name__)


def run_preparation_pipeline(
    config: Optional[Dict[str, Any]] = None,
    enforce_negative_queries: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end reproducible preparation pipeline.

    Steps:
      1. Load configuration and resolve paths.
      2. Verify raw benchmark and corpus existence.
      3. Select 194-query mini benchmark deterministically.
      4. Validate all ground-truth character spans against exact document text.
      5. Discover referenced contracts and generate contract manifest.
      6. Split benchmark queries into train and test sets using SHA-256 thresholding.
      7. Validate/freeze unanswerable negative queries dataset.
      8. Produce comprehensive preparation summary with checksums and provenance.

    Returns:
        Preparation summary dictionary.
    """
    if config is None:
        config = load_yaml_config()

    resolved_paths = get_resolved_paths(config)
    raw_dir: Path = resolved_paths["raw_dir"]
    processed_dir: Path = resolved_paths["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rel: str = config["paths"]["benchmark_file"]
    benchmark_file = raw_dir / benchmark_rel
    corpus_root = raw_dir / "corpus"

    if not benchmark_file.exists():
        raise FileNotFoundError(f"Raw benchmark file missing at {benchmark_file}. Run scripts/download_dataset.py first.")
    if not corpus_root.exists():
        raise FileNotFoundError(f"Raw corpus directory missing at {corpus_root}. Run scripts/download_dataset.py first.")

    mini_size = int(config.get("benchmark", {}).get("mini_size", 194))
    if mini_size != 194:
        raise ValueError(f"Official benchmark mini_size must be 194, found {mini_size} in config.")

    logger.info(f"Starting preparation pipeline: selecting {mini_size} queries from {benchmark_file}...")

    # Step 1: Select mini benchmark
    mini_out = resolved_paths["mini_benchmark_output"]
    mini_tests, mini_sha256 = select_and_validate_mini_benchmark(
        raw_benchmark_path=benchmark_file,
        output_path=mini_out,
        expected_size=mini_size,
    )

    # Step 2: Validate ground truth spans
    val_summary_out = resolved_paths["validation_summary_output"]
    is_valid_spans, span_summary = validate_ground_truth_spans(
        tests=mini_tests,
        corpus_root=corpus_root,
        summary_output_path=val_summary_out,
    )
    if not is_valid_spans:
        raise RuntimeError(
            f"Critical validation failure: {span_summary['invalid_snippets']} invalid spans, "
            f"{len(span_summary['missing_documents'])} missing documents."
        )

    # Step 3: Discover contracts and generate manifest
    manifest_out = resolved_paths["contract_manifest_output"]
    contracts, manifest_summary = load_indexed_contracts(
        tests=mini_tests,
        corpus_root=corpus_root,
        manifest_output_path=manifest_out,
    )

    # Step 4: Deterministic train/test split
    train_out = resolved_paths["train_queries_output"]
    test_out = resolved_paths["test_queries_output"]
    train_records, test_records, split_stats = split_benchmark(
        tests=mini_tests,
        train_output_path=train_out,
        test_output_path=test_out,
    )

    # Step 5: Unanswerable queries dataset
    neg_path = resolved_paths["unanswerable_queries_file"]
    neg_records, neg_sha256, neg_report = freeze_negative_queries(
        output_path=neg_path,
        overwrite_existing=False,
    )

    # Step 6: Assemble machine-readable preparation summary
    provenance_file = raw_dir / "provenance.json"
    provenance_data = {}
    if provenance_file.exists():
        provenance_data = json.loads(provenance_file.read_text(encoding="utf-8"))

    summary = {
        "status": "SUCCESS",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_provenance": provenance_data,
        "mini_benchmark": {
            "requested_size": mini_size,
            "actual_size": len(mini_tests),
            "output_file": str(mini_out),
            "sha256": mini_sha256,
        },
        "train_test_split": split_stats,
        "indexed_contracts": {
            "unique_count": len(contracts),
            "manifest_file": str(manifest_out),
            "manifest_sha256": manifest_summary.get("manifest_sha256"),
        },
        "span_validation": {
            "is_valid": is_valid_spans,
            "total_snippets": span_summary["total_annotated_snippets"],
            "valid_snippets": span_summary["valid_snippets"],
            "invalid_snippets": span_summary["invalid_snippets"],
            "summary_file": str(val_summary_out),
        },
        "unanswerable_queries": {
            "total_count": len(neg_records),
            "train_count": neg_report["train_count"],
            "test_count": neg_report["test_count"],
            "categories": neg_report["categories"],
            "file": str(neg_path),
            "sha256": neg_sha256,
        },
    }

    prep_summary_out = resolved_paths["preparation_summary_output"]
    serialized_summary = json.dumps(summary, indent=2, ensure_ascii=False)
    prep_summary_out.write_text(serialized_summary, encoding="utf-8")
    summary["preparation_summary_sha256"] = hashlib.sha256(serialized_summary.encode("utf-8")).hexdigest()
    logger.info(f"Preparation pipeline completed successfully. Summary saved to {prep_summary_out}")

    return summary
