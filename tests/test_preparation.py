"""End-to-end integration and reproducibility tests for the data preparation pipeline."""

import json
from pathlib import Path
import pytest

from src.data.prepare import run_preparation_pipeline


def test_preparation_pipeline_idempotency_and_checksum_stability(tmp_path):
    """Verify that running the preparation pipeline multiple times yields bit-for-bit identical outputs."""
    raw_benchmark = Path("data/raw/benchmarks/cuad.json")
    raw_corpus = Path("data/raw/corpus")

    if not raw_benchmark.exists() or not raw_corpus.exists():
        pytest.skip("Raw dataset not available. Run download first.")

    # Configure isolated temporary output directories
    proc_dir_1 = tmp_path / "proc_run_1"
    proc_dir_2 = tmp_path / "proc_run_2"

    config_1 = {
        "benchmark": {"mini_size": 194},
        "paths": {
            "raw_dir": "data/raw",
            "processed_dir": str(proc_dir_1),
            "benchmark_file": "benchmarks/cuad.json",
            "corpus_dir": "corpus/cuad",
            "mini_benchmark_output": str(proc_dir_1 / "cuad_mini.json"),
            "train_queries_output": str(proc_dir_1 / "train_queries.json"),
            "test_queries_output": str(proc_dir_1 / "test_queries.json"),
            "contract_manifest_output": str(proc_dir_1 / "contract_manifest.json"),
            "validation_summary_output": str(proc_dir_1 / "span_validation_summary.json"),
            "preparation_summary_output": str(proc_dir_1 / "preparation_summary.json"),
            "unanswerable_queries_file": str(proc_dir_1 / "unanswerable_queries.json"),
        },
    }

    config_2 = {
        "benchmark": {"mini_size": 194},
        "paths": {
            "raw_dir": "data/raw",
            "processed_dir": str(proc_dir_2),
            "benchmark_file": "benchmarks/cuad.json",
            "corpus_dir": "corpus/cuad",
            "mini_benchmark_output": str(proc_dir_2 / "cuad_mini.json"),
            "train_queries_output": str(proc_dir_2 / "train_queries.json"),
            "test_queries_output": str(proc_dir_2 / "test_queries.json"),
            "contract_manifest_output": str(proc_dir_2 / "contract_manifest.json"),
            "validation_summary_output": str(proc_dir_2 / "span_validation_summary.json"),
            "preparation_summary_output": str(proc_dir_2 / "preparation_summary.json"),
            "unanswerable_queries_file": str(proc_dir_2 / "unanswerable_queries.json"),
        },
    }

    summary_1 = run_preparation_pipeline(config=config_1)
    summary_2 = run_preparation_pipeline(config=config_2)

    # 1. Mini benchmark checksum identity
    assert summary_1["mini_benchmark"]["sha256"] == summary_2["mini_benchmark"]["sha256"]

    # 2. Train/Test counts identity
    assert summary_1["train_test_split"]["train_count"] == summary_2["train_test_split"]["train_count"]
    assert summary_1["train_test_split"]["test_count"] == summary_2["train_test_split"]["test_count"]
    assert summary_1["train_test_split"]["train_sha256"] == summary_2["train_test_split"]["train_sha256"]
    assert summary_1["train_test_split"]["test_sha256"] == summary_2["train_test_split"]["test_sha256"]

    # 3. Contract manifest checksum identity
    assert summary_1["indexed_contracts"]["unique_count"] == summary_2["indexed_contracts"]["unique_count"]
    assert summary_1["indexed_contracts"]["unique_count"] == 29

    # 4. Span validation identity (272/272)
    assert summary_1["span_validation"]["valid_snippets"] == 272
    assert summary_1["span_validation"]["invalid_snippets"] == 0
    assert summary_2["span_validation"]["valid_snippets"] == 272
    assert summary_2["span_validation"]["invalid_snippets"] == 0

    # 5. Negative query checksum identity
    assert summary_1["unanswerable_queries"]["sha256"] == summary_2["unanswerable_queries"]["sha256"]
