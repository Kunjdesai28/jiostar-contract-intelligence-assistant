"""Tests for Self-Improvement Loop components and logger."""

import json
from pathlib import Path
import pytest

from src.improvement.experiment_logger import ExperimentLogger
from src.improvement.objectives import get_default_objective_spec
from src.improvement.proposer import ExperimentProposer


def test_proposer_generates_distinct_candidates_with_full_metadata():
    proposer = ExperimentProposer()
    base_config = {
        "pipeline": {"name": "bm25_baseline"},
        "query_processing": {"mode": "original"},
        "tokenization": {"lowercase": True, "enhancement_mode": "none"},
        "chunking": {"strategy": "fixed_character", "chunk_size": 800, "overlap": 150},
        "bm25": {"k1": 1.5, "b": 0.75},
        "abstention": {"threshold": 0.0, "strategy": "top_score_threshold"},
    }

    proposals = []
    parent = base_config
    for it in range(1, 6):
        prop = proposer.propose(iteration=it, parent_config=parent, diagnostics={"dominant_failure": "wrong_section"})
        proposals.append(prop)
        # Verify required keys
        assert prop["iteration"] == it
        assert prop["proposer"] == "rule_based"
        assert "candidate_config_id" in prop
        assert "proposed_change" in prop
        assert "scientific_rationale" in prop
        assert "expected_direction" in prop
        assert "candidate_config" in prop

    # Verify all candidate config IDs are distinct
    cand_ids = [p["candidate_config_id"] for p in proposals]
    assert len(set(cand_ids)) == 5


def test_experiment_logger_appends_and_writes_summary(tmp_path):
    log_file = tmp_path / "test_log.jsonl"
    summary_file = tmp_path / "test_summary.json"

    logger = ExperimentLogger(log_path=log_file, summary_path=summary_file)

    sample_record = {
        "iteration": 1,
        "candidate_config_id": "candidate_001",
        "decision": "accepted",
        "proposed_change": {"parameter": "query_processing.mode", "new_value": "focused"},
    }
    logger.log_iteration(sample_record)
    assert log_file.exists()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    logged_obj = json.loads(lines[0])
    assert logged_obj["iteration"] == 1
    assert "timestamp" in logged_obj

    # Test summary write
    base_m = {
        "configuration": {},
        "retrieval_metrics_character_level": {"recall@5": 0.1169, "precision@5": 0.0141},
        "abstention_metrics": {"correct_abstention_rate": 0.0, "false_abstention_rate": 0.0},
    }
    final_m = {
        "configuration": {},
        "retrieval_metrics_character_level": {"recall@5": 0.1850, "precision@5": 0.0145},
        "abstention_metrics": {"correct_abstention_rate": 0.0, "false_abstention_rate": 0.0},
    }
    summary = logger.write_summary(
        baseline_metrics=base_m,
        final_metrics=final_m,
        iteration_records=[sample_record],
        objective_spec=get_default_objective_spec(),
        total_duration_seconds=5.2,
    )
    assert summary_file.exists()
    assert summary["total_iterations_executed"] == 1
    assert summary["iterations_accepted"] == 1
