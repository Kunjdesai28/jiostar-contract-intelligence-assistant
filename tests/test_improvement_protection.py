"""Regression tests for frozen artifact protection and safe replay behavior."""

import subprocess
import sys
from pathlib import Path
import pytest

from src.config import get_project_root
from src.improvement.runner import (
    ImprovementLoopRunner,
    is_protected_artifact,
    PROTECTED_RELATIVE_PATHS,
)


def test_default_replay_destinations():
    """Verify ImprovementLoopRunner defaults to non-destructive replay paths."""
    runner = ImprovementLoopRunner()
    assert runner.log_path.name == "replay_log.jsonl"
    assert runner.summary_path.name == "replay_summary.json"
    assert runner.final_config_path.name == "replay_final.yaml"


def test_protected_artifacts_detection_relative_and_absolute():
    """Verify is_protected_artifact catches relative, absolute, and normalized paths."""
    root = get_project_root()

    # Relative paths
    assert is_protected_artifact("configs/final.yaml")
    assert is_protected_artifact("experiments/improvement_log.jsonl")
    assert is_protected_artifact("experiments/improvement_summary.json")
    assert is_protected_artifact("configs/baseline.yaml")
    assert is_protected_artifact("experiments/final_test_metrics.json")
    assert is_protected_artifact("data/unanswerable_queries.json")

    # Absolute paths
    assert is_protected_artifact((root / "configs" / "final.yaml").resolve())
    assert is_protected_artifact((root / "experiments" / "improvement_log.jsonl").resolve())

    # Traversal and normalized relative paths
    assert is_protected_artifact("configs/../configs/final.yaml")
    assert is_protected_artifact("./experiments/improvement_log.jsonl")

    # Safe replay paths must NOT be detected as protected
    assert not is_protected_artifact("experiments/replay_log.jsonl")
    assert not is_protected_artifact("experiments/replay_summary.json")
    assert not is_protected_artifact("configs/replay_final.yaml")
    assert not is_protected_artifact("experiments/reproduced_final_train.json")


def test_runner_blocks_protected_artifacts():
    """Verify runner rejects protected artifact destinations with PermissionError."""
    with pytest.raises(PermissionError) as exc_info:
        ImprovementLoopRunner(final_config_path="configs/final.yaml")
    assert "Protected artifact violation" in str(exc_info.value)
    assert "final.yaml" in str(exc_info.value)

    with pytest.raises(PermissionError) as exc_info:
        ImprovementLoopRunner(log_path="experiments/improvement_log.jsonl")
    assert "Protected artifact violation" in str(exc_info.value)
    assert "improvement_log.jsonl" in str(exc_info.value)

    with pytest.raises(PermissionError) as exc_info:
        ImprovementLoopRunner(summary_path="experiments/improvement_summary.json")
    assert "Protected artifact violation" in str(exc_info.value)
    assert "improvement_summary.json" in str(exc_info.value)


def test_runner_accepts_custom_safe_paths(tmp_path):
    """Verify runner successfully initializes when given clean temporary paths."""
    custom_log = tmp_path / "custom_log.jsonl"
    custom_sum = tmp_path / "custom_summary.json"
    custom_cfg = tmp_path / "custom_final.yaml"

    runner = ImprovementLoopRunner(
        log_path=custom_log,
        summary_path=custom_sum,
        final_config_path=custom_cfg,
    )
    assert runner.log_path == custom_log
    assert runner.summary_path == custom_sum
    assert runner.final_config_path == custom_cfg


def test_existing_replay_log_cleared_safely(tmp_path):
    """Verify that existing replay logs are truncated cleanly on new run initialization."""
    replay_log = tmp_path / "replay_log.jsonl"
    replay_log.write_text("{\"stale_iteration\": 99}\n", encoding="utf-8")
    assert replay_log.stat().st_size > 0

    _ = ImprovementLoopRunner(
        log_path=replay_log,
        summary_path=tmp_path / "sum.json",
        final_config_path=tmp_path / "cfg.yaml",
    )
    # File should have been reset to 0 bytes
    assert replay_log.stat().st_size == 0


def test_cli_blocks_protected_destination():
    """Verify CLI returns exit code 1 when attempting to overwrite protected configs/final.yaml."""
    cmd = [
        sys.executable,
        "scripts/run_improvement_loop.py",
        "--final-config-path",
        "configs/final.yaml",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "SECURITY ERROR" in res.stderr or "SECURITY ERROR" in res.stdout


def test_cli_safe_defaults_help():
    """Verify CLI help text documents replay default destinations."""
    cmd = [sys.executable, "scripts/run_improvement_loop.py", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "replay_log.jsonl" in res.stdout
    assert "replay_final.yaml" in res.stdout
