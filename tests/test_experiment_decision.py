"""Tests for Decision Engine enforcing frozen objectives and guardrails."""

import pytest

from src.improvement.decision import evaluate_experiment_decision
from src.improvement.objectives import get_default_objective_spec


def test_decision_accepts_when_recall_improves_and_guardrails_pass():
    accepted_metrics = {
        "retrieval_metrics_character_level": {"recall@5": 0.1169, "precision@5": 0.0141},
        "abstention_metrics": {"correct_abstention_rate": 0.0, "false_abstention_rate": 0.0},
        "elapsed_seconds": 2.0,
    }
    candidate_metrics = {
        "retrieval_metrics_character_level": {"recall@5": 0.1850, "precision@5": 0.0145},
        "abstention_metrics": {"correct_abstention_rate": 0.0, "false_abstention_rate": 0.0},
        "elapsed_seconds": 2.1,
    }
    baseline_metrics = {"elapsed_seconds": 2.0}

    decision, reason, guardrails = evaluate_experiment_decision(
        accepted_metrics=accepted_metrics,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
    )
    assert decision == "accepted"
    assert "improved from 0.1169 to 0.1850" in reason
    assert guardrails["primary_objective_improved"]["passed"] is True
    assert guardrails["precision_guardrail"]["passed"] is True


def test_decision_rejects_when_recall_fails_to_improve():
    accepted_metrics = {
        "retrieval_metrics_character_level": {"recall@5": 0.1500, "precision@5": 0.0140},
        "abstention_metrics": {"correct_abstention_rate": 0.0, "false_abstention_rate": 0.0},
        "elapsed_seconds": 2.0,
    }
    candidate_metrics = {
        "retrieval_metrics_character_level": {"recall@5": 0.1400, "precision@5": 0.0150},
        "abstention_metrics": {"correct_abstention_rate": 0.0, "false_abstention_rate": 0.0},
        "elapsed_seconds": 2.0,
    }
    baseline_metrics = {"elapsed_seconds": 2.0}

    decision, reason, guardrails = evaluate_experiment_decision(
        accepted_metrics=accepted_metrics,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
    )
    assert decision == "rejected"
    assert "Recall@5 did not improve" in reason
    assert guardrails["primary_objective_improved"]["passed"] is False


def test_decision_rejects_when_precision_guardrail_violated():
    accepted_metrics = {
        "retrieval_metrics_character_level": {"recall@5": 0.1169, "precision@5": 0.0200},
        "abstention_metrics": {"correct_abstention_rate": 0.0, "false_abstention_rate": 0.0},
        "elapsed_seconds": 2.0,
    }
    # Precision drops by 50% (from 0.0200 to 0.0100), exceeding 20% limit
    candidate_metrics = {
        "retrieval_metrics_character_level": {"recall@5": 0.1500, "precision@5": 0.0100},
        "abstention_metrics": {"correct_abstention_rate": 0.0, "false_abstention_rate": 0.0},
        "elapsed_seconds": 2.0,
    }
    baseline_metrics = {"elapsed_seconds": 2.0}

    decision, reason, guardrails = evaluate_experiment_decision(
        accepted_metrics=accepted_metrics,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
    )
    assert decision == "rejected"
    assert "Precision@5 guardrail violated" in reason
    assert guardrails["precision_guardrail"]["passed"] is False


def test_decision_rejects_when_false_abstention_guardrail_violated():
    accepted_metrics = {
        "retrieval_metrics_character_level": {"recall@5": 0.1169, "precision@5": 0.0141},
        "abstention_metrics": {"correct_abstention_rate": 0.0, "false_abstention_rate": 0.0},
        "elapsed_seconds": 2.0,
    }
    # False abstention jumps to 15% (allowed max increase is 5pp)
    candidate_metrics = {
        "retrieval_metrics_character_level": {"recall@5": 0.1500, "precision@5": 0.0141},
        "abstention_metrics": {"correct_abstention_rate": 0.5, "false_abstention_rate": 0.15},
        "elapsed_seconds": 2.0,
    }
    baseline_metrics = {"elapsed_seconds": 2.0}

    decision, reason, guardrails = evaluate_experiment_decision(
        accepted_metrics=accepted_metrics,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
    )
    assert decision == "rejected"
    assert "False abstention rate guardrail violated" in reason
    assert guardrails["false_abstention_guardrail"]["passed"] is False
