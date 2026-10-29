"""Deterministic Acceptance and Rejection Engine.

Evaluates candidate experiment results against the current accepted configuration
using predeclared primary objective and frozen guardrails.
"""

from typing import Any, Dict, Tuple

from src.improvement.objectives import get_default_objective_spec


def evaluate_experiment_decision(
    accepted_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    spec: Dict[str, Any] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Evaluate candidate metrics against accepted baseline.

    Args:
        accepted_metrics: Metrics of the currently accepted configuration.
        candidate_metrics: Metrics of the candidate configuration.
        baseline_metrics: Metrics of original baseline (used for baseline latency checks).
        spec: Objective and guardrail specifications (defaults to frozen spec).

    Returns:
        Tuple of (decision: 'accepted'|'rejected', decision_reason: str, guardrail_results: Dict).
    """
    if spec is None:
        spec = get_default_objective_spec()

    guardrails = spec["guardrails"]
    min_delta = spec["primary_objective"]["min_delta"]

    # Extract primary objective
    acc_rec5 = float(accepted_metrics.get("retrieval_metrics_character_level", {}).get("recall@5", 0.0))
    cand_rec5 = float(candidate_metrics.get("retrieval_metrics_character_level", {}).get("recall@5", 0.0))

    # Extract secondary guardrail metrics
    acc_prec5 = float(accepted_metrics.get("retrieval_metrics_character_level", {}).get("precision@5", 0.0))
    cand_prec5 = float(candidate_metrics.get("retrieval_metrics_character_level", {}).get("precision@5", 0.0))

    acc_abs = accepted_metrics.get("abstention_metrics", {})
    acc_neg_abst = float(acc_abs.get("correct_abstention_rate") or acc_abs.get("unanswerable_queries", {}).get("correct_abstention_rate", 0.0))
    cand_abs = candidate_metrics.get("abstention_metrics", {})
    cand_neg_abst = float(cand_abs.get("correct_abstention_rate") or cand_abs.get("unanswerable_queries", {}).get("correct_abstention_rate", 0.0))

    acc_false_abst = float(acc_abs.get("false_abstention_rate") or acc_abs.get("answerable_queries", {}).get("false_abstention_rate", 0.0))
    cand_false_abst = float(cand_abs.get("false_abstention_rate") or cand_abs.get("answerable_queries", {}).get("false_abstention_rate", 0.0))

    base_latency = float(baseline_metrics.get("elapsed_seconds", 1.0))
    cand_latency = float(candidate_metrics.get("elapsed_seconds", 1.0))

    guardrail_results: Dict[str, Any] = {}

    # Check 1: Primary Objective Improvement
    delta_rec5 = cand_rec5 - acc_rec5
    primary_improved = delta_rec5 > min_delta
    guardrail_results["primary_objective_improved"] = {
        "passed": primary_improved,
        "current_accepted": acc_rec5,
        "candidate": cand_rec5,
        "delta": round(delta_rec5, 6),
        "required_delta": min_delta,
    }

    # Check 2: Precision@5 Guardrail (max 20% relative drop)
    # Allowed floor: acc_prec5 * (1.0 - 0.20)
    prec_floor = acc_prec5 * (1.0 - guardrails["max_precision_at_5_relative_drop"])
    prec_passed = cand_prec5 >= (prec_floor - 1e-6)
    guardrail_results["precision_guardrail"] = {
        "passed": prec_passed,
        "current_accepted": acc_prec5,
        "candidate": cand_prec5,
        "allowed_floor": round(prec_floor, 6),
    }

    # Check 3: Correct Negative Abstention Guardrail (max 10pp drop)
    neg_abst_floor = acc_neg_abst - guardrails["max_correct_abstention_drop_pp"]
    neg_abst_passed = cand_neg_abst >= (neg_abst_floor - 1e-6)
    guardrail_results["negative_abstention_guardrail"] = {
        "passed": neg_abst_passed,
        "current_accepted": acc_neg_abst,
        "candidate": cand_neg_abst,
        "allowed_floor": round(neg_abst_floor, 4),
    }

    # Check 4: False Abstention Guardrail (max 5pp increase)
    false_abst_ceiling = acc_false_abst + guardrails["max_false_abstention_increase_pp"]
    false_abst_passed = cand_false_abst <= (false_abst_ceiling + 1e-6)
    guardrail_results["false_abstention_guardrail"] = {
        "passed": false_abst_passed,
        "current_accepted": acc_false_abst,
        "candidate": cand_false_abst,
        "allowed_ceiling": round(false_abst_ceiling, 4),
    }

    # Check 5: Latency Guardrail (max 3x baseline)
    max_latency = base_latency * guardrails["max_latency_multiplier"]
    latency_passed = cand_latency <= (max_latency + 1.0)
    guardrail_results["latency_guardrail"] = {
        "passed": latency_passed,
        "baseline_latency": base_latency,
        "candidate_latency": cand_latency,
        "allowed_max": round(max_latency, 2),
    }

    # Decision synthesis
    all_guardrails_passed = (
        prec_passed and neg_abst_passed and false_abst_passed and latency_passed
    )

    if not primary_improved:
        decision = "rejected"
        decision_reason = (
            f"Rejected: Primary objective Recall@5 did not improve "
            f"(candidate {cand_rec5:.4f} vs accepted {acc_rec5:.4f}, delta: {delta_rec5:+.6f})."
        )
    elif not prec_passed:
        decision = "rejected"
        decision_reason = (
            f"Rejected: Precision@5 guardrail violated "
            f"(candidate {cand_prec5:.4f} dropped below allowed floor {prec_floor:.4f})."
        )
    elif not neg_abst_passed:
        decision = "rejected"
        decision_reason = (
            f"Rejected: Correct negative abstention guardrail violated "
            f"(candidate {cand_neg_abst:.4f} dropped below allowed floor {neg_abst_floor:.4f})."
        )
    elif not false_abst_passed:
        decision = "rejected"
        decision_reason = (
            f"Rejected: False abstention rate guardrail violated "
            f"(candidate {cand_false_abst:.4f} exceeded allowed ceiling {false_abst_ceiling:.4f})."
        )
    elif not latency_passed:
        decision = "rejected"
        decision_reason = (
            f"Rejected: Latency guardrail violated "
            f"(candidate {cand_latency:.2f}s exceeded 3x baseline {max_latency:.2f}s)."
        )
    else:
        decision = "accepted"
        pct_gain = ((cand_rec5 - acc_rec5) / max(1e-6, acc_rec5)) * 100
        decision_reason = (
            f"Accepted: Recall@5 improved from {acc_rec5:.4f} to {cand_rec5:.4f} (+{pct_gain:.2f}% relative) "
            f"with all guardrails satisfied."
        )

    return decision, decision_reason, guardrail_results
