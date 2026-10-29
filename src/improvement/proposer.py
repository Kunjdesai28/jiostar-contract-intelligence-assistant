"""Deterministic Configuration Proposal Engine for Retrieval Self-Improvement.

Diagnoses retrieval failure distributions and proposes controlled, principled
parameter modifications across query processing, lexical enhancements,
BM25 parameters, chunking strategies, and abstention calibration.
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ExperimentProposer:
    """Proposes principled configuration modifications for controlled retrieval experiments."""

    def __init__(self, search_space: Optional[Dict[str, Any]] = None):
        """Initialize proposer.

        Args:
            search_space: Optional custom search space dictionary.
        """
        self.search_space = search_space or {}
        self.proposed_configs_history: List[Dict[str, Any]] = []
        self.visited_hashes: Set[str] = set()

    def _hash_config(self, cfg: Dict[str, Any]) -> str:
        """Create a deterministic signature of configuration parameters."""
        qp = cfg.get("query_processing", {}).get("mode", "original")
        tok = cfg.get("tokenization", {}).get("enhancement_mode", "none")
        k1 = cfg.get("bm25", {}).get("k1", 1.5)
        b = cfg.get("bm25", {}).get("b", 0.75)
        c_size = cfg.get("chunking", {}).get("chunk_size", 800)
        c_overlap = cfg.get("chunking", {}).get("overlap", 150)
        c_strat = cfg.get("chunking", {}).get("strategy", "fixed_character")
        abs_thresh = cfg.get("abstention", {}).get("threshold", 0.0)
        abs_strat = cfg.get("abstention", {}).get("strategy", "top_score_threshold")
        return f"{qp}|{tok}|{k1}|{b}|{c_size}|{c_overlap}|{c_strat}|{abs_thresh}|{abs_strat}"

    def propose(
        self,
        iteration: int,
        parent_config: Dict[str, Any],
        diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Propose the next controlled configuration candidate.

        Args:
            iteration: Current loop iteration (1-indexed).
            parent_config: Accepted reference configuration.
            diagnostics: Error diagnostic report of current accepted configuration.

        Returns:
            Dict containing proposal metadata, rationale, and new candidate_config.
        """
        dominant_failure = diagnostics.get("dominant_failure", "wrong_section")
        candidate_config = copy.deepcopy(parent_config)
        candidate_config_id = f"candidate_{iteration:03d}"
        candidate_config["pipeline"]["name"] = candidate_config_id

        # Predefined sequence of controlled hypothesis tests
        if iteration == 1:
            # Experiment Family A: Two-Stage Query Processing
            # Diagnosis: Preamble terms dominate BM25 ranking over internal clauses
            candidate_config["query_processing"] = {"mode": "two_stage"}
            param_change = {
                "parameter": "query_processing.mode",
                "old_value": parent_config.get("query_processing", {}).get("mode", "original"),
                "new_value": "two_stage",
            }
            rationale = (
                "Diagnosed dominant wrong_section error caused by preamble dominance. "
                "Use two-stage retrieval: document context filters candidate contracts, "
                "then substantive legal question ranks clauses within target contracts."
            )
            expected_direction = "Increase Recall@5 by focusing within-document BM25 ranking on substantive clause."

        elif iteration == 2:
            # Experiment Family B: Lexical Enhancement
            # Diagnosis: Morphological variation (e.g. sublicensing vs sublicense, indemnification vs indemnify)
            candidate_config["tokenization"]["enhancement_mode"] = "legal_stem"
            param_change = {
                "parameter": "tokenization.enhancement_mode",
                "old_value": parent_config.get("tokenization", {}).get("enhancement_mode", "none"),
                "new_value": "legal_stem",
            }
            rationale = (
                "Diagnosed vocabulary mismatch for legal inflectional suffixes (-ing, -ation, -ed, -ment). "
                "Apply conservative legal stemming while strictly preserving legal modals."
            )
            expected_direction = "Increase Recall@5 by matching inflected legal terms."

        elif iteration == 3:
            # Experiment Family C: Chunk Overlap & Span Coverage
            # Diagnosis: Partial span errors caused by boundary fragmentation
            candidate_config["chunking"]["overlap"] = 250
            param_change = {
                "parameter": "chunking.overlap",
                "old_value": parent_config.get("chunking", {}).get("overlap", 150),
                "new_value": 250,
            }
            rationale = (
                "Diagnosed partial_span failures where target clauses cross chunk boundaries. "
                "Increase sliding-window overlap from 150 to 250 chars to improve full clause retention."
            )
            expected_direction = "Increase Recall@5 and reduce boundary fragmentation."

        elif iteration == 4:
            # Experiment Family D: BM25 Parameter Tuning
            # Diagnosis: Test whether lower k1 (1.2) and lower b (0.65) improve ranking on candidate 003
            candidate_config["bm25"]["k1"] = 1.2
            candidate_config["bm25"]["b"] = 0.65
            param_change = {
                "parameter": "bm25.k1_and_b",
                "old_value": f"k1={parent_config.get('bm25', {}).get('k1', 1.5)}, b={parent_config.get('bm25', {}).get('b', 0.75)}",
                "new_value": "k1=1.2, b=0.65",
            }
            rationale = (
                "Fine-tune BM25 parameters on top of candidate_003: lower k1 (1.2) to reach faster term frequency saturation, "
                "and lower b (0.65) to reduce length penalization on dense contractual clauses."
            )
            expected_direction = "Test whether modified term saturation improves Recall@5."

        elif iteration == 5:
            # Experiment Family E: Clause-Aware Chunking
            # Diagnosis: Test structural chunking against fixed sliding window
            candidate_config["chunking"]["strategy"] = "clause_aware"
            candidate_config["chunking"]["chunk_size"] = 1000
            param_change = {
                "parameter": "chunking.strategy",
                "old_value": parent_config.get("chunking", {}).get("strategy", "fixed_character"),
                "new_value": "clause_aware (size=1000)",
            }
            rationale = (
                "Test structure-aware chunking splitting along numbered sections and article headings "
                "to align passage boundaries with natural contractual clause units."
            )
            expected_direction = "Improve clause boundary alignment and character recall."

        else:
            # Fallback iteration: Abstention calibration
            candidate_config["abstention"]["threshold"] = 4.0
            param_change = {
                "parameter": "abstention.threshold",
                "old_value": parent_config.get("abstention", {}).get("threshold", 0.0),
                "new_value": 4.0,
            }
            rationale = "Calibrate abstention threshold on focused query score distribution."
            expected_direction = "Increase negative abstention."

        proposal = {
            "iteration": iteration,
            "proposer": "rule_based",
            "candidate_config_id": candidate_config_id,
            "parent_config_id": parent_config.get("pipeline", {}).get("name", "bm25_baseline"),
            "observed_failure_pattern": dominant_failure,
            "proposed_change": param_change,
            "scientific_rationale": rationale,
            "expected_direction": expected_direction,
            "candidate_config": candidate_config,
        }

        self.proposed_configs_history.append(proposal)
        return proposal
