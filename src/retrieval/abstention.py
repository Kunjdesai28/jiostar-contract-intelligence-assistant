"""Abstention decision engine for legal contract retrieval."""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AbstentionEngine:
    """Evaluates candidate relevance scores to determine whether the system should abstain.

    Strategies supported:
      - 'top_score_threshold': If the top-1 retrieval candidate's score is <= threshold,
        the system abstains and returns no passages.
      - 'candidate_filtering': In addition to top-1 abstention, filters out individual candidates
        scoring below the threshold.
    """

    def __init__(
        self,
        enabled: bool = True,
        threshold: float = 0.0,
        strategy: str = "top_score_threshold",
    ):
        """Initialize abstention engine.

        Args:
            enabled: If False, abstention is bypassed and all ranked candidates are returned.
            threshold: Minimum score threshold. Boundary rule: score <= threshold triggers abstention.
            strategy: Abstention evaluation strategy.
        """
        self.enabled = enabled
        self.threshold = float(threshold)
        self.strategy = strategy

    def evaluate(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Tuple[bool, List[Dict[str, Any]], str]:
        """Evaluate candidates against the abstention threshold.

        Args:
            candidates: Ranked list of candidate dictionaries containing 'score'.

        Returns:
            Tuple of:
              - should_abstain (bool): True if system decides to abstain.
              - filtered_candidates (List[Dict]): Candidates to return (empty if abstaining).
              - rationale (str): Explanation of decision.
        """
        if not self.enabled:
            return False, candidates, "Abstention disabled; returning all candidates."

        # Case 1: No candidates retrieved at all (e.g. query has no lexical overlap)
        if not candidates:
            return True, [], "No matching candidates retrieved from corpus."

        top_score = float(candidates[0].get("score", 0.0))

        # Case 2: Top candidate score does not strictly exceed threshold
        # Boundary: score <= threshold triggers abstention
        if top_score <= self.threshold:
            return (
                True,
                [],
                f"Top candidate score {top_score:.4f} did not exceed threshold {self.threshold:.4f}.",
            )

        # Case 3: Confident candidates found
        if self.strategy == "candidate_filtering":
            # Retain only candidates that exceed threshold
            filtered = [c for c in candidates if float(c.get("score", 0.0)) > self.threshold]
            return False, filtered, f"Retrieved {len(filtered)} candidates exceeding threshold {self.threshold:.4f}."

        # Default strategy: top_score_threshold (return all top-K candidates if top_score > threshold)
        return False, candidates, f"Top candidate score {top_score:.4f} exceeded threshold {self.threshold:.4f}."
