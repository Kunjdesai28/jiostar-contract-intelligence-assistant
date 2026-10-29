"""Abstention decision engine for legal contract retrieval."""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

STOPWORDS: Set[str] = {
    "a", "an", "the", "in", "on", "of", "to", "for", "with", "at", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "shall", "should", "will", "would",
    "may", "might", "must", "this", "that", "these", "those", "it", "its",
    "what", "which", "who", "whom", "where", "when", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "just",
    "don", "now", "contract", "agreement", "clause", "provision", "consider",
    "between", "and", "or", "there", "their", "under", "party", "parties",
    "pursuant", "hereof", "herein", "hereto", "regarding"
}


def extract_content_words(text: str) -> List[str]:
    """Extract significant content words from query text."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


class AbstentionEngine:
    """Evaluates candidate relevance scores and evidence signals to determine abstention.

    Supported strategies:
      - 'top_score_threshold': Abstains if top candidate score <= threshold.
      - 'candidate_filtering': Filters individual candidates below threshold.
      - 'normalized_score_coverage': Evaluates query-length normalized BM25 score
        and content word coverage guardrail.
    """

    def __init__(
        self,
        enabled: bool = True,
        threshold: float = 0.0,
        strategy: str = "top_score_threshold",
        min_norm_score: float = 2.4,
        min_words_for_guard: int = 5,
        max_matched_for_guard: int = 1,
    ):
        """Initialize abstention engine.

        Args:
            enabled: If False, abstention is bypassed.
            threshold: Minimum raw score threshold for score-based strategies.
            strategy: Abstention evaluation strategy.
            min_norm_score: Minimum normalized score (top_score / len(content_words)).
            min_words_for_guard: Minimum content words required to trigger coverage guardrail.
            max_matched_for_guard: Maximum matched words that trigger abstention under guardrail.
        """
        self.enabled = enabled
        self.threshold = float(threshold)
        self.strategy = strategy
        self.min_norm_score = float(min_norm_score)
        self.min_words_for_guard = int(min_words_for_guard)
        self.max_matched_for_guard = int(max_matched_for_guard)

    def evaluate(
        self,
        candidates: List[Dict[str, Any]],
        query: Optional[str] = None,
        query_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[Dict[str, Any]], str]:
        """Evaluate candidates against abstention criteria.

        Args:
            candidates: Ranked list of candidate dictionaries containing 'score'.
            query: Optional original user query string.
            query_info: Optional parsed query information from QueryProcessor.

        Returns:
            Tuple of:
              - should_abstain (bool): True if system decides to abstain.
              - filtered_candidates (List[Dict]): Candidates to return (empty if abstaining).
              - rationale (str): Explanation of decision.
        """
        if not self.enabled:
            return False, candidates, "Abstention disabled; returning all candidates."

        # Case 1: No candidates retrieved at all
        if not candidates:
            return True, [], "No matching candidates retrieved from corpus."

        top_score = float(candidates[0].get("score", 0.0))

        # Case 2: Normalized score and content coverage strategy
        if self.strategy in ("normalized_score_coverage", "combined_norm_coverage"):
            # Determine focused text for content word extraction
            target_text = ""
            if query_info and query_info.get("focused_query"):
                target_text = query_info["focused_query"]
            elif query:
                target_text = query

            content_words = extract_content_words(target_text) if target_text else []
            num_words = max(1, len(content_words))
            norm_score = top_score / num_words

            # 2a. Check normalized BM25 score per content word
            if norm_score < self.min_norm_score:
                return (
                    True,
                    [],
                    f"Normalized relevance score {norm_score:.2f} did not meet minimum threshold {self.min_norm_score:.2f}.",
                )

            # 2b. Check content word coverage guardrail for specific/out-of-domain queries
            if content_words:
                top_text = candidates[0].get("text", "").lower()
                passage_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", top_text))
                matched_words = [w for w in content_words if w in passage_words]

                if len(content_words) >= self.min_words_for_guard and len(matched_words) <= self.max_matched_for_guard:
                    return (
                        True,
                        [],
                        f"Insufficient evidence coverage: only {len(matched_words)} of {len(content_words)} "
                        f"key terms found in top candidate passage.",
                    )

            return (
                False,
                candidates,
                f"Candidate passed normalized relevance ({norm_score:.2f} >= {self.min_norm_score:.2f}) "
                f"and evidence coverage checks.",
            )

        # Case 3: Raw candidate score thresholding
        if top_score <= self.threshold:
            return (
                True,
                [],
                f"Top candidate score {top_score:.4f} did not exceed threshold {self.threshold:.4f}.",
            )

        if self.strategy == "candidate_filtering":
            filtered = [c for c in candidates if float(c.get("score", 0.0)) > self.threshold]
            return False, filtered, f"Retrieved {len(filtered)} candidates exceeding threshold {self.threshold:.4f}."

        return False, candidates, f"Top candidate score {top_score:.4f} exceeded threshold {self.threshold:.4f}."
