"""Query Processor for Legal Clause Retrieval.

Separates contract-identification preambles from substantive legal questions
to eliminate preamble bias where contract title and party names dominate
document header ranking over the target contractual clause.
"""

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Common preamble patterns found in legal benchmark questions:
# "Consider the Amendment No. 3 to Wireless Content License Agreement between Fox Mobile Entertainment, Inc. and Glu Mobile Inc.; What is the expiration date of this contract?"
PREAMBLE_PATTERNS = [
    # 1. "Consider the <Title> between <Parties>; <Question>" (semicolon delimited)
    re.compile(
        r"^\s*(?:please\s+)?consider\s+(?:the\s+)?(.*?);\s*([A-Z0-9].*)$",
        re.IGNORECASE | re.DOTALL,
    ),
    # 2. "Consider the <Title> ... . <Question starting with interrogative word>"
    re.compile(
        r"^\s*(?:please\s+)?consider\s+(?:the\s+)?(.*?)\.\s+((?:What|Is|Are|Does|Do|Did|Can|Could|Which|Where|When|Who|How|Identify|Specify|State)\b.*)$",
        re.IGNORECASE | re.DOTALL,
    ),
    # 3. "In / Under / Regarding / With respect to the <Title> [between ...]; <Question>"
    re.compile(
        r"^\s*(?:in|under|regarding|for|with respect to|referring to)\s+(?:the\s+)?(.*?)(?:;\s*|\:\s*)([A-Z0-9].*)$",
        re.IGNORECASE | re.DOTALL,
    ),
    # 4. Generic semicolon separation where first part introduces contract and second part asks question
    re.compile(
        r"^\s*(.*?(?:agreement|contract|amendment|lease|license|plan|exhibit|indenture).*?);\s*([A-Z0-9].*)$",
        re.IGNORECASE | re.DOTALL,
    ),
]

# Typical interrogative or imperative starts for substantive legal questions
QUESTION_INDICATORS = (
    "what",
    "is",
    "are",
    "does",
    "do",
    "did",
    "can",
    "could",
    "will",
    "would",
    "shall",
    "may",
    "which",
    "where",
    "when",
    "who",
    "how",
    "identify",
    "specify",
    "state",
    "find",
    "describe",
    "determine",
    "list",
    "provide",
)


class QueryProcessor:
    """Deterministic query processor separating document context and legal information need."""

    def __init__(self, mode: str = "focused"):
        """Initialize QueryProcessor.

        Args:
            mode: Processing mode:
                - 'original': returns original query without modification.
                - 'focused': extracts substantive legal question, removing preamble bias.
                - 'two_stage': separates document-identification context from substantive question.
        """
        if mode not in ("original", "focused", "two_stage"):
            raise ValueError(f"Unknown query processing mode: {mode}. Must be 'original', 'focused', or 'two_stage'.")
        self.mode = mode

    def process(self, query: str) -> Dict[str, Any]:
        """Process an input query string into document context and focused question.

        Args:
            query: Raw user or benchmark query.

        Returns:
            Dict containing:
                - original_query: str
                - document_context: Optional[str]
                - focused_query: str
                - processing_applied: bool
                - mode: str
        """
        if not query or not isinstance(query, str):
            return {
                "original_query": query,
                "document_context": None,
                "focused_query": query,
                "processing_applied": False,
                "mode": self.mode,
            }

        cleaned = query.strip()
        if self.mode == "original":
            return {
                "original_query": cleaned,
                "document_context": None,
                "focused_query": cleaned,
                "processing_applied": False,
                "mode": "original",
            }

        # Attempt preamble extraction
        doc_context: Optional[str] = None
        substantive_q: Optional[str] = None

        for pattern in PREAMBLE_PATTERNS:
            match = pattern.match(cleaned)
            if match:
                preamble_candidate = match.group(1).strip()
                question_candidate = match.group(2).strip()

                # Verify question candidate looks like a substantive question
                first_word = question_candidate.split()[0].lower() if question_candidate.split() else ""
                has_q_indicator = any(first_word.startswith(ind) for ind in QUESTION_INDICATORS)
                ends_with_qmark = question_candidate.endswith("?")

                if (has_q_indicator or ends_with_qmark) and len(question_candidate.split()) >= 3:
                    doc_context = preamble_candidate
                    substantive_q = question_candidate
                    break

        if doc_context and substantive_q:
            # Conservative cleanup: make sure substantive question retains full qualifiers
            focused_text = substantive_q if self.mode in ("focused", "two_stage") else cleaned
            return {
                "original_query": cleaned,
                "document_context": doc_context,
                "focused_query": focused_text,
                "processing_applied": True,
                "mode": self.mode,
            }

        # If no preamble pattern matched, retain original query as focused query
        return {
            "original_query": cleaned,
            "document_context": None,
            "focused_query": cleaned,
            "processing_applied": False,
            "mode": self.mode,
        }
