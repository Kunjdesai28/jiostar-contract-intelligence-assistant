"""Deterministic tokenization module for legal contracts and queries."""

import re
from typing import List, Optional


class LegalTokenizer:
    """Lightweight, deterministic tokenizer optimized for legal texts and queries.

    Features:
      - Lowercasing for case insensitivity.
      - Alphanumeric token extraction preserving legal section numbers (e.g., '10.1', '3(a)').
      - Consistent punctuation handling without aggressive stemming.
      - Robust to Unicode, punctuation sequences, and empty inputs.
    """

    def __init__(self, lowercase: bool = True, enhancement_mode: str = "none"):
        self.lowercase = lowercase
        self.enhancement_mode = enhancement_mode
        # Pattern captures Unicode alphanumeric words and compound identifiers like '10.1' or 'post-execution'
        # Matches Unicode word characters optionally connected by hyphens or internal dots
        self.token_pattern = re.compile(r"\w+(?:[-.]\w+)*", re.UNICODE)

    def tokenize(self, text: Optional[str]) -> List[str]:
        """Convert input text string into a list of normalized tokens.

        Args:
            text: Input string or None.

        Returns:
            List of string tokens. Empty list if input is empty or None.
        """
        if not text or not isinstance(text, str):
            return []

        processed = text.lower() if self.lowercase else text
        tokens = self.token_pattern.findall(processed)

        # Strip any stray trailing/leading dots or hyphens from tokens
        cleaned: List[str] = []
        for t in tokens:
            stripped = t.strip(".-")
            if stripped:
                cleaned.append(stripped)

        if self.enhancement_mode == "legal_stem":
            from src.retrieval.lexical_enhancement import normalize_legal_token
            return [normalize_legal_token(tok) for tok in cleaned]

        return cleaned


# Default global instance for convenience
_DEFAULT_TOKENIZER = LegalTokenizer(lowercase=True)


def tokenize_legal_text(text: Optional[str], lowercase: bool = True) -> List[str]:
    """Functional wrapper for legal text tokenization."""
    if lowercase:
        return _DEFAULT_TOKENIZER.tokenize(text)
    return LegalTokenizer(lowercase=False).tokenize(text)
