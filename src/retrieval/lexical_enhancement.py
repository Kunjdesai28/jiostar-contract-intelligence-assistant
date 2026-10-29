"""Lexical Enhancement Module for Legal Terminology.

Addresses morphological variation in legal agreements (e.g. sublicensing/sublicense,
indemnification/indemnify, termination/terminate) through rule-based legal suffix
normalization while preserving critical legal distinctions (shall/may, exclusive/non-exclusive,
cause/convenience).
"""

import re
from typing import List, Optional

# Irregular legal stems mapping
LEGAL_IRREGULAR_STEMS = {
    "indemnification": "indemnify",
    "indemnifying": "indemnify",
    "indemnified": "indemnify",
    "indemnity": "indemnify",
    "indemnities": "indemnify",
    "sublicensing": "sublicense",
    "sublicensed": "sublicense",
    "sublicensee": "sublicense",
    "sublicensees": "sublicense",
    "arbitration": "arbitrate",
    "arbitrator": "arbitrate",
    "arbitrators": "arbitrate",
    "confidentiality": "confidential",
    "severability": "severable",
    "remedies": "remedy",
    "liabilities": "liability",
    "warranties": "warranty",
    "parties": "party",
    "entities": "entity",
    "indebtedness": "debt",
}

# Strict preserve list: terms whose exact form carries distinct legal consequence
PRESERVED_LEGAL_TERMS = {
    "shall",
    "may",
    "must",
    "not",
    "no",
    "exclusive",
    "non-exclusive",
    "nonexclusive",
    "cause",
    "convenience",
    "material",
    "immaterial",
    "warranty",
    "as-is",
    "covenant",
}


def normalize_legal_token(token: str) -> str:
    """Normalize a single token using legal morphological rules.

    Args:
        token: Lowercase token string.

    Returns:
        Stemmed or normalized token string.
    """
    if len(token) <= 3 or token in PRESERVED_LEGAL_TERMS:
        return token

    # Check irregular stems dictionary
    if token in LEGAL_IRREGULAR_STEMS:
        return LEGAL_IRREGULAR_STEMS[token]

    # Plurals ending in -ies -> -y (e.g., remedies -> remedy, already in dict but general fallback)
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"

    # Plurals ending in -es
    if token.endswith("es") and len(token) > 4 and token[-3] in "sxyz":
        return token[:-2]

    # Simple plurals ending in -s (avoid double 'ss' like 'loss', 'unless', 'express')
    if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
        token = token[:-1]

    # Suffix -ification -> -ify (e.g. modification -> modify, notification -> notify)
    if token.endswith("ification") and len(token) > 9:
        return token[:-9] + "ify"

    # Suffix -ation -> -e or stem (e.g. termination -> terminate, expiration -> expire, delegation -> delegate)
    if token.endswith("ation") and len(token) > 6:
        stem = token[:-5]
        if stem.endswith("termin"):
            return "terminate"
        if stem.endswith("expir"):
            return "expire"
        if stem.endswith("deleg"):
            return "delegate"
        if stem.endswith("oblig"):
            return "obligate"
        return stem + "e"

    # Suffix -ment -> stem (e.g. assignment -> assign, amendment -> amend, agreement -> agree)
    if token.endswith("ment") and len(token) > 6:
        stem = token[:-4]
        if stem.endswith("agree"):
            return stem
        if stem.endswith("assign"):
            return stem
        if stem.endswith("amend"):
            return stem
        return stem

    # Suffix -ing -> stem / -e (e.g. assigning -> assign, governing -> govern, licensing -> license)
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        if stem.endswith("licens"):
            return stem + "e"
        if stem.endswith("sublicens"):
            return stem + "e"
        if stem.endswith("govern") or stem.endswith("assign"):
            return stem
        # If double consonant, drop one (e.g. running -> run, permitting -> permit)
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
            return stem[:-1]
        return stem

    # Suffix -ed -> stem / -e (e.g. assigned -> assign, governed -> govern, terminated -> terminate)
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        if stem.endswith("terminat") or stem.endswith("licens"):
            return stem + "e"
        if stem.endswith("govern") or stem.endswith("assign"):
            return stem
        return stem

    return token


class LexicalEnhancer:
    """Configurable legal lexical enhancement engine."""

    def __init__(self, mode: str = "none"):
        """Initialize LexicalEnhancer.

        Args:
            mode: 'none' (standard tokenization) or 'legal_stem' (morphological normalization).
        """
        if mode not in ("none", "legal_stem"):
            raise ValueError(f"Unknown lexical enhancement mode: {mode}. Must be 'none' or 'legal_stem'.")
        self.mode = mode

    def enhance_tokens(self, tokens: List[str]) -> List[str]:
        """Apply lexical enhancement to a token sequence.

        Args:
            tokens: List of raw string tokens.

        Returns:
            List of enhanced tokens.
        """
        if self.mode == "none":
            return tokens

        enhanced: List[str] = []
        for t in tokens:
            normalized = normalize_legal_token(t)
            enhanced.append(normalized)
        return enhanced
