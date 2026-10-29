"""Tests for LexicalEnhancer and legal term normalization."""

import pytest

from src.retrieval.lexical_enhancement import LexicalEnhancer, normalize_legal_token


def test_morphological_normalization():
    # Sublicensing variations
    assert normalize_legal_token("sublicensing") == "sublicense"
    assert normalize_legal_token("sublicensed") == "sublicense"
    assert normalize_legal_token("sublicense") == "sublicense"

    # Indemnification variations
    assert normalize_legal_token("indemnification") == "indemnify"
    assert normalize_legal_token("indemnified") == "indemnify"
    assert normalize_legal_token("indemnifying") == "indemnify"

    # Termination variations
    assert normalize_legal_token("termination") == "terminate"
    assert normalize_legal_token("terminated") == "terminate"

    # Assignment variations
    assert normalize_legal_token("assignment") == "assign"
    assert normalize_legal_token("assigning") == "assign"


def test_strict_preservation_of_critical_legal_terms():
    # Modals and critical distinctions must never be collapsed or corrupted
    assert normalize_legal_token("shall") == "shall"
    assert normalize_legal_token("may") == "may"
    assert normalize_legal_token("must") == "must"
    assert normalize_legal_token("exclusive") == "exclusive"
    assert normalize_legal_token("non-exclusive") == "non-exclusive"
    assert normalize_legal_token("cause") == "cause"
    assert normalize_legal_token("convenience") == "convenience"


def test_enhancer_modes():
    tokens = ["sublicensing", "shall", "indemnification", "exclusive"]
    enhancer_none = LexicalEnhancer(mode="none")
    assert enhancer_none.enhance_tokens(tokens) == tokens

    enhancer_stem = LexicalEnhancer(mode="legal_stem")
    enhanced = enhancer_stem.enhance_tokens(tokens)
    assert enhanced == ["sublicense", "shall", "indemnify", "exclusive"]
