"""Unit tests for the legal text tokenizer."""

import pytest
from src.retrieval.tokenizer import LegalTokenizer, tokenize_legal_text


def test_tokenizer_lowercasing_and_punctuation():
    """Verify that casing is normalized and punctuation handled cleanly."""
    tokenizer = LegalTokenizer(lowercase=True)
    text = "Section 10.1: Termination for Convenience! (Effective Immediately)."
    tokens = tokenizer.tokenize(text)

    assert "section" in tokens
    assert "10.1" in tokens
    assert "termination" in tokens
    assert "convenience" in tokens
    assert "effective" in tokens
    assert "immediately" in tokens
    # Ensure punctuation like '!' or ':' does not pollute tokens
    for t in tokens:
        assert not t.endswith(":")
        assert not t.endswith("!")


def test_tokenizer_legal_expressions_and_hyphenation():
    """Verify retention of hyphenated and specialized legal terminology."""
    tokenizer = LegalTokenizer()
    text = "non-competition, sub-licensing, intellectual property, and co-branding agreements."
    tokens = tokenizer.tokenize(text)

    assert "non-competition" in tokens
    assert "sub-licensing" in tokens
    assert "intellectual" in tokens
    assert "property" in tokens
    assert "co-branding" in tokens


def test_tokenizer_numbers_and_section_codes():
    """Verify extraction of numeric section and subsection codes."""
    tokenizer = LegalTokenizer()
    text = "Under Exhibit 10.26, Clause 4.1.2 applies to Party A."
    tokens = tokenizer.tokenize(text)

    assert "10.26" in tokens
    assert "4.1.2" in tokens
    assert "exhibit" in tokens


def test_tokenizer_unicode_and_symbols():
    """Verify handling of currency symbols, accented characters, and symbols."""
    tokenizer = LegalTokenizer()
    text = "Compensation of €50,000 for café operations under §12."
    tokens = tokenizer.tokenize(text)

    assert "café" in tokens
    assert "50" in tokens
    assert "000" in tokens
    assert "12" in tokens


def test_tokenizer_empty_and_none_inputs():
    """Verify that empty, whitespace, and None inputs return empty lists."""
    tokenizer = LegalTokenizer()
    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize("   \n\t  ") == []
    assert tokenizer.tokenize(None) == []


def test_tokenize_legal_text_functional_wrapper():
    """Verify functional wrapper works consistently with class."""
    tokens = tokenize_legal_text("Governing Law: State of Delaware.")
    assert tokens == ["governing", "law", "state", "of", "delaware"]
