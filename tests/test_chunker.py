"""Unit tests for document chunking logic and character-offset invariants."""

import pytest
from src.retrieval.chunker import chunk_document, chunk_collection


def test_chunk_document_exact_character_slices():
    """Verify that every chunk text matches the exact document slice."""
    doc_text = "Standard clause 1. Party A covenants to pay €500. Clause 2. Termination on 30 days notice."
    chunks = chunk_document(
        doc_id="DOC_001",
        file_path="contracts/c1.txt",
        text=doc_text,
        chunk_size=30,
        overlap=10,
    )
    assert len(chunks) > 0

    for c in chunks:
        start = c["start_char"]
        end = c["end_char"]
        assert c["text"] == doc_text[start:end]
        assert end > start
        assert start >= 0
        assert end <= len(doc_text)


def test_chunk_document_complete_coverage():
    """Verify that every character in the document is covered by at least one chunk."""
    doc_text = "A" * 1500
    chunks = chunk_document(
        doc_id="DOC_COV",
        file_path="cov.txt",
        text=doc_text,
        chunk_size=400,
        overlap=100,
    )

    covered_chars = set()
    for c in chunks:
        covered_chars.update(range(c["start_char"], c["end_char"]))

    assert len(covered_chars) == len(doc_text)
    assert covered_chars == set(range(len(doc_text)))


def test_chunk_document_short_document():
    """Verify that documents shorter than chunk_size result in a single valid chunk."""
    short_text = "Short agreement."
    chunks = chunk_document(
        doc_id="DOC_SHORT",
        file_path="short.txt",
        text=short_text,
        chunk_size=800,
        overlap=150,
    )
    assert len(chunks) == 1
    assert chunks[0]["start_char"] == 0
    assert chunks[0]["end_char"] == len(short_text)
    assert chunks[0]["text"] == short_text


def test_chunk_document_empty_document():
    """Verify that an empty document generates zero chunks."""
    chunks = chunk_document(
        doc_id="DOC_EMPTY",
        file_path="empty.txt",
        text="",
        chunk_size=800,
        overlap=150,
    )
    assert chunks == []


def test_chunk_document_unicode_handling():
    """Verify correct character offset indexing with multi-byte Unicode characters."""
    unicode_text = "Party A agrees to confidentiality in São Paulo with €1,000,000 consideration. ⚖️ Agreement."
    chunks = chunk_document(
        doc_id="DOC_UNI",
        file_path="uni.txt",
        text=unicode_text,
        chunk_size=40,
        overlap=15,
    )
    assert len(chunks) > 1
    for c in chunks:
        assert c["text"] == unicode_text[c["start_char"]:c["end_char"]]


def test_chunk_document_invalid_parameters():
    """Verify that invalid chunk_size or overlap parameters raise ValueError."""
    text = "Valid text."
    with pytest.raises(ValueError, match="positive integer"):
        chunk_document("DOC", "p", text, chunk_size=0, overlap=0)

    with pytest.raises(ValueError, match="positive integer"):
        chunk_document("DOC", "p", text, chunk_size=-10, overlap=0)

    with pytest.raises(ValueError, match="non-negative"):
        chunk_document("DOC", "p", text, chunk_size=100, overlap=-5)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        chunk_document("DOC", "p", text, chunk_size=100, overlap=100)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        chunk_document("DOC", "p", text, chunk_size=100, overlap=120)


def test_chunk_collection_determinism():
    """Verify that chunking a collection is 100% deterministic across repeated runs."""
    contracts = {
        "doc2.txt": {"doc_id": "D2", "text": "Content of document two."},
        "doc1.txt": {"doc_id": "D1", "text": "Content of document one with more characters."},
    }
    chunks1 = chunk_collection(contracts, chunk_size=20, overlap=5)
    chunks2 = chunk_collection(contracts, chunk_size=20, overlap=5)

    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1["chunk_id"] == c2["chunk_id"]
        assert c1["start_char"] == c2["start_char"]
        assert c1["end_char"] == c2["end_char"]
        assert c1["text"] == c2["text"]
