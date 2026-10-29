"""Unit tests for the BM25 retrieval engine."""

import pytest
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.tokenizer import LegalTokenizer


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "chunk_01",
            "doc_id": "DOC_01",
            "file_path": "c1.txt",
            "start_char": 0,
            "end_char": 100,
            "text": "This agreement shall be governed by the laws of the State of New York.",
        },
        {
            "chunk_id": "chunk_02",
            "doc_id": "DOC_01",
            "file_path": "c1.txt",
            "start_char": 100,
            "end_char": 200,
            "text": "Either party may terminate this agreement for convenience upon 30 days prior written notice.",
        },
        {
            "chunk_id": "chunk_03",
            "doc_id": "DOC_02",
            "file_path": "c2.txt",
            "start_char": 0,
            "end_char": 100,
            "text": "Licensee agrees not to grant any sublicensing rights without prior written consent.",
        },
    ]


def test_bm25_retrieval_relevance_ranking(sample_chunks):
    """Verify that a query ranks the most relevant passage first with highest score."""
    retriever = BM25Retriever(chunks=sample_chunks)
    results = retriever.search(query="sublicensing rights written consent", top_k=3)

    assert len(results) > 0
    top = results[0]
    assert top["chunk_id"] == "chunk_03"
    assert top["document"] == "c2.txt"
    assert top["score"] > 0
    assert top["rank"] == 1


def test_bm25_empty_and_whitespace_queries(sample_chunks):
    """Verify that empty or whitespace queries return empty results."""
    retriever = BM25Retriever(chunks=sample_chunks)
    assert retriever.search("") == []
    assert retriever.search("   \t\n  ") == []
    assert retriever.search(None) == []


def test_bm25_zero_overlap_query(sample_chunks):
    """Verify behavior when query shares zero vocabulary with indexed corpus."""
    retriever = BM25Retriever(chunks=sample_chunks)
    results = retriever.search("xylophone quantum astrophysics", top_k=3)
    # All scores will be 0.0
    for r in results:
        assert r["score"] == 0.0


def test_bm25_configurable_top_k(sample_chunks):
    """Verify that top_k cutoff returns exactly requested count."""
    retriever = BM25Retriever(chunks=sample_chunks)
    res_1 = retriever.search("agreement", top_k=1)
    res_2 = retriever.search("agreement", top_k=2)
    res_5 = retriever.search("agreement", top_k=5)

    assert len(res_1) == 1
    assert len(res_2) == 2
    assert len(res_5) == 3  # Only 3 total chunks exist in fixture


def test_bm25_stable_tie_breaking(sample_chunks):
    """Verify stable tie-breaking when candidate scores are identical."""
    retriever = BM25Retriever(chunks=sample_chunks)
    # Both chunk_01 and chunk_02 share 'agreement'
    res = retriever.search("agreement", top_k=3)
    ranks = [r["rank"] for r in res]
    assert ranks == [1, 2, 3]
    # Ordering must be deterministic
    res_repeat = retriever.search("agreement", top_k=3)
    assert [r["chunk_id"] for r in res] == [r["chunk_id"] for r in res_repeat]
