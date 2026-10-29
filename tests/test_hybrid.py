"""Tests for Reciprocal Rank Fusion and Hybrid Retrieval."""

import pytest

from src.retrieval.hybrid import HybridRRFRetriever, compute_rrf_scores


def test_rrf_score_arithmetic():
    list1 = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
    list2 = [{"chunk_id": "c2"}, {"chunk_id": "c3"}]

    scores = compute_rrf_scores([list1, list2], rrf_k=60)

    # c1: 1/(60+1) = 1/61
    # c2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61
    # c3: 1/(60+2) = 1/62
    expected_c1 = 1.0 / 61.0
    expected_c2 = (1.0 / 62.0) + (1.0 / 61.0)
    expected_c3 = 1.0 / 62.0

    assert abs(scores["c1"] - expected_c1) < 1e-6
    assert abs(scores["c2"] - expected_c2) < 1e-6
    assert abs(scores["c3"] - expected_c3) < 1e-6

    # c2 appears in both lists, so its fused score must exceed c1 and c3
    assert scores["c2"] > scores["c1"]
    assert scores["c2"] > scores["c3"]


def test_hybrid_retriever_deduplication_and_ordering():
    class MockRetriever:
        def __init__(self, candidates):
            self.candidates = candidates

        def search(self, query, top_k=5):
            return self.candidates[:top_k]

    c1 = {"chunk_id": "c1", "doc_id": "d1", "file_path": "f1", "start_char": 0, "end_char": 100, "char_length": 100, "text": "t1"}
    c2 = {"chunk_id": "c2", "doc_id": "d2", "file_path": "f2", "start_char": 0, "end_char": 100, "char_length": 100, "text": "t2"}
    c3 = {"chunk_id": "c3", "doc_id": "d3", "file_path": "f3", "start_char": 0, "end_char": 100, "char_length": 100, "text": "t3"}

    bm25_mock = MockRetriever([c1, c2])
    sem_mock = MockRetriever([c2, c3])
    sem_mock.is_ready = True

    hybrid = HybridRRFRetriever(bm25_retriever=bm25_mock, semantic_retriever=sem_mock, rrf_k=60)
    results = hybrid.search(query="test", top_k=3)

    assert len(results) == 3
    # c2 appears in both, rank 1
    assert results[0]["chunk_id"] == "c2"
    assert results[0]["rank"] == 1
    # Offsets and document provenance must be strictly preserved
    assert results[0]["start_char"] == 0
    assert results[0]["end_char"] == 100
    assert results[0]["method"] == "hybrid_rrf"


def test_hybrid_fallback_when_semantic_unavailable():
    class MockRetriever:
        def __init__(self, candidates):
            self.candidates = candidates

        def search(self, query, top_k=5):
            return self.candidates[:top_k]

    c1 = {"chunk_id": "c1", "doc_id": "d1", "file_path": "f1", "start_char": 0, "end_char": 100, "char_length": 100, "text": "t1"}
    bm25_mock = MockRetriever([c1])

    # Semantic is None or is_ready False
    hybrid = HybridRRFRetriever(bm25_retriever=bm25_mock, semantic_retriever=None)
    results = hybrid.search(query="test", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
