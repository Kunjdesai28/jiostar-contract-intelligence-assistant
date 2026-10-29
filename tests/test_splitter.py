"""Unit tests for deterministic train/test splitting logic."""

import hashlib
import pytest

from src.data.splitter import assign_split, split_benchmark


def test_assign_split_determinism():
    """Verify that identical query text always receives the exact same split."""
    query = "What is the governing law of the agreement?"
    split1 = assign_split(query)
    split2 = assign_split(query)
    assert split1 == split2
    assert split1 in {"train", "test"}


def test_assign_split_whitespace_stripping():
    """Verify that leading/trailing whitespaces are stripped prior to hashing."""
    base = "Highlight the termination clause."
    padded = "   Highlight the termination clause.  \n\t "
    assert assign_split(base) == assign_split(padded)


def test_assign_split_independent_calculation():
    """Verify that assign_split matches manual SHA-256 integer calculation."""
    sample_queries = [
        "Query Alpha",
        "Does this contract have non-compete covenants?",
        "What is the expiration date?",
        "Arbitration clauses and governing law jurisdiction",
    ]
    for q in sample_queries:
        expected_h = int(hashlib.sha256(q.strip().encode("utf-8")).hexdigest(), 16)
        expected_split = "test" if expected_h % 10 < 3 else "train"
        assert assign_split(q) == expected_split


def test_split_benchmark_partition_integrity(tmp_path):
    """Verify that split_benchmark partitions records without loss or duplication."""
    records = [{"query": f"Test Question {i}", "data": i} for i in range(50)]
    train_out = tmp_path / "train.json"
    test_out = tmp_path / "test.json"

    train_recs, test_recs, stats = split_benchmark(
        records,
        train_output_path=train_out,
        test_output_path=test_out,
    )

    assert len(train_recs) + len(test_recs) == 50
    assert stats["total_queries"] == 50
    assert stats["train_count"] == len(train_recs)
    assert stats["test_count"] == len(test_recs)

    # Every record has correct split field
    for r in train_recs:
        assert r["split"] == "train"
        assert assign_split(r["query"]) == "train"
    for r in test_recs:
        assert r["split"] == "test"
        assert assign_split(r["query"]) == "test"

    # Disjoint check
    train_queries = {r["query"] for r in train_recs}
    test_queries = {r["query"] for r in test_recs}
    assert train_queries.isdisjoint(test_queries)
