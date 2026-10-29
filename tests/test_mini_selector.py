"""Unit tests for mini benchmark selection logic."""

import json
from pathlib import Path
import pytest

from src.data.mini_selector import mini, select_and_validate_mini_benchmark


def test_mini_exact_count_and_determinism():
    """Verify that mini() selects exactly 194 records and is deterministic across runs."""
    raw_path = Path("data/raw/benchmarks/cuad.json")
    if not raw_path.exists():
        pytest.skip("Raw benchmark not downloaded yet.")

    data = json.loads(raw_path.read_text(encoding="utf-8"))
    tests = data["tests"]

    # Run selection twice
    selected_1 = mini(tests, n=194)
    selected_2 = mini(tests, n=194)

    assert len(selected_1) == 194
    assert len(selected_2) == 194

    # Assert exact identity and order
    for t1, t2 in zip(selected_1, selected_2):
        assert t1["query"] == t2["query"]
        assert t1["snippets"] == t2["snippets"]


def test_mini_preserves_original_input():
    """Verify that the input list and its items are not mutated."""
    synthetic_tests = [
        {"query": f"Query {i}", "snippets": [{"file_path": f"doc_{i % 3}.txt", "span": [0, 10], "answer": "text"}]}
        for i in range(10)
    ]
    original_copy = json.loads(json.dumps(synthetic_tests))

    selected = mini(synthetic_tests, n=5)
    assert len(selected) == 5
    assert synthetic_tests == original_copy


def test_mini_handles_fewer_than_n_records():
    """Verify that if tests <= n, all tests are returned without slicing."""
    synthetic = [
        {"query": "Q1", "snippets": [{"file_path": "a.txt", "span": [0, 1], "answer": "a"}]},
        {"query": "Q2", "snippets": [{"file_path": "b.txt", "span": [0, 1], "answer": "b"}]},
    ]
    assert mini(synthetic, n=194) == synthetic


def test_mini_malformed_snippet_handling():
    """Verify that records with missing snippets raise explicit errors when sorting."""
    bad_record = [
        {"query": "No snippets"},
        {"query": "Valid 1", "snippets": [{"file_path": "a.txt"}]},
    ]
    with pytest.raises(ValueError, match="missing required snippets"):
        mini(bad_record, n=1)

    bad_snippet = [
        {"query": "Bad snippet", "snippets": [{"span": [0, 1]}]},
        {"query": "Valid 2", "snippets": [{"file_path": "b.txt"}]},
    ]
    with pytest.raises(ValueError, match="missing 'file_path'"):
        mini(bad_snippet, n=1)


def test_select_and_validate_mini_benchmark_file(tmp_path):
    """Test full pipeline loading, selection, and output serialization."""
    raw_path = Path("data/raw/benchmarks/cuad.json")
    if not raw_path.exists():
        pytest.skip("Raw benchmark not downloaded yet.")

    out_file = tmp_path / "mini.json"
    selected, sha = select_and_validate_mini_benchmark(
        raw_benchmark_path=raw_path,
        output_path=out_file,
        expected_size=194,
    )
    assert len(selected) == 194
    assert out_file.exists()
    assert len(sha) == 64
