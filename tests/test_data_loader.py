"""Unit tests for contract discovery, path security, and document loading."""

import hashlib
from pathlib import Path
import pytest

from src.data.loader import extract_referenced_contract_paths, load_indexed_contracts, resolve_safe_corpus_path


def test_extract_referenced_contract_paths_deduplication():
    """Verify that multiple snippets referencing identical files are cleanly deduplicated."""
    tests = [
        {"query": "Q1", "snippets": [{"file_path": "cuad/docA.txt", "span": [0, 10], "answer": "a"}]},
        {"query": "Q2", "snippets": [
            {"file_path": "cuad/docA.txt", "span": [20, 30], "answer": "b"},
            {"file_path": "cuad/docB.txt", "span": [5, 15], "answer": "c"}
        ]},
        {"query": "Q3", "snippets": [{"file_path": "cuad/docB.txt", "span": [50, 60], "answer": "d"}]},
    ]
    extracted = extract_referenced_contract_paths(tests)
    assert extracted == ["cuad/docA.txt", "cuad/docB.txt"]


def test_resolve_safe_corpus_path_traversal_prevention(tmp_path):
    """Verify that path traversal attempts (e.g. ../../) are caught and rejected."""
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()

    # Valid relative subpath
    valid = resolve_safe_corpus_path(corpus_root, "sub/contract.txt")
    assert valid == (corpus_root / "sub" / "contract.txt").resolve()

    # Directory traversal attack
    with pytest.raises(PermissionError, match="Path traversal detected"):
        resolve_safe_corpus_path(corpus_root, "../../etc/passwd")

    with pytest.raises(PermissionError, match="Path traversal detected"):
        resolve_safe_corpus_path(corpus_root, "../secret.txt")


def test_load_indexed_contracts_utf8_and_hash(tmp_path):
    """Verify UTF-8 text preservation and SHA-256 calculation."""
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    doc_path = corpus_root / "doc1.txt"
    sample_text = "Standard legal agreement: §1. Party A shall pay €500.\nLine two."
    doc_path.write_text(sample_text, encoding="utf-8")

    tests = [{"query": "Q", "snippets": [{"file_path": "doc1.txt", "span": [0, 10], "answer": "Standard"}]}]
    manifest_file = tmp_path / "manifest.json"

    contracts, manifest = load_indexed_contracts(tests, corpus_root, manifest_output_path=manifest_file)
    assert "doc1.txt" in contracts
    contract_data = contracts["doc1.txt"]

    assert contract_data["text"] == sample_text
    assert contract_data["char_length"] == len(sample_text)
    assert contract_data["content_sha256"] == hashlib.sha256(sample_text.encode("utf-8")).hexdigest()
    assert manifest_file.exists()
