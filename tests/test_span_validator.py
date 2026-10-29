"""Unit and integration tests for ground-truth character offset span validation."""

import json
from pathlib import Path
import pytest

from src.data.span_validator import validate_ground_truth_spans


def test_span_validator_synthetic_scenarios(tmp_path):
    """Test all edge cases using controlled synthetic document fixtures."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    # Create synthetic doc with Unicode and multiline text
    doc_content = "Line 1: Hello World.\nLine 2: Confidential agreement with café € symbols.\nLine 3: Final clause."
    doc_file = corpus_dir / "test_doc.txt"
    doc_file.write_text(doc_content, encoding="utf-8")

    # 1. Valid exact span
    valid_span_start = doc_content.index("Confidential agreement")
    valid_span_end = valid_span_start + len("Confidential agreement")
    valid_tests = [
        {
            "query": "Find confidentiality clause",
            "snippets": [
                {
                    "file_path": "test_doc.txt",
                    "span": [valid_span_start, valid_span_end],
                    "answer": "Confidential agreement",
                }
            ],
        }
    ]
    is_valid, summary = validate_ground_truth_spans(valid_tests, corpus_dir)
    assert is_valid is True
    assert summary["valid_snippets"] == 1
    assert summary["invalid_snippets"] == 0

    # 2. Negative start offset
    bad_neg_start = [
        {
            "query": "Negative start",
            "snippets": [{"file_path": "test_doc.txt", "span": [-1, 10], "answer": "text"}],
        }
    ]
    is_valid, summary = validate_ground_truth_spans(bad_neg_start, corpus_dir)
    assert is_valid is False
    assert any(m["error_type"] == "negative_start_offset" for m in summary["offset_mismatches"])

    # 3. End beyond document length
    bad_beyond_end = [
        {
            "query": "End beyond doc",
            "snippets": [{"file_path": "test_doc.txt", "span": [0, len(doc_content) + 50], "answer": "text"}],
        }
    ]
    is_valid, summary = validate_ground_truth_spans(bad_beyond_end, corpus_dir)
    assert is_valid is False
    assert any(m["error_type"] == "end_offset_out_of_bounds" for m in summary["offset_mismatches"])

    # 4. Empty span (start == end)
    empty_span = [
        {
            "query": "Empty span",
            "snippets": [{"file_path": "test_doc.txt", "span": [5, 5], "answer": ""}],
        }
    ]
    is_valid, summary = validate_ground_truth_spans(empty_span, corpus_dir)
    assert is_valid is False
    assert any(m["error_type"] == "empty_or_inverted_span" for m in summary["offset_mismatches"])

    # 5. Incorrect answer text mismatch
    text_mismatch = [
        {
            "query": "Mismatched answer",
            "snippets": [
                {
                    "file_path": "test_doc.txt",
                    "span": [valid_span_start, valid_span_end],
                    "answer": "Wrong answer text",
                }
            ],
        }
    ]
    is_valid, summary = validate_ground_truth_spans(text_mismatch, corpus_dir)
    assert is_valid is False
    assert any(m["error_type"] == "text_mismatch" for m in summary["offset_mismatches"])

    # 6. Missing document
    missing_doc = [
        {
            "query": "Missing doc test",
            "snippets": [{"file_path": "nonexistent.txt", "span": [0, 5], "answer": "hello"}],
        }
    ]
    is_valid, summary = validate_ground_truth_spans(missing_doc, corpus_dir)
    assert is_valid is False
    assert summary["missing_documents_count"] == 1


def test_span_validator_real_cuad_mini():
    """Integration test: Verify 100% exact span match across actual CUAD mini benchmark."""
    mini_p = Path("data/processed/cuad_mini.json")
    corpus_p = Path("data/raw/corpus")

    if not mini_p.exists() or not corpus_p.exists():
        pytest.skip("Processed dataset not available for integration test.")

    mini_data = json.loads(mini_p.read_text(encoding="utf-8"))
    tests = mini_data["tests"]

    is_valid, summary = validate_ground_truth_spans(tests, corpus_p)
    assert is_valid is True
    assert summary["invalid_snippets"] == 0
    assert summary["missing_documents_count"] == 0
    assert summary["valid_snippets"] == 272
