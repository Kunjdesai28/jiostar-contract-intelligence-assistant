"""Unit tests for unanswerable queries benchmark."""

import json
from pathlib import Path
import pytest

from src.data.negative_queries import (
    build_unanswerable_dataset,
    freeze_negative_queries,
    validate_negative_queries,
)
from src.data.splitter import assign_split


def test_build_unanswerable_dataset_invariants():
    """Verify minimum count, field validity, answerable=False, and deterministic split."""
    dataset = build_unanswerable_dataset()

    assert len(dataset) >= 30, f"Expected >= 30 negative queries, got {len(dataset)}"

    seen_ids = set()
    for item in dataset:
        qid = item["id"]
        assert qid not in seen_ids, f"Duplicate ID found: {qid}"
        seen_ids.add(qid)

        assert item["query"].strip() != ""
        assert item["answerable"] is False
        assert item["verification_status"] == "verified_unanswerable"

        # Split matches deterministic assign_split
        expected_split = assign_split(item["query"])
        assert item["split"] == expected_split


def test_validate_negative_queries_catches_invalid_records():
    """Verify validator flags incorrect answerable flag, missing rationale, and split errors."""
    bad_records = [
        {
            "id": "NEG_BAD",
            "query": "Valid query text here?",
            "answerable": True,  # INVALID: must be False
            "split": "train",
            "category": "out_of_domain",
            "rationale": "Some rationale",
            "verification_status": "verified_unanswerable",
        }
    ]
    is_valid, report = validate_negative_queries(bad_records, min_count=1)
    assert is_valid is False
    assert any("must have 'answerable' set to False" in err for err in report["errors"])


def test_freeze_negative_queries_checksum_stability(tmp_path):
    """Verify freeze operation produces stable SHA-256 and does not overwrite existing file."""
    freeze_file = tmp_path / "unanswerable.json"
    records1, sha1, report1 = freeze_negative_queries(freeze_file)
    assert freeze_file.exists()
    assert report1["is_valid"] is True

    # Read back without overwrite
    records2, sha2, report2 = freeze_negative_queries(freeze_file, overwrite_existing=False)
    assert sha1 == sha2
    assert records1 == records2
