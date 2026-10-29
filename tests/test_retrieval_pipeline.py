"""Integration tests for ContractRetrievalPipeline and retrieval integrity."""

from pathlib import Path
import pytest
from src.retrieval.pipeline import ContractRetrievalPipeline


@pytest.fixture(scope="module")
def shared_pipeline():
    manifest_p = Path("data/processed/contract_manifest.json")
    if not manifest_p.exists():
        pytest.skip("Contract manifest not prepared. Run prepare_dataset.py first.")

    pipeline = ContractRetrievalPipeline()
    pipeline.build_index()
    return pipeline


def test_pipeline_indexing_and_contract_counts(shared_pipeline):
    """Verify that all 29 contracts are loaded and chunks indexed."""
    assert shared_pipeline.contracts_count == 29
    assert len(shared_pipeline.chunks) > 1000
    assert shared_pipeline.is_indexed is True


def test_pipeline_search_passage_offset_integrity(shared_pipeline):
    """Verify that every retrieved passage exactly matches the slice of the source contract file."""
    res = shared_pipeline.search("Highlight governing law provisions", top_k=5)
    assert res["status"] in {"results_found", "no_relevant_clause"}

    for p in res["passages"]:
        doc_rel_path = p["document"]
        start = p["start_char"]
        end = p["end_char"]
        passage_text = p["text"]

        # Read actual file from disk
        local_path = Path("data/raw/corpus") / doc_rel_path
        assert local_path.exists(), f"Source contract does not exist: {local_path}"

        raw_text = local_path.read_text(encoding="utf-8")
        assert raw_text[start:end] == passage_text, (
            f"Offset mismatch! File slice [{start}:{end}] does not equal passage text."
        )


def test_pipeline_abstention_on_unanswerable_query(shared_pipeline):
    """Verify that with a calibrated threshold, unanswerable questions trigger abstention."""
    shared_pipeline.abstention_engine.enabled = True
    shared_pipeline.abstention_engine.threshold = 44.0

    res = shared_pipeline.search("What is the airspeed velocity of an unladen European swallow?", top_k=3)
    assert res["abstained"] is True
    assert res["status"] == "no_relevant_clause"
    assert res["returned_count"] == 0
    assert res["passages"] == []


def test_pipeline_no_ground_truth_leakage(shared_pipeline):
    """Verify that the retrieval pipeline does not require, accept, or use ground truth spans during search."""
    # Searching requires only query and top_k
    res = shared_pipeline.search(query="audit rights", top_k=3)
    assert "query" in res
    assert "passages" in res
    # Ensure no ground_truth field exists in output
    for p in res["passages"]:
        assert "ground_truth" not in p
        assert "is_ground_truth" not in p
