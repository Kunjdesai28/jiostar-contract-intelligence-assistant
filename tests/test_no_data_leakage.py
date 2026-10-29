"""Tests verifying data leakage prevention and train/test separation in Phase 3."""

import hashlib
import json
from pathlib import Path
import pytest

from src.config import get_resolved_paths
from src.retrieval.chunker import chunk_document
from src.retrieval.clause_chunker import chunk_contract_by_clauses
from src.retrieval.query_processor import QueryProcessor

FROZEN_NEGATIVE_QUERY_CHECKSUM = "7eaf304f4a22a4567100fb283a53f1bfc7b458eaa2f7feb7d78ba66db3d539d3"


def test_frozen_negative_query_checksum_immutable():
    paths = get_resolved_paths()
    unans_path = paths["unanswerable_queries_file"]
    assert unans_path.exists()
    content_bytes = unans_path.read_bytes()
    assert hashlib.sha256(content_bytes).hexdigest() == FROZEN_NEGATIVE_QUERY_CHECKSUM


def test_chunkers_do_not_accept_or_use_ground_truth():
    import inspect
    from src.retrieval.chunker import chunk_document
    from src.retrieval.clause_chunker import chunk_contract_by_clauses

    # Signature must only accept document attributes (doc_id, file_path, text), never gold snippets
    chunk_sig = inspect.signature(chunk_document)
    assert "snippets" not in chunk_sig.parameters
    assert "gold" not in chunk_sig.parameters
    assert "answers" not in chunk_sig.parameters

    clause_sig = inspect.signature(chunk_contract_by_clauses)
    assert "snippets" not in clause_sig.parameters
    assert "gold" not in clause_sig.parameters
    assert "answers" not in clause_sig.parameters


def test_query_processor_operates_strictly_on_query_string():
    import inspect
    from src.retrieval.query_processor import QueryProcessor

    sig = inspect.signature(QueryProcessor.process)
    param_names = list(sig.parameters.keys())
    assert param_names == ["self", "query"]


def test_pipeline_search_does_not_accept_gold_snippets():
    import inspect
    from src.retrieval.pipeline import ContractRetrievalPipeline

    sig = inspect.signature(ContractRetrievalPipeline.search)
    param_names = list(sig.parameters.keys())
    assert "gold" not in param_names
    assert "ground_truth" not in param_names
    assert "snippets" not in param_names
