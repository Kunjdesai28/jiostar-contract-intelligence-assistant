"""Test consistency across CLI, Streamlit backend, and evaluation pipeline."""

import json
from pathlib import Path
import pytest
from src.retrieval.pipeline import ContractRetrievalPipeline
from cli import main as cli_main


def test_pipeline_and_cli_output_consistency(capsys, monkeypatch):
    """Verify that CLI search producing JSON matches the pipeline output directly."""
    query = "Highlight provisions concerning confidentiality and non-disclosure."
    top_k = 3

    # Direct pipeline execution
    pipeline = ContractRetrievalPipeline()
    pipeline.build_index()
    direct_res = pipeline.search(query=query, top_k=top_k)

    # CLI execution with --format json
    test_args = ["cli.py", "search", "--query", query, "--top-k", str(top_k), "--format", "json"]
    monkeypatch.setattr("sys.argv", test_args)

    ret_code = cli_main()
    assert ret_code == 0

    captured = capsys.readouterr()
    cli_res = json.loads(captured.out)

    assert cli_res["query"] == direct_res["query"]
    assert cli_res["status"] == direct_res["status"]
    assert cli_res["abstained"] == direct_res["abstained"]
    assert cli_res["returned_count"] == direct_res["returned_count"]

    assert len(cli_res["passages"]) == len(direct_res["passages"])
    for cli_p, dir_p in zip(cli_res["passages"], direct_res["passages"]):
        assert cli_p["chunk_id"] == dir_p["chunk_id"]
        assert cli_p["document"] == dir_p["document"]
        assert cli_p["start_char"] == dir_p["start_char"]
        assert cli_p["end_char"] == dir_p["end_char"]
        assert cli_p["score"] == dir_p["score"]
        assert cli_p["text"] == dir_p["text"]
