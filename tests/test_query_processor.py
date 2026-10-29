"""Tests for QueryProcessor handling benchmark preambles, natural queries, and legal preservation."""

import pytest

from src.retrieval.query_processor import QueryProcessor


def test_query_processor_benchmark_preamble():
    qp = QueryProcessor(mode="focused")
    raw = (
        "Consider the Amendment No. 3 to Wireless Content License Agreement between Fox Mobile "
        "Entertainment, Inc. and Glu Mobile Inc.; What is the expiration date of this contract?"
    )
    res = qp.process(raw)
    assert res["processing_applied"] is True
    assert res["document_context"] is not None
    assert "Amendment No. 3" in res["document_context"]
    assert res["focused_query"] == "What is the expiration date of this contract?"
    assert res["original_query"] == raw.strip()


def test_query_processor_short_natural_query():
    qp = QueryProcessor(mode="focused")
    raw = "Does this agreement permit sublicensing?"
    res = qp.process(raw)
    assert res["processing_applied"] is False
    assert res["document_context"] is None
    assert res["focused_query"] == raw


def test_query_processor_semicolon_without_preamble():
    qp = QueryProcessor(mode="focused")
    # Semicolon in normal sentence without contract preamble
    raw = "The vendor provides software; what is the warranty period?"
    res = qp.process(raw)
    # Even if split or not, substantive question must be retained
    assert "warranty period?" in res["focused_query"]


def test_query_processor_multiple_legal_conditions():
    qp = QueryProcessor(mode="focused")
    raw = (
        "Consider the Agreement; Does the licensee have exclusive rights, "
        "and is termination permitted for convenience or only for cause?"
    )
    res = qp.process(raw)
    assert res["processing_applied"] is True
    # Substantive conditions must NOT be stripped or mutated
    assert "exclusive rights" in res["focused_query"]
    assert "termination permitted for convenience" in res["focused_query"]
    assert "for cause" in res["focused_query"]


def test_query_processor_original_mode():
    qp = QueryProcessor(mode="original")
    raw = "Consider the Agreement; What is the governing law?"
    res = qp.process(raw)
    assert res["processing_applied"] is False
    assert res["focused_query"] == raw


def test_query_processor_empty_and_whitespace():
    qp = QueryProcessor(mode="focused")
    assert qp.process("")["focused_query"] == ""
    assert qp.process("   ")["focused_query"] == ""
    assert qp.process(None)["focused_query"] is None
