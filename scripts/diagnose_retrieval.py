#!/usr/bin/env python3
"""Diagnostic audit script to inspect retrieval performance on training queries."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.retrieval.pipeline import ContractRetrievalPipeline
from src.evaluation.metrics import compute_query_character_metrics


def run_diagnostics(num_samples: int = 15, output_file: str = "experiments/phase2_failure_diagnostics.json"):
    print("=" * 80)
    print("JioStar Contract Intelligence Assistant — Retrieval Failure Diagnostics Audit")
    print("=" * 80)

    pipeline = ContractRetrievalPipeline()
    pipeline.build_index()

    train_path = Path("data/processed/train_queries.json")
    if not train_path.exists():
        print(f"Error: {train_path} does not exist!")
        return 1

    train_data = json.loads(train_path.read_text(encoding="utf-8"))
    tests = train_data.get("tests", [])

    print(f"Loaded {len(tests)} answerable training queries. Auditing top {num_samples} samples...\n")

    diagnostic_records: List[Dict[str, Any]] = []

    doc_hits_at_1 = 0
    doc_hits_at_5 = 0
    doc_hits_at_10 = 0
    span_overlaps_at_1 = 0
    span_overlaps_at_5 = 0
    span_overlaps_at_10 = 0

    for idx in range(min(num_samples, len(tests))):
        test_item = tests[idx]
        query = test_item["query"]
        gold_snippets = test_item.get("snippets", [])
        primary_gold = gold_snippets[0] if gold_snippets else {}
        gold_doc = primary_gold.get("file_path", "")
        gold_span = primary_gold.get("span", [0, 0])
        gold_text = primary_gold.get("answer", "")

        res = pipeline.search(query=query, top_k=10)
        passages = res["passages"]

        ret_docs = [p["file_path"] for p in passages]
        hit_1 = (ret_docs[0] == gold_doc) if ret_docs else False
        hit_5 = (gold_doc in ret_docs[:5])
        hit_10 = (gold_doc in ret_docs[:10])

        if hit_1:
            doc_hits_at_1 += 1
        if hit_5:
            doc_hits_at_5 += 1
        if hit_10:
            doc_hits_at_10 += 1

        # Check character overlaps
        m1 = compute_query_character_metrics(gold_snippets, passages[:1])
        m5 = compute_query_character_metrics(gold_snippets, passages[:5])
        m10 = compute_query_character_metrics(gold_snippets, passages[:10])

        has_ov_1 = m1["overlap_chars"] > 0
        has_ov_5 = m5["overlap_chars"] > 0
        has_ov_10 = m10["overlap_chars"] > 0

        if has_ov_1:
            span_overlaps_at_1 += 1
        if has_ov_5:
            span_overlaps_at_5 += 1
        if has_ov_10:
            span_overlaps_at_10 += 1

        # Determine failure cause
        if has_ov_5:
            failure_category = "SUCCESS_IN_TOP_5"
        elif hit_5:
            failure_category = "CHUNK_BOUNDARY_OR_WITHIN_DOC_RANKING"
        elif hit_10:
            failure_category = "RANKED_BETWEEN_6_AND_10"
        else:
            failure_category = "WRONG_DOCUMENT_RETRIEVED"

        record = {
            "query_index": idx,
            "query": query,
            "gold_document": gold_doc,
            "gold_span": gold_span,
            "gold_text_preview": gold_text[:100] + ("..." if len(gold_text) > 100 else ""),
            "doc_hit_at_1": hit_1,
            "doc_hit_at_5": hit_5,
            "doc_hit_at_10": hit_10,
            "overlap_chars_at_1": m1["overlap_chars"],
            "overlap_chars_at_5": m5["overlap_chars"],
            "recall_at_5": m5["recall"],
            "precision_at_5": m5["precision"],
            "failure_category": failure_category,
            "top_5_passages": [
                {
                    "rank": p["rank"],
                    "document": p["file_path"],
                    "is_gold_doc": (p["file_path"] == gold_doc),
                    "start_char": p["start_char"],
                    "end_char": p["end_char"],
                    "score": p["score"],
                    "text_preview": p["text"].strip().replace("\n", " ")[:120] + "...",
                }
                for p in passages[:5]
            ],
        }
        diagnostic_records.append(record)

        # Print human-readable inspection card
        print(f"[Sample #{idx + 1}]")
        print(f"  Query:            \"{query}\"")
        print(f"  Gold Document:    {gold_doc}")
        print(f"  Gold Span:        [{gold_span[0]}:{gold_span[1]}] (length: {gold_span[1] - gold_span[0]} chars)")
        print(f"  Document Hit:     At 1: {hit_1} | At 5: {hit_5} | At 10: {hit_10}")
        print(f"  Overlap at 5:     {m5['overlap_chars']} chars (Recall@5: {m5['recall']:.4f})")
        print(f"  Diagnostic Cause: {failure_category}")
        print("  Top-3 Retrieved Passages:")
        for p in passages[:3]:
            is_gold_marker = "[GOLD DOC]" if p["file_path"] == gold_doc else "[OTHER DOC]"
            print(f"    - Rank {p['rank']} ({p['score']:.2f}) {is_gold_marker} [{p['start_char']}:{p['end_char']}] {p['file_path'].split('/')[-1][:40]}")
        print("-" * 80)

    summary = {
        "num_audited": len(diagnostic_records),
        "doc_hit_rate_at_1": f"{doc_hits_at_1}/{len(diagnostic_records)} ({doc_hits_at_1/len(diagnostic_records)*100:.1f}%)",
        "doc_hit_rate_at_5": f"{doc_hits_at_5}/{len(diagnostic_records)} ({doc_hits_at_5/len(diagnostic_records)*100:.1f}%)",
        "span_overlap_rate_at_5": f"{span_overlaps_at_5}/{len(diagnostic_records)} ({span_overlaps_at_5/len(diagnostic_records)*100:.1f}%)",
        "diagnostic_breakdown": {
            "SUCCESS_IN_TOP_5": sum(1 for r in diagnostic_records if r["failure_category"] == "SUCCESS_IN_TOP_5"),
            "CHUNK_BOUNDARY_OR_WITHIN_DOC_RANKING": sum(1 for r in diagnostic_records if r["failure_category"] == "CHUNK_BOUNDARY_OR_WITHIN_DOC_RANKING"),
            "RANKED_BETWEEN_6_AND_10": sum(1 for r in diagnostic_records if r["failure_category"] == "RANKED_BETWEEN_6_AND_10"),
            "WRONG_DOCUMENT_RETRIEVED": sum(1 for r in diagnostic_records if r["failure_category"] == "WRONG_DOCUMENT_RETRIEVED"),
        },
        "records": diagnostic_records,
    }

    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[SUMMARY] Diagnostics written to {out_p}")
    print(f"  Doc Hit Rate @ 1: {summary['doc_hit_rate_at_1']}")
    print(f"  Doc Hit Rate @ 5: {summary['doc_hit_rate_at_5']}")
    print(f"  Span Overlap Rate @ 5: {summary['span_overlap_rate_at_5']}")
    print(f"  Breakdown: {summary['diagnostic_breakdown']}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(run_diagnostics(num_samples=15))
