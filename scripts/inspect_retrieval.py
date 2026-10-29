#!/usr/bin/env python3
"""Interactive inspection utility to test and inspect retrieval results."""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.retrieval.pipeline import ContractRetrievalPipeline


def main():
    parser = argparse.ArgumentParser(description="Inspect BM25 Retrieval and Passage Evidence")
    parser.add_argument("--query", type=str, required=True, help="Query string to search")
    parser.add_argument("--top-k", type=int, default=3, help="Number of passages to return (default: 3)")
    parser.add_argument("--threshold", type=float, default=None, help="Optional override for abstention threshold")

    args = parser.parse_args()

    pipeline = ContractRetrievalPipeline()
    if args.threshold is not None:
        pipeline.abstention_engine.threshold = args.threshold

    pipeline.build_index()
    res = pipeline.search(query=args.query, top_k=args.top_k)

    print("=" * 75)
    print(f"QUERY: {args.query}")
    print(f"STATUS: {res['status']} | ABSTAINED: {res['abstained']}")
    print(f"RATIONALE: {res['abstention_rationale']}")
    print(f"PASSAGES RETURNED: {res['returned_count']}")
    print("=" * 75)

    for p in res["passages"]:
        print(f"\n[RANK {p['rank']}] Score: {p['score']} | Document: {p['document']}")
        print(f"Offsets: [{p['start_char']}:{p['end_char']}] ({p['end_char'] - p['start_char']} chars)")
        print("-" * 50)
        # Display preview
        preview = p["text"].strip().replace("\n", " ")
        if len(preview) > 300:
            preview = preview[:300] + "..."
        print(preview)

    print("\n" + "=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
