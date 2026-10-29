#!/usr/bin/env python3
"""Unified Command-Line Interface (CLI) for JioStar Contract Intelligence Assistant."""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.retrieval.pipeline import ContractRetrievalPipeline


def main():
    parser = argparse.ArgumentParser(
        description="JioStar Contract Intelligence Assistant CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search subcommand
    search_parser = subparsers.add_parser("search", help="Search legal clauses across indexed contracts")
    search_parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural language legal query or clause requirement",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidate passages to retrieve",
    )
    search_parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "json"],
        help="Output format (text or json)",
    )
    search_parser.add_argument(
        "--config",
        type=str,
        default="configs/final.yaml",
        help="Path to pipeline configuration YAML",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "search":
        pipeline = ContractRetrievalPipeline(config_path=args.config)
        pipeline.build_index()
        result = pipeline.search(query=args.query, top_k=args.top_k)

        if args.format == "json":
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0

        # Human-readable text format
        print("=" * 80)
        print("JioStar Contract Intelligence Assistant — Retrieval Results")
        print("=" * 80)
        print(f"Query:               {result['query']}")
        print(f"Status:              {result['status']}")
        print(f"Abstained:           {result['abstained']}")
        print(f"Decision Rationale:  {result['abstention_rationale']}")
        print(f"Passages Retrieved:  {result['returned_count']} (Requested top-k: {result['requested_top_k']})")
        print("=" * 80)

        if result["abstained"] or not result["passages"]:
            print("\n[NOTICE] No sufficiently relevant contractual clauses were found to answer this query.")
            print(f"Reason: {result['abstention_rationale']}\n")
            return 0

        for p in result["passages"]:
            print(f"\n[Rank {p['rank']}] Score: {p['score']:.4f}")
            print(f"  Source Document:  {p['document']}")
            print(f"  Document ID:      {p.get('doc_id', 'N/A')}")
            print(f"  Character Range:  [{p['start_char']}:{p['end_char']}] ({p['end_char'] - p['start_char']} characters)")
            print("  Evidence Text:")
            print("  " + "-" * 70)
            # Indent text lines cleanly
            lines = p["text"].strip().split("\n")
            for line in lines[:8]:
                print(f"    {line}")
            if len(lines) > 8:
                print("    [... remaining lines truncated in display ...]")
            print("  " + "-" * 70)

        print("\n" + "=" * 80)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
