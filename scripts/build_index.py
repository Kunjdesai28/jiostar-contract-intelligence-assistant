#!/usr/bin/env python3
"""Build and verify the BM25 retrieval index over the 29 contracts."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.retrieval.pipeline import ContractRetrievalPipeline


def main():
    print("=" * 70)
    print("JioStar Contract Intelligence Assistant - Build BM25 Index")
    print("=" * 70)
    try:
        pipeline = ContractRetrievalPipeline()
        total_chunks = pipeline.build_index(force_rebuild=True)
        total_tokens = sum(len(c) for c in pipeline.retriever.tokenized_corpus)

        print("\n[SUCCESS] BM25 Index constructed successfully.")
        print(f"  Indexed Contracts:  {pipeline.contracts_count} contracts")
        print(f"  Generated Chunks:   {total_chunks} chunks (chunk_size={pipeline.chunk_size}, overlap={pipeline.overlap})")
        print(f"  Total Indexed Terms:{total_tokens:,} tokens")
        print(f"  BM25 Parameters:    k1={pipeline.k1}, b={pipeline.b}")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Failed to build index: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
