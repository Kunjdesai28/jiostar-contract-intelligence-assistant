#!/usr/bin/env python3
"""Script to download the official LegalBench-RAG CUAD dataset components."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.download import acquire_cuad_dataset


def main():
    print("=" * 70)
    print("JioStar Contract Intelligence Assistant - Dataset Download")
    print("=" * 70)
    try:
        provenance = acquire_cuad_dataset()
        print("\n[SUCCESS] Dataset download and validation completed successfully.")
        print(f"  Source:         {provenance['dataset_source']}")
        print(f"  Commit SHA:     {provenance['commit_sha']}")
        print(f"  Benchmark File: {provenance['benchmark_file']}")
        print(f"  Corpus Files:   {provenance['corpus_file_count']} files under {provenance['corpus_dir']}")
        print(f"  Total Files:    {provenance['total_files']}")
        print(f"  Newly Fetched:  {provenance['newly_downloaded']}")
        print(f"  From Cache:     {provenance['cached_files']}")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Dataset acquisition failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
