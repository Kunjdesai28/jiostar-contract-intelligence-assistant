"""Dataset acquisition module for LegalBench-RAG CUAD benchmark and corpus."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from huggingface_hub import HfApi, hf_hub_download

from src.config import get_project_root, get_resolved_paths, load_yaml_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def acquire_cuad_dataset(
    config: Optional[Dict[str, Any]] = None,
    force_redownload: bool = False,
) -> Dict[str, Any]:
    """Download official LegalBench-RAG CUAD benchmark JSON and CUAD corpus files.

    Downloads strictly the CUAD components:
      - benchmarks/cuad.json
      - corpus/cuad/*.txt

    Stores files under data/raw/ preserving relative structure:
      data/raw/benchmarks/cuad.json
      data/raw/corpus/cuad/

    Args:
        config: Optional configuration dictionary.
        force_redownload: If True, re-download files even if already present.

    Returns:
        Provenance dictionary detailing source, revision, counts, and status.
    """
    if config is None:
        config = load_yaml_config()

    resolved_paths = get_resolved_paths(config)
    raw_dir: Path = resolved_paths["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    repo_id = config["dataset"]["repo_id"]
    revision = config["dataset"]["revision"]

    logger.info(f"Connecting to Hugging Face repository '{repo_id}' at revision '{revision}'...")
    api = HfApi()

    try:
        repo_info = api.repo_info(repo_id=repo_id, revision=revision, repo_type="dataset")
        resolved_commit_sha = repo_info.sha
        logger.info(f"Repository commit SHA verified: {resolved_commit_sha}")
    except Exception as e:
        logger.error(f"Failed to query repository '{repo_id}': {e}")
        raise RuntimeError(f"Network or Hugging Face API error connecting to {repo_id}: {e}") from e

    # Retrieve file listing to filter strictly for CUAD components
    try:
        all_files = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type="dataset")
    except Exception as e:
        logger.error(f"Failed to list repository files: {e}")
        raise RuntimeError(f"Could not retrieve file list from {repo_id}: {e}") from e

    target_benchmark = config["paths"]["benchmark_file"]  # "benchmarks/cuad.json"
    target_corpus_prefix = config["paths"]["corpus_dir"].rstrip("/") + "/"  # "corpus/cuad/"

    cuad_files: List[str] = []
    if target_benchmark in all_files:
        cuad_files.append(target_benchmark)
    else:
        raise FileNotFoundError(f"Benchmark '{target_benchmark}' not found in remote repository!")

    corpus_files = [f for f in all_files if f.startswith(target_corpus_prefix)]
    if not corpus_files:
        raise FileNotFoundError(f"No corpus files found under '{target_corpus_prefix}' in repository!")

    cuad_files.extend(corpus_files)
    logger.info(f"Identified {len(cuad_files)} CUAD target files (1 benchmark + {len(corpus_files)} corpus files).")

    # Download each target file preserving relative directory layout
    downloaded_count = 0
    skipped_count = 0

    for rel_path in cuad_files:
        dest_path = raw_dir / rel_path
        if dest_path.exists() and not force_redownload and dest_path.stat().st_size > 0:
            skipped_count += 1
            continue

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            downloaded_file = hf_hub_download(
                repo_id=repo_id,
                filename=rel_path,
                revision=revision,
                repo_type="dataset",
                local_dir=str(raw_dir),
            )
            downloaded_count += 1
        except Exception as e:
            logger.error(f"Failed to download '{rel_path}': {e}")
            raise RuntimeError(f"Download failed for '{rel_path}': {e}") from e

    logger.info(f"Acquisition complete. Newly downloaded: {downloaded_count}, Already present: {skipped_count}.")

    # Validate that all files exist and are readable UTF-8
    logger.info("Validating downloaded files readability...")
    benchmark_path = raw_dir / target_benchmark
    if not benchmark_path.exists() or benchmark_path.stat().st_size == 0:
        raise RuntimeError(f"Benchmark file {benchmark_path} is missing or empty after download!")

    try:
        with open(benchmark_path, "r", encoding="utf-8") as bf:
            benchmark_data = json.load(bf)
        logger.info(f"Verified benchmark JSON syntax: contains {len(benchmark_data.get('tests', []))} test records.")
    except Exception as e:
        raise RuntimeError(f"Corrupted benchmark JSON at {benchmark_path}: {e}") from e

    # Sample validate corpus files
    for cf in corpus_files:
        cp = raw_dir / cf
        if not cp.exists() or cp.stat().st_size == 0:
            raise RuntimeError(f"Corpus file {cp} is missing or empty!")

    provenance = {
        "dataset_source": config["dataset"]["source"],
        "repo_id": repo_id,
        "revision_configured": revision,
        "commit_sha": resolved_commit_sha,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_file": target_benchmark,
        "corpus_dir": target_corpus_prefix,
        "corpus_file_count": len(corpus_files),
        "total_files": len(cuad_files),
        "newly_downloaded": downloaded_count,
        "cached_files": skipped_count,
    }

    provenance_path = raw_dir / "provenance.json"
    with open(provenance_path, "w", encoding="utf-8") as pf:
        json.dump(provenance, pf, indent=2)

    logger.info(f"Provenance saved to {provenance_path}")
    return provenance
