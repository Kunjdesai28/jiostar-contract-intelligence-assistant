"""Contract discovery, path safety resolution, UTF-8 loading, and manifest generation."""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def extract_referenced_contract_paths(tests: List[Dict[str, Any]]) -> List[str]:
    """Inspect benchmark records and extract unique contract relative paths from snippets."""
    referenced: Set[str] = set()
    for t in tests:
        snippets = t.get("snippets", [])
        for s in snippets:
            fp = s.get("file_path")
            if not fp:
                raise ValueError(f"Snippet missing file_path in test query: {t.get('query')}")
            referenced.add(fp)
    return sorted(list(referenced))


def resolve_safe_corpus_path(corpus_root: Path, rel_path: str) -> Path:
    """Safely resolve contract path within corpus root, rejecting directory traversal attempts."""
    resolved_corpus_root = corpus_root.resolve()
    # Normalize slashes
    clean_rel = os.path.normpath(rel_path.strip()).lstrip("/\\")
    target_path = (resolved_corpus_root / clean_rel).resolve()

    # Security check: Ensure target does not escape corpus_root
    try:
        target_path.relative_to(resolved_corpus_root)
    except ValueError:
        raise PermissionError(f"Path traversal detected! Attempted path '{rel_path}' escapes root '{resolved_corpus_root}'")

    return target_path


def load_indexed_contracts(
    tests: List[Dict[str, Any]],
    corpus_root: Path | str,
    manifest_output_path: Optional[Path | str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Discover, validate, and load all contracts referenced by the selected benchmark records.

    For each unique contract:
      - Validates file exists and is within corpus_root
      - Reads exact original UTF-8 text without normalization
      - Generates stable document ID and content SHA-256

    Args:
        tests: List of selected benchmark records.
        corpus_root: Root directory of downloaded corpus (e.g. data/raw/corpus).
        manifest_output_path: Optional path to save contract_manifest.json.

    Returns:
        Tuple of (contracts_collection_dict, manifest_summary_dict).
    """
    c_root = Path(corpus_root).resolve()
    if not c_root.exists():
        raise FileNotFoundError(f"Corpus root does not exist: {c_root}")

    referenced_rel_paths = extract_referenced_contract_paths(tests)
    logger.info(f"Discovered {len(referenced_rel_paths)} unique contract references across benchmark records.")

    contracts: Dict[str, Dict[str, Any]] = {}
    manifest_entries: List[Dict[str, Any]] = []

    for idx, rel_path in enumerate(referenced_rel_paths):
        resolved_path = resolve_safe_corpus_path(c_root, rel_path)

        if not resolved_path.exists():
            raise FileNotFoundError(f"Referenced contract file missing: {resolved_path} (from '{rel_path}')")
        if not resolved_path.is_file():
            raise ValueError(f"Target contract path is not a file: {resolved_path}")

        # Read using strict UTF-8 text-mode without stripping or transforming
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        char_len = len(raw_text)
        content_sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        # Stable document identifier (e.g. DOC_001, DOC_002 or based on normalized basename)
        doc_id = f"DOC_{idx+1:03d}_{resolved_path.stem}"

        contract_info = {
            "doc_id": doc_id,
            "benchmark_relative_path": rel_path,
            "local_file_path": str(resolved_path),
            "text": raw_text,
            "char_length": char_len,
            "content_sha256": content_sha256,
        }

        manifest_entry = {
            "doc_id": doc_id,
            "benchmark_relative_path": rel_path,
            "local_file_path": str(resolved_path),
            "char_length": char_len,
            "content_sha256": content_sha256,
        }

        contracts[rel_path] = contract_info
        manifest_entries.append(manifest_entry)

    manifest_summary = {
        "unique_contract_count": len(contracts),
        "corpus_root": str(c_root),
        "contracts": manifest_entries,
    }

    if manifest_output_path:
        out_p = Path(manifest_output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        serialized_manifest = json.dumps(manifest_summary, indent=2, ensure_ascii=False)
        out_p.write_text(serialized_manifest, encoding="utf-8")
        manifest_summary["manifest_sha256"] = hashlib.sha256(serialized_manifest.encode("utf-8")).hexdigest()
        logger.info(f"Saved contract manifest to {out_p} ({len(contracts)} contracts)")

    return contracts, manifest_summary
