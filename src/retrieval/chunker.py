"""Document chunking module preserving exact character offsets and source metadata."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def chunk_document(
    doc_id: str,
    file_path: str,
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Dict[str, Any]]:
    """Divide a document's raw UTF-8 text into overlapping character passages.

    Guarantees:
      1. chunk["text"] == text[chunk["start_char"]:chunk["end_char"]] exactly.
      2. Complete document character coverage (no uncovered characters).
      3. Deterministic boundaries and stable chunk IDs.
      4. No empty chunks generated.

    Args:
        doc_id: Unique document identifier.
        file_path: Relative or canonical contract path.
        text: Raw document text string (unnormalized).
        chunk_size: Target character length per chunk (must be > 0).
        overlap: Character overlap between consecutive chunks (0 <= overlap < chunk_size).

    Returns:
        List of chunk dictionaries with exact character boundaries.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})")

    doc_len = len(text)
    if doc_len == 0:
        return []

    # If document is shorter than or equal to chunk_size, return single chunk
    if doc_len <= chunk_size:
        return [
            {
                "chunk_id": f"{doc_id}_chunk_0000",
                "doc_id": doc_id,
                "file_path": file_path,
                "start_char": 0,
                "end_char": doc_len,
                "text": text,
            }
        ]

    chunks: List[Dict[str, Any]] = []
    step = chunk_size - overlap
    start = 0
    chunk_idx = 0

    while start < doc_len:
        end = min(start + chunk_size, doc_len)
        chunk_text = text[start:end]

        # Invariant check: verify exact substring equality
        assert chunk_text == text[start:end], "Chunk text slicing mismatch!"

        chunks.append(
            {
                "chunk_id": f"{doc_id}_chunk_{chunk_idx:04d}",
                "doc_id": doc_id,
                "file_path": file_path,
                "start_char": start,
                "end_char": end,
                "text": chunk_text,
            }
        )
        chunk_idx += 1

        if end == doc_len:
            break

        start += step

    return chunks


def chunk_collection(
    contracts: Dict[str, Dict[str, Any]],
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Dict[str, Any]]:
    """Chunk all contracts in the loaded contract collection.

    Args:
        contracts: Dictionary of contracts as returned by src.data.loader.load_indexed_contracts.
        chunk_size: Character length per chunk.
        overlap: Character overlap.

    Returns:
        Flattened list of all chunks across all contracts.
    """
    all_chunks: List[Dict[str, Any]] = []
    # Sort keys for deterministic processing order
    for rel_path in sorted(contracts.keys()):
        contract = contracts[rel_path]
        doc_id = contract["doc_id"]
        text = contract["text"]
        doc_chunks = chunk_document(
            doc_id=doc_id,
            file_path=rel_path,
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(doc_chunks)

    logger.info(f"Chunked {len(contracts)} contracts into {len(all_chunks)} chunks (size={chunk_size}, overlap={overlap}).")
    return all_chunks
