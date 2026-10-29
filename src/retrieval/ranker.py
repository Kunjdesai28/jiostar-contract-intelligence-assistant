"""Ranking, deduplication, and stable tie-breaking for retrieval candidates."""

from typing import Any, Dict, List


def rank_candidates(
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Sort and format candidate passages with deterministic tie-breaking.

    Tie-breaking rule:
      Primary: Score descending (-score).
      Secondary: chunk_id ascending (ensures 100% deterministic ordering across runs).

    Args:
        candidates: List of candidate dictionaries with 'score', 'chunk_id', etc.
        top_k: Maximum number of ranked candidates to return.

    Returns:
        Ranked list of candidates with 1-based 'rank' assigned.
    """
    if not candidates:
        return []

    # Filter out potential duplicate chunk_ids while preserving the highest score
    seen_chunks: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        cid = c["chunk_id"]
        if cid not in seen_chunks or c["score"] > seen_chunks[cid]["score"]:
            seen_chunks[cid] = c

    unique_candidates = list(seen_chunks.values())

    # Deterministic sort: highest score first, then alphabetically by chunk_id
    sorted_candidates = sorted(
        unique_candidates,
        key=lambda item: (-float(item["score"]), str(item.get("chunk_id", ""))),
    )

    ranked: List[Dict[str, Any]] = []
    for rank_idx, item in enumerate(sorted_candidates[:top_k], start=1):
        formatted = dict(item)
        formatted["rank"] = rank_idx
        ranked.append(formatted)

    return ranked
