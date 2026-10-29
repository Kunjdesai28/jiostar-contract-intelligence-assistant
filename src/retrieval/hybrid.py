"""Hybrid Retrieval combining BM25 and Semantic Retrieval using Reciprocal Rank Fusion (RRF).

RRF Formula:
  RRF_score(d) = sum_{r in R} (1.0 / (rrf_k + rank_r(d)))
where rank_r(d) is 1-indexed.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_rrf_scores(
    ranked_lists: List[List[Dict[str, Any]]],
    rrf_k: int = 60,
) -> Dict[str, float]:
    """Compute Reciprocal Rank Fusion scores across multiple ranked candidate lists.

    Args:
        ranked_lists: List of ranked lists, where each list contains candidate dicts with 'chunk_id'.
        rrf_k: Smoothing constant (default 60).

    Returns:
        Dict mapping chunk_id to fused RRF score.
    """
    scores: Dict[str, float] = {}
    for r_list in ranked_lists:
        for rank, item in enumerate(r_list, start=1):
            c_id = item["chunk_id"]
            reciprocal_rank = 1.0 / (float(rrf_k) + float(rank))
            scores[c_id] = scores.get(c_id, 0.0) + reciprocal_rank
    return scores


class HybridRRFRetriever:
    """Combines BM25 and Dense Semantic candidates using Reciprocal Rank Fusion."""

    def __init__(
        self,
        bm25_retriever: Any,
        semantic_retriever: Optional[Any] = None,
        rrf_k: int = 60,
        candidate_pool_size: int = 30,
    ):
        """Initialize Hybrid retriever.

        Args:
            bm25_retriever: Primary BM25Retriever instance.
            semantic_retriever: Optional DenseSemanticRetriever instance.
            rrf_k: Fusion constant k for RRF (default 60).
            candidate_pool_size: Number of candidates to retrieve from each retriever before fusion.
        """
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.rrf_k = int(rrf_k)
        self.candidate_pool_size = int(candidate_pool_size)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve and fuse candidate passages using RRF.

        Args:
            query: Query text.
            top_k: Number of final ranked candidates to return.

        Returns:
            Ranked list of candidates with RRF scores and exact metadata.
        """
        # Retrieve candidate pools
        bm25_candidates = self.bm25_retriever.search(query=query, top_k=self.candidate_pool_size)

        semantic_candidates: List[Dict[str, Any]] = []
        if self.semantic_retriever is not None and getattr(self.semantic_retriever, "is_ready", False):
            try:
                semantic_candidates = self.semantic_retriever.search(query=query, top_k=self.candidate_pool_size)
            except Exception as e:
                logger.warning(f"Semantic search failed during hybrid retrieval: {e}. Falling back to BM25.")
                semantic_candidates = []

        # If semantic is unavailable, return BM25 candidates up to top_k directly
        if not semantic_candidates:
            return bm25_candidates[:top_k]

        # Map candidate details by chunk_id
        chunk_map: Dict[str, Dict[str, Any]] = {}
        for c in bm25_candidates:
            chunk_map[c["chunk_id"]] = c
        for c in semantic_candidates:
            if c["chunk_id"] not in chunk_map:
                chunk_map[c["chunk_id"]] = c

        # Compute RRF
        rrf_scores = compute_rrf_scores(
            ranked_lists=[bm25_candidates, semantic_candidates],
            rrf_k=self.rrf_k,
        )

        # Deterministic sorting: primary by score desc, secondary by chunk_id asc
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: (-rrf_scores[cid], cid),
        )

        final_results: List[Dict[str, Any]] = []
        for rank, cid in enumerate(sorted_chunk_ids[:top_k], start=1):
            base_chunk = chunk_map[cid]
            final_results.append({
                "rank": rank,
                "score": float(rrf_scores[cid]),
                "chunk_id": base_chunk["chunk_id"],
                "doc_id": base_chunk["doc_id"],
                "file_path": base_chunk["file_path"],
                "start_char": base_chunk["start_char"],
                "end_char": base_chunk["end_char"],
                "char_length": base_chunk["char_length"],
                "text": base_chunk["text"],
                "method": "hybrid_rrf",
            })

        return final_results
