"""BM25 Okapi retrieval engine for legal contract chunks."""

import logging
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Okapi

from src.retrieval.ranker import rank_candidates
from src.retrieval.tokenizer import LegalTokenizer

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Production BM25 Okapi retrieval engine preserving document offsets and metadata."""

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        tokenizer: Optional[LegalTokenizer] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """Initialize BM25 index over contract chunks.

        Args:
            chunks: List of chunk dictionaries containing 'text', 'chunk_id', 'file_path', 'start_char', 'end_char'.
            tokenizer: LegalTokenizer instance. Defaults to LegalTokenizer(lowercase=True).
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
        """
        self.chunks = chunks
        self.tokenizer = tokenizer or LegalTokenizer(lowercase=True)
        self.k1 = float(k1)
        self.b = float(b)

        if not self.chunks:
            logger.warning("BM25Retriever initialized with an empty chunk list!")
            self.bm25_index = None
            self.tokenized_corpus = []
        else:
            logger.info(f"Tokenizing {len(self.chunks)} chunks for BM25 index...")
            self.tokenized_corpus = [
                self.tokenizer.tokenize(chunk["text"]) for chunk in self.chunks
            ]
            self.bm25_index = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
            logger.info(f"BM25 index built successfully (k1={self.k1}, b={self.b}).")

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K ranked passages for a given query.

        Args:
            query: Natural language question or clause search terms.
            top_k: Number of ranked passages to return.

        Returns:
            List of ranked passage dictionaries formatted with source contract metadata and scores.
        """
        if not query or not isinstance(query, str) or not query.strip():
            return []

        if not self.bm25_index or not self.chunks:
            return []

        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25_index.get_scores(query_tokens)

        # Assemble candidate records
        candidates: List[Dict[str, Any]] = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            # Construct candidate result
            candidate = {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk.get("doc_id", ""),
                "document": chunk["file_path"],
                "file_path": chunk["file_path"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "text": chunk["text"],
                "score": round(float(score), 4),
            }
            candidates.append(candidate)

        # Stable tie-breaking ranker
        return rank_candidates(candidates, top_k=top_k)
