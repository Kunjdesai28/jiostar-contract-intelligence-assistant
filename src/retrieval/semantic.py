"""Dense Semantic Retrieval using sentence transformer embeddings.

Features:
  - Local embedding generation and local caching to data/processed/.
  - Exact vector cosine similarity ranking.
  - Preserves exact source document paths, chunk IDs, and character offsets.
  - Graceful fallback and error reporting if dependencies/models are unavailable.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DenseSemanticRetriever:
    """Dense vector retriever using embedding similarity."""

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[Path | str] = None,
    ):
        """Initialize dense semantic retriever.

        Args:
            chunks: List of indexed chunk dictionaries.
            model_name: HuggingFace model identifier.
            cache_dir: Optional directory to store embeddings cache.
        """
        self.chunks = chunks
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/processed")
        self.model = None
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.is_ready = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Attempt to load sentence_transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading dense embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self._build_or_load_embeddings()
            self.is_ready = True
        except ImportError:
            logger.warning(
                "sentence_transformers is not installed. DenseSemanticRetriever will operate in fallback mode."
            )
            self.is_ready = False
        except Exception as e:
            logger.warning(
                f"Failed to load sentence-transformers model '{self.model_name}': {e}. Operating in fallback mode."
            )
            self.is_ready = False

    def _get_cache_key(self) -> str:
        """Compute unique deterministic cache key based on chunk IDs and model name."""
        content_repr = f"{self.model_name}:" + ":".join(c["chunk_id"] for c in self.chunks)
        return hashlib.sha256(content_repr.encode("utf-8")).hexdigest()[:16]

    def _build_or_load_embeddings(self) -> None:
        """Load embeddings from cache or encode chunks and cache."""
        if self.model is None or not self.chunks:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._get_cache_key()
        cache_file = self.cache_dir / f"embeddings_{cache_key}.npy"

        if cache_file.exists():
            logger.info(f"Loading cached dense embeddings from {cache_file}")
            self.chunk_embeddings = np.load(cache_file)
            return

        logger.info(f"Encoding {len(self.chunks)} passages with {self.model_name}...")
        texts = [c["text"] for c in self.chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        self.chunk_embeddings = embeddings
        np.save(cache_file, embeddings)
        logger.info(f"Saved {len(embeddings)} passage embeddings to cache: {cache_file}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k candidates using cosine similarity.

        Args:
            query: Query text.
            top_k: Number of candidates to return.

        Returns:
            Ranked list of candidates with exact offsets and scores.
        """
        if not self.is_ready or self.model is None or self.chunk_embeddings is None:
            raise RuntimeError(
                "DenseSemanticRetriever is not ready. sentence_transformers is unavailable or failed to initialize."
            )

        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        # Dot product of normalized vectors equals cosine similarity
        similarities = np.dot(self.chunk_embeddings, query_vec)

        top_indices = np.argsort(-similarities)[:top_k]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "score": float(similarities[idx]),
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "file_path": chunk["file_path"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "char_length": chunk["char_length"],
                "text": chunk["text"],
                "method": "semantic_dense",
            })

        return results
