"""Unified Contract Retrieval Pipeline coordinating chunking, BM25 indexing, and abstention."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import get_project_root, get_resolved_paths, load_yaml_config
from src.retrieval.abstention import AbstentionEngine
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.chunker import chunk_collection
from src.retrieval.clause_chunker import chunk_collection_clause_aware
from src.retrieval.hybrid import HybridRRFRetriever
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.semantic import DenseSemanticRetriever
from src.retrieval.tokenizer import LegalTokenizer

logger = logging.getLogger(__name__)


class ContractRetrievalPipeline:
    """Production retrieval pipeline for legal contracts.

    Supports:
      - Document chunking preserving exact UTF-8 character offsets (fixed or clause-aware).
      - Query processing (original, focused, or two-stage preamble removal).
      - BM25 Okapi and Hybrid RRF indexing and ranking.
      - Calibrated score-threshold and query-length normalized abstention.
      - Complete provenance tracking.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Path | str] = None,
    ):
        """Initialize pipeline with configuration.

        Args:
            config: Optional pre-loaded configuration dictionary.
            config_path: Optional path to baseline YAML config. Defaults to configs/baseline.yaml.
        """
        root = get_project_root()
        if config is not None:
            self.config = config
        else:
            if config_path:
                c_path = Path(config_path)
            elif (root / "configs" / "final.yaml").exists():
                c_path = root / "configs" / "final.yaml"
            else:
                c_path = root / "configs" / "baseline.yaml"
            self.config = load_yaml_config(c_path)

        # Merge data paths configuration
        self.data_paths = get_resolved_paths()

        # Component configurations
        chunk_cfg = self.config.get("chunking", {})
        self.chunk_strategy = chunk_cfg.get("strategy", "fixed_character")
        self.chunk_size = int(chunk_cfg.get("chunk_size", 800))
        self.overlap = int(chunk_cfg.get("overlap", 150))

        tok_cfg = self.config.get("tokenization", {})
        enhancement_mode = tok_cfg.get("enhancement_mode", "none")
        self.tokenizer = LegalTokenizer(
            lowercase=tok_cfg.get("lowercase", True),
            enhancement_mode=enhancement_mode,
        )

        bm25_cfg = self.config.get("bm25", {})
        self.k1 = float(bm25_cfg.get("k1", 1.5))
        self.b = float(bm25_cfg.get("b", 0.75))

        ret_cfg = self.config.get("retrieval", {})
        self.retrieval_method = ret_cfg.get("method", "bm25")
        self.default_top_k = int(ret_cfg.get("top_k", 5))

        qp_cfg = self.config.get("query_processing", {})
        self.query_mode = qp_cfg.get("mode", "original")
        self.query_processor = QueryProcessor(mode=self.query_mode)

        abs_cfg = self.config.get("abstention", {})
        self.abstention_engine = AbstentionEngine(
            enabled=abs_cfg.get("enabled", True),
            threshold=float(abs_cfg.get("threshold", 0.0)),
            strategy=abs_cfg.get("strategy", "top_score_threshold"),
            min_norm_score=float(abs_cfg.get("min_norm_score", 2.4)),
            min_words_for_guard=int(abs_cfg.get("min_words_for_guard", 5)),
            max_matched_for_guard=int(abs_cfg.get("max_matched_for_guard", 1)),
        )

        self.chunks: List[Dict[str, Any]] = []
        self.retriever: Optional[Any] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.semantic_retriever: Optional[DenseSemanticRetriever] = None
        self.is_indexed: bool = False
        self.contracts_count: int = 0
        self.contracts: Dict[str, Dict[str, Any]] = {}

    def load_contracts_corpus(self) -> Dict[str, Dict[str, Any]]:
        """Load the 29 indexed contracts using the contract manifest or raw corpus."""
        manifest_path = self.data_paths["contract_manifest_output"]
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Contract manifest not found at {manifest_path}. Please run scripts/prepare_dataset.py first."
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contracts: Dict[str, Dict[str, Any]] = {}

        for entry in manifest.get("contracts", []):
            rel_path = entry["benchmark_relative_path"]
            local_path = Path(entry["local_file_path"])
            if not local_path.exists():
                raise FileNotFoundError(f"Contract file listed in manifest missing on disk: {local_path}")

            with open(local_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            contracts[rel_path] = {
                "doc_id": entry["doc_id"],
                "file_path": rel_path,
                "local_file_path": str(local_path),
                "text": raw_text,
                "char_length": len(raw_text),
                "content_sha256": entry["content_sha256"],
            }

        self.contracts = contracts
        self.contracts_count = len(contracts)
        return contracts

    def build_index(self, force_rebuild: bool = False) -> int:
        """Chunk contracts and build the retrieval index.

        Args:
            force_rebuild: If True, re-chunks and re-indexes even if already indexed.

        Returns:
            Total number of chunks indexed.
        """
        if self.is_indexed and not force_rebuild:
            logger.info("Pipeline index already built. Skipping rebuild.")
            return len(self.chunks)

        contracts = self.load_contracts_corpus()

        if self.chunk_strategy == "clause_aware":
            logger.info("Chunking collection using clause-aware boundary detection...")
            self.chunks = chunk_collection_clause_aware(
                contracts=contracts,
                max_chunk_size=self.chunk_size,
            )
        else:
            logger.info(f"Chunking collection (fixed_character, size={self.chunk_size}, overlap={self.overlap})...")
            self.chunks = chunk_collection(
                contracts=contracts,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )

        logger.info(f"Building BM25 index over {len(self.chunks)} chunks (k1={self.k1}, b={self.b})...")
        self.bm25_retriever = BM25Retriever(
            chunks=self.chunks,
            tokenizer=self.tokenizer,
            k1=self.k1,
            b=self.b,
        )

        if self.retrieval_method == "hybrid":
            logger.info("Initializing DenseSemanticRetriever for hybrid search...")
            self.semantic_retriever = DenseSemanticRetriever(chunks=self.chunks)
            self.retriever = HybridRRFRetriever(
                bm25_retriever=self.bm25_retriever,
                semantic_retriever=self.semantic_retriever,
            )
        else:
            self.retriever = self.bm25_retriever

        self.is_indexed = True
        return len(self.chunks)

    def search_candidates(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve raw candidate passages without abstention filtering.

        Applies query processing (focused or two-stage) if configured.
        """
        if not self.is_indexed or self.retriever is None:
            self.build_index()

        q_info = self.query_processor.process(query)

        if self.query_mode == "two_stage" and q_info.get("document_context"):
            # Stage 1: Document identification using document context
            doc_context = q_info["document_context"]
            doc_candidates = self.bm25_retriever.search(query=doc_context, top_k=20)
            candidate_doc_ids = list(dict.fromkeys(c["doc_id"] for c in doc_candidates))[:3]

            # Stage 2: Filter chunks to candidate documents and rank with focused query
            filtered_chunks = [c for c in self.chunks if c["doc_id"] in candidate_doc_ids]
            if filtered_chunks:
                stage2_retriever = BM25Retriever(
                    chunks=filtered_chunks,
                    tokenizer=self.tokenizer,
                    k1=self.k1,
                    b=self.b,
                )
                return stage2_retriever.search(query=q_info["focused_query"], top_k=top_k)

        # Standard or focused mode
        active_query = q_info["focused_query"] if self.query_mode == "focused" else query
        return self.retriever.search(query=active_query, top_k=top_k)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute end-to-end retrieval, ranking, and abstention for a given question.

        Args:
            query: Question or search query string.
            top_k: Optional number of results to return (defaults to configured top_k).

        Returns:
            Structured result dictionary matching required schema.
        """
        k = int(top_k if top_k is not None else self.default_top_k)
        config_name = self.config.get("pipeline", {}).get("name", "bm25_baseline")

        if not query or not isinstance(query, str) or not query.strip():
            return {
                "query": query,
                "status": "invalid_query",
                "abstained": True,
                "abstention_rationale": "Query string is empty or invalid.",
                "requested_top_k": k,
                "returned_count": 0,
                "passages": [],
                "config_name": config_name,
            }

        try:
            # 1. Retrieve candidates
            raw_candidates = self.search_candidates(query=query, top_k=k)

            # 2. Extract query info for context-aware abstention
            q_info = self.query_processor.process(query)

            # 3. Evaluate abstention
            should_abstain, final_passages, rationale = self.abstention_engine.evaluate(
                raw_candidates, query=query, query_info=q_info
            )

            status = "no_relevant_clause" if should_abstain else "results_found"

            return {
                "query": query,
                "status": status,
                "abstained": should_abstain,
                "abstention_rationale": rationale,
                "requested_top_k": k,
                "returned_count": len(final_passages),
                "passages": final_passages,
                "config_name": config_name,
            }

        except Exception as e:
            logger.error(f"Error during retrieval search for query '{query}': {e}", exc_info=True)
            return {
                "query": query,
                "status": "system_error",
                "abstained": True,
                "abstention_rationale": f"System error during retrieval: {str(e)}",
                "requested_top_k": k,
                "returned_count": 0,
                "passages": [],
                "config_name": config_name,
            }
