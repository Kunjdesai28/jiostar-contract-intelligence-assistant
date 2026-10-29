"""Verification script testing the final selected pipeline configuration."""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.pipeline import ContractRetrievalPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_final_pipeline")


def main() -> int:
    final_cfg_path = Path("configs/final.yaml")
    if not final_cfg_path.exists():
        logger.error(f"Final configuration not found at {final_cfg_path}. Run self-improvement loop first.")
        return 1

    logger.info(f"Loading final selected pipeline configuration from {final_cfg_path}...")
    pipeline = ContractRetrievalPipeline(config_path=final_cfg_path)
    num_chunks = pipeline.build_index()
    logger.info(f"Pipeline initialized and indexed {num_chunks} chunks successfully.")

    test_queries = [
        # 1. Preamble query
        (
            "Consider the Amendment No. 3 to Wireless Content License Agreement between Fox Mobile Entertainment, Inc. and Glu Mobile Inc.; What is the expiration date of this contract?",
            "Answerable (with preamble)",
        ),
        # 2. Short natural legal query
        (
            "Does this agreement permit sublicensing or assignment?",
            "Answerable (short natural query)",
        ),
        # 3. Completely unrelated unanswerable query
        (
            "What is the average rainfall in the Amazon rainforest during monsoon season?",
            "Unanswerable (out-of-domain negative)",
        ),
    ]

    for q, desc in test_queries:
        logger.info(f"\n--- Testing Query: {desc} ---")
        logger.info(f"Query: {q}")
        res = pipeline.search(query=q, top_k=3)
        logger.info(f"Status: {res['status']}, Abstained: {res['abstained']}, Returned Count: {res['returned_count']}")
        logger.info(f"Rationale: {res['abstention_rationale']}")
        if res["passages"]:
            top = res["passages"][0]
            logger.info(f"Top Passage Rank 1 (Score={top.get('score', 0):.4f}): {top['file_path']} [{top['start_char']}:{top['end_char']}]")
            logger.info(f"Snippet excerpt: {top['text'][:120]}...")

    logger.info("\nFinal pipeline verification completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
