"""Investigation script analyzing retrieval features for training-only abstention calibration."""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_resolved_paths
from src.retrieval.pipeline import ContractRetrievalPipeline

STOPWORDS = {
    "a", "an", "the", "in", "on", "of", "to", "for", "with", "at", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "shall", "should", "will", "would",
    "may", "might", "must", "this", "that", "these", "those", "it", "its",
    "what", "which", "who", "whom", "where", "when", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "just",
    "don", "now", "contract", "agreement", "clause", "provision", "consider",
    "between", "and", "or", "there", "their", "under"
}


def extract_content_words(text: str) -> List[str]:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


def main():
    paths = get_resolved_paths()
    train_pos = json.loads(paths["train_queries_output"].read_text(encoding="utf-8"))["tests"]
    all_neg = json.loads(paths["unanswerable_queries_file"].read_text(encoding="utf-8"))
    train_neg = [q for q in all_neg if q.get("split") == "train"]

    print(f"Loaded {len(train_pos)} answerable train queries and {len(train_neg)} unanswerable train queries.")

    pipeline = ContractRetrievalPipeline(config_path="configs/final_phase3.yaml")
    pipeline.build_index()

    feature_records = []

    # Process positive queries
    for q_item in train_pos:
        query = q_item["query"]
        q_info = pipeline.query_processor.process(query)
        focused_q = q_info["focused_query"]
        content_words = extract_content_words(focused_q)

        candidates = pipeline.search_candidates(query=query, top_k=5)
        top1_score = candidates[0]["score"] if candidates else 0.0
        top2_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
        score_margin = top1_score - top2_score
        top1_text = candidates[0]["text"] if candidates else ""

        # Content word overlap ratio in top-1 passage
        top1_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", top1_text.lower()))
        matched_content = [w for w in content_words if w in top1_words]
        overlap_ratio = len(matched_content) / max(1, len(content_words))

        feature_records.append({
            "is_answerable": 1,
            "query": query,
            "focused_query": focused_q,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "score_margin": score_margin,
            "num_content_words": len(content_words),
            "matched_content_words": len(matched_content),
            "content_overlap_ratio": overlap_ratio,
        })

    # Process negative queries
    for q_item in train_neg:
        query = q_item["query"]
        q_info = pipeline.query_processor.process(query)
        focused_q = q_info["focused_query"]
        content_words = extract_content_words(focused_q)

        candidates = pipeline.search_candidates(query=query, top_k=5)
        top1_score = candidates[0]["score"] if candidates else 0.0
        top2_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
        score_margin = top1_score - top2_score
        top1_text = candidates[0]["text"] if candidates else ""

        top1_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", top1_text.lower()))
        matched_content = [w for w in content_words if w in top1_words]
        overlap_ratio = len(matched_content) / max(1, len(content_words))

        feature_records.append({
            "is_answerable": 0,
            "query": query,
            "category": q_item.get("category"),
            "focused_query": focused_q,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "score_margin": score_margin,
            "num_content_words": len(content_words),
            "matched_content_words": len(matched_content),
            "content_overlap_ratio": overlap_ratio,
        })

    # Compute statistics
    pos_records = [r for r in feature_records if r["is_answerable"] == 1]
    neg_records = [r for r in feature_records if r["is_answerable"] == 0]

    def stats(records, key):
        vals = [r[key] for r in records]
        return {
            "min": round(min(vals), 4),
            "median": round(sorted(vals)[len(vals) // 2], 4),
            "mean": round(sum(vals) / len(vals), 4),
            "max": round(max(vals), 4),
        }

    summary = {
        "positive_top1_score": stats(pos_records, "top1_score"),
        "negative_top1_score": stats(neg_records, "top1_score"),
        "positive_content_overlap_ratio": stats(pos_records, "content_overlap_ratio"),
        "negative_content_overlap_ratio": stats(neg_records, "content_overlap_ratio"),
        "positive_score_margin": stats(pos_records, "score_margin"),
        "negative_score_margin": stats(neg_records, "score_margin"),
    }

    print("\n--- STATISTICAL COMPARISON (TRAINING SPLIT) ---")
    print(json.dumps(summary, indent=2))

    Path("experiments/phase4_abstention_investigation.json").write_text(
        json.dumps({"summary": summary, "records": feature_records}, indent=2), encoding="utf-8"
    )
    print("\nSaved detailed investigation to experiments/phase4_abstention_investigation.json")


if __name__ == "__main__":
    main()
