"""Stage B: Training-only abstention investigation, calibration, and cross-validation.

Evaluates candidate abstention strategies strictly on the 159 training queries:
  - 140 answerable training queries
  - 19 unanswerable training queries
Held-out test split is NEVER accessed.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_resolved_paths
from src.evaluation.metrics import compute_query_character_metrics
from src.retrieval.pipeline import ContractRetrievalPipeline

logging.basicConfig(level=logging.WARNING)

STOPWORDS = {
    "a", "an", "the", "in", "on", "of", "to", "for", "with", "at", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "shall", "should", "will", "would",
    "may", "might", "must", "this", "that", "these", "those", "it", "its",
    "what", "which", "who", "whom", "where", "when", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "just",
    "don", "now", "contract", "agreement", "clause", "provision", "consider",
    "between", "and", "or", "there", "their", "under", "party", "parties",
    "pursuant", "hereof", "herein", "hereto", "regarding"
}


def extract_content_words(text: str) -> List[str]:
    """Extract significant content words from query text."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


def get_query_passage_overlap(content_words: List[str], passage_text: str) -> Tuple[float, int, int]:
    """Calculate content word match ratio against passage text."""
    if not content_words:
        return 1.0, 0, 0
    p_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", passage_text.lower()))
    matched = [w for w in content_words if w in p_words]
    ratio = len(matched) / len(content_words)
    return ratio, len(matched), len(content_words)


def run_training_feature_extraction():
    paths = get_resolved_paths()
    train_pos = json.loads(paths["train_queries_output"].read_text(encoding="utf-8"))["tests"]
    all_neg = json.loads(paths["unanswerable_queries_file"].read_text(encoding="utf-8"))
    train_neg = [q for q in all_neg if q.get("split") == "train"]

    print(f"Loaded {len(train_pos)} answerable train queries and {len(train_neg)} unanswerable train queries.")

    pipeline = ContractRetrievalPipeline(config_path="configs/final_phase3.yaml")
    pipeline.build_index()

    # Pre-compute retrieval candidates for all train queries
    pos_data = []
    for q_item in train_pos:
        query = q_item["query"]
        candidates = pipeline.search_candidates(query=query, top_k=5)
        q_info = pipeline.query_processor.process(query)
        focused_q = q_info.get("focused_query", query)
        content_words = extract_content_words(focused_q)

        top1_text = candidates[0]["text"] if candidates else ""
        top1_score = candidates[0]["score"] if candidates else 0.0
        top2_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
        overlap_ratio, matched_cnt, total_cnt = get_query_passage_overlap(content_words, top1_text)

        norm_score = top1_score / max(1, len(content_words))

        # Check gold span metrics without abstention
        snippets = q_item.get("snippets", [])
        metrics = compute_query_character_metrics(snippets, candidates[:5])
        rec5 = metrics["recall"]
        prec5 = metrics["precision"]

        pos_data.append({
            "id": f"POS_{len(pos_data)+1:03d}",
            "query": query,
            "snippets": snippets,
            "focused_query": focused_q,
            "content_words": content_words,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "score_margin": top1_score - top2_score,
            "norm_score": norm_score,
            "overlap_ratio": overlap_ratio,
            "matched_cnt": matched_cnt,
            "total_cnt": total_cnt,
            "candidates": candidates,
            "snippets": snippets,
            "baseline_rec5": rec5,
            "baseline_prec5": prec5,
        })

    neg_data = []
    for q_item in train_neg:
        query = q_item["query"]
        candidates = pipeline.search_candidates(query=query, top_k=5)
        q_info = pipeline.query_processor.process(query)
        focused_q = q_info.get("focused_query", query)
        content_words = extract_content_words(focused_q)

        top1_text = candidates[0]["text"] if candidates else ""
        top1_score = candidates[0]["score"] if candidates else 0.0
        top2_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
        overlap_ratio, matched_cnt, total_cnt = get_query_passage_overlap(content_words, top1_text)

        norm_score = top1_score / max(1, len(content_words))

        neg_data.append({
            "id": q_item["id"],
            "category": q_item.get("category"),
            "query": query,
            "focused_query": focused_q,
            "content_words": content_words,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "score_margin": top1_score - top2_score,
            "norm_score": norm_score,
            "overlap_ratio": overlap_ratio,
            "matched_cnt": matched_cnt,
            "total_cnt": total_cnt,
            "candidates": candidates,
        })

    return pos_data, neg_data


def evaluate_abstention_strategy(
    pos_data: List[Dict[str, Any]],
    neg_data: List[Dict[str, Any]],
    decision_fn,
) -> Dict[str, Any]:
    """Evaluate an abstention decision function on positive and negative query data."""
    # Negative queries: decision_fn should return True (abstain)
    correct_neg_abstentions = 0
    neg_abstention_details = []
    for item in neg_data:
        should_abstain, reason = decision_fn(item)
        if should_abstain:
            correct_neg_abstentions += 1
        neg_abstention_details.append({
            "id": item["id"],
            "category": item["category"],
            "abstained": should_abstain,
            "reason": reason,
        })

    # Positive queries: decision_fn should return False (do not abstain)
    false_pos_abstentions = 0
    rec5_list = []
    prec5_list = []
    for item in pos_data:
        should_abstain, reason = decision_fn(item)
        if should_abstain:
            false_pos_abstentions += 1
            rec5_list.append(0.0)
            prec5_list.append(0.0)
        else:
            rec5_list.append(item["baseline_rec5"])
            prec5_list.append(item["baseline_prec5"])

    mean_rec5 = sum(rec5_list) / len(rec5_list) if rec5_list else 0.0
    mean_prec5 = sum(prec5_list) / len(prec5_list) if prec5_list else 0.0

    return {
        "correct_neg_abstentions": correct_neg_abstentions,
        "total_neg": len(neg_data),
        "neg_abstention_rate": correct_neg_abstentions / max(1, len(neg_data)),
        "false_pos_abstentions": false_pos_abstentions,
        "total_pos": len(pos_data),
        "false_pos_rate": false_pos_abstentions / max(1, len(pos_data)),
        "mean_recall_at_5": mean_rec5,
        "mean_precision_at_5": mean_prec5,
        "neg_details": neg_abstention_details,
    }


def main():
    pos_data, neg_data = run_training_feature_extraction()

    baseline_rec5 = sum(p["baseline_rec5"] for p in pos_data) / len(pos_data)
    baseline_prec5 = sum(p["baseline_prec5"] for p in pos_data) / len(pos_data)
    print(f"Phase 3 Baseline (no abstention): Recall@5 = {baseline_rec5:.4f}, Precision@5 = {baseline_prec5:.4f}")
    min_allowed_rec5 = baseline_rec5 * 0.95
    max_allowed_false_pos = int(0.05 * len(pos_data))  # 7 out of 140
    print(f"Guardrails: Max False Abstentions <= {max_allowed_false_pos} (5%), Min Recall@5 >= {min_allowed_rec5:.4f} (-5% max)")

    # Grid search across interpretable strategies
    candidate_experiments = []

    # Strategy 1: Content word overlap ratio threshold
    # If overlap_ratio < threshold -> abstain
    for min_overlap in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        def make_overlap_rule(th):
            def rule(item):
                # When content words exist, check overlap
                if item["total_cnt"] >= 2 and item["overlap_ratio"] < th:
                    return True, f"Content overlap {item['overlap_ratio']:.2f} < {th:.2f}"
                return False, "Sufficient overlap"
            return rule

        res = evaluate_abstention_strategy(pos_data, neg_data, make_overlap_rule(min_overlap))
        candidate_experiments.append({
            "name": f"content_overlap_th_{min_overlap}",
            "type": "content_word_overlap",
            "params": {"min_overlap": min_overlap},
            "results": res,
        })

    # Strategy 2: Absolute missing content words / minimum matched content words
    # If matched_cnt < min_matched -> abstain
    for min_matched in [1, 2]:
        def make_min_matched_rule(mm):
            def rule(item):
                if item["total_cnt"] >= 3 and item["matched_cnt"] < mm:
                    return True, f"Matched content words {item['matched_cnt']} < {mm}"
                return False, "Sufficient matched content words"
            return rule

        res = evaluate_abstention_strategy(pos_data, neg_data, make_min_matched_rule(min_matched))
        candidate_experiments.append({
            "name": f"min_matched_content_{min_matched}",
            "type": "min_matched_content",
            "params": {"min_matched": min_matched},
            "results": res,
        })

    # Strategy 3: Normalized score threshold (top1_score / num_content_words)
    for min_norm_score in [1.5, 2.0, 2.5, 3.0]:
        def make_norm_score_rule(mns):
            def rule(item):
                if item["norm_score"] < mns:
                    return True, f"Normalized BM25 score {item['norm_score']:.2f} < {mns:.2f}"
                return False, "Sufficient normalized score"
            return rule

        res = evaluate_abstention_strategy(pos_data, neg_data, make_norm_score_rule(min_norm_score))
        candidate_experiments.append({
            "name": f"norm_bm25_score_th_{min_norm_score}",
            "type": "norm_score",
            "params": {"min_norm_score": min_norm_score},
            "results": res,
        })

    # Strategy 4: Hybrid Rule: Content Word Coverage OR (Score Margin & Top Score)
    # A passage must cover at least 25% of query content words AND have top1_score >= threshold
    for min_overlap in [0.20, 0.25]:
        for min_top1 in [8.0, 10.0, 12.0]:
            def make_hybrid_rule(mo, mt):
                def rule(item):
                    if item["total_cnt"] >= 2 and item["overlap_ratio"] < mo:
                        return True, f"Content overlap {item['overlap_ratio']:.2f} < {mo:.2f}"
                    if item["top1_score"] < mt:
                        return True, f"Top-1 BM25 score {item['top1_score']:.2f} < {mt:.2f}"
                    return False, "Sufficient evidence"
                return rule

            res = evaluate_abstention_strategy(pos_data, neg_data, make_hybrid_rule(min_overlap, min_top1))
            candidate_experiments.append({
                "name": f"hybrid_overlap_{min_overlap}_score_{min_top1}",
                "type": "hybrid_overlap_score",
                "params": {"min_overlap": min_overlap, "min_top1": min_top1},
                "results": res,
            })

    # Strategy 5: Combined Normalized BM25 Score and Content Coverage Guardrail
    for min_ns in [2.0, 2.2, 2.4, 2.5]:
        def make_combined_rule(mns):
            def rule(item):
                if item["norm_score"] < mns:
                    return True, f"Normalized BM25 score {item['norm_score']:.2f} < {mns:.2f}"
                if item["total_cnt"] >= 5 and item["matched_cnt"] <= 1:
                    return True, f"Only {item['matched_cnt']} content words matched out of {item['total_cnt']}"
                return False, "Sufficient evidence"
            return rule

        res = evaluate_abstention_strategy(pos_data, neg_data, make_combined_rule(min_ns))
        candidate_experiments.append({
            "name": f"combined_norm_{min_ns}_coverage_guard",
            "type": "combined_norm_coverage",
            "params": {"min_norm_score": min_ns, "min_words_for_guard": 5, "max_matched_for_guard": 1},
            "results": res,
        })

    print("\n--- CANDIDATE EXPERIMENTS SUMMARY ---")
    valid_candidates = []
    for exp in candidate_experiments:
        r = exp["results"]
        passes_false_pos = r["false_pos_abstentions"] <= max_allowed_false_pos
        passes_rec5 = r["mean_recall_at_5"] >= min_allowed_rec5
        improves_neg = r["correct_neg_abstentions"] > 0
        is_valid = passes_false_pos and passes_rec5 and improves_neg
        if is_valid:
            valid_candidates.append(exp)

        status_str = "VALID" if is_valid else "FAIL"
        print(
            f"[{status_str}] {exp['name']}: Correct Negs = {r['correct_neg_abstentions']}/19 "
            f"({r['neg_abstention_rate']:.1%}), False Pos = {r['false_pos_abstentions']}/140 "
            f"({r['false_pos_rate']:.1%}), Recall@5 = {r['mean_recall_at_5']:.4f}"
        )

    # 5-Fold Stratified Cross Validation on top candidate
    print(f"\n--- 5-FOLD CROSS VALIDATION ON VALID CANDIDATES ({len(valid_candidates)}) ---")
    cv_results = {}
    import random
    rng = random.Random(42)

    # Create 5 folds within training data
    pos_shuffled = list(pos_data)
    neg_shuffled = list(neg_data)
    rng.shuffle(pos_shuffled)
    rng.shuffle(neg_shuffled)

    folds = []
    n_folds = 5
    for f in range(n_folds):
        pos_val = [pos_shuffled[i] for i in range(len(pos_shuffled)) if i % n_folds == f]
        pos_tr = [pos_shuffled[i] for i in range(len(pos_shuffled)) if i % n_folds != f]
        neg_val = [neg_shuffled[i] for i in range(len(neg_shuffled)) if i % n_folds == f]
        neg_tr = [neg_shuffled[i] for i in range(len(neg_shuffled)) if i % n_folds != f]
        folds.append((pos_tr, neg_tr, pos_val, neg_val))

    for exp in valid_candidates:
        name = exp["name"]
        fold_metrics = []
        for f, (p_tr, n_tr, p_val, n_val) in enumerate(folds):
            # Evaluate on out-of-fold validation
            if exp["type"] == "content_word_overlap":
                rule_fn = make_overlap_rule(exp["params"]["min_overlap"])
            elif exp["type"] == "min_matched_content":
                rule_fn = make_min_matched_rule(exp["params"]["min_matched"])
            elif exp["type"] == "content_coverage_guardrail":
                rule_fn = rule_entity_overlap
            elif exp["type"] == "combined_norm_coverage":
                rule_fn = make_combined_rule(exp["params"]["min_norm_score"])
            else:
                rule_fn = make_norm_score_rule(exp["params"]["min_norm_score"])

            eval_res = evaluate_abstention_strategy(p_val, n_val, rule_fn)
            fold_metrics.append({
                "fold": f + 1,
                "val_correct_negs": eval_res["correct_neg_abstentions"],
                "val_total_negs": eval_res["total_neg"],
                "val_false_pos": eval_res["false_pos_abstentions"],
                "val_total_pos": eval_res["total_pos"],
                "val_recall_at_5": eval_res["mean_recall_at_5"],
            })

        avg_neg_rate = sum(fm["val_correct_negs"] / fm["val_total_negs"] for fm in fold_metrics) / n_folds
        avg_false_pos_rate = sum(fm["val_false_pos"] / fm["val_total_pos"] for fm in fold_metrics) / n_folds
        avg_rec5 = sum(fm["val_recall_at_5"] for fm in fold_metrics) / n_folds

        cv_results[name] = {
            "folds": fold_metrics,
            "oof_neg_abstention_rate": avg_neg_rate,
            "oof_false_pos_rate": avg_false_pos_rate,
            "oof_recall_at_5": avg_rec5,
        }
        print(f"CV {name}: OOF Neg Abstention Rate = {avg_neg_rate:.1%}, OOF False Pos Rate = {avg_false_pos_rate:.1%}, OOF Recall@5 = {avg_rec5:.4f}")

    # Save experiments
    Path("experiments/phase4_abstention_experiments.json").write_text(
        json.dumps(candidate_experiments, indent=2), encoding="utf-8"
    )
    Path("experiments/phase4_train_validation.json").write_text(
        json.dumps(cv_results, indent=2), encoding="utf-8"
    )
    print("\nSaved experiments to experiments/phase4_abstention_experiments.json and experiments/phase4_train_validation.json")


if __name__ == "__main__":
    main()
