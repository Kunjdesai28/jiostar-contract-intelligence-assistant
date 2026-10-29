# PHASE 4 — FINAL CONFIGURATION FREEZE MANIFEST

**Freeze Status**: **LOCKED AND PERMANENTLY IMMUTABLE**  
**Freeze Timestamp**: `2026-10-29T23:27:00+05:30`  
**Held-Out Test Evaluation Status**: **NOT YET EXECUTED**

---

## 1. Machine-Readable Cryptographic Manifest

```json
{
  "freeze_status": "LOCKED",
  "freeze_timestamp_utc": "2026-10-29T17:57:00Z",
  "freeze_timestamp_local": "2026-10-29T23:27:00+05:30",
  "git_branch": "main",
  "base_commit_sha": "06493a0",
  "configuration_path": "configs/final.yaml",
  "configuration_sha256": "6448998c99d0c236dd0644dd3abd6c62b679a0749187b3ce8f14ff210caa7718",
  "dataset_manifest_path": "data/processed/contract_manifest.json",
  "frozen_negative_query_checksum": "7eaf304f4a22a4567100fb283a53f1bfc7b458eaa2f7feb7d78ba66db3d539d3",
  "held_out_test_status": "NOT_YET_EXECUTED",
  "pipeline": {
    "name": "candidate_final_phase4",
    "version": 2,
    "chunking": {
      "strategy": "clause_aware",
      "chunk_size": 1000,
      "overlap": 250
    },
    "tokenization": {
      "lowercase": true,
      "enhancement_mode": "legal_stem"
    },
    "retrieval": {
      "method": "bm25",
      "top_k": 5,
      "k1": 1.5,
      "b": 0.75
    },
    "query_processing": {
      "mode": "two_stage"
    },
    "abstention": {
      "enabled": true,
      "strategy": "normalized_score_coverage",
      "threshold": 0.0,
      "min_norm_score": 2.4,
      "min_words_for_guard": 5,
      "max_matched_for_guard": 1
    }
  },
  "training_selection_metrics": {
    "answerable_queries": 140,
    "unanswerable_queries": 19,
    "precision_at_1": 0.0718,
    "recall_at_1": 0.1385,
    "precision_at_3": 0.0544,
    "recall_at_3": 0.2670,
    "precision_at_5": 0.0365,
    "recall_at_5": 0.3096,
    "precision_at_10": 0.0250,
    "recall_at_10": 0.4208,
    "correct_negative_abstentions": "10/19 (52.6%)",
    "false_positive_abstentions": "5/140 (3.6%)",
    "answerable_coverage": "135/140 (96.4%)",
    "five_fold_cv_oof_negative_rate": "51.7%",
    "five_fold_cv_oof_false_positive_rate": "3.6%",
    "mean_latency_ms": 27.81
  }
}
```

---

## 2. Selection Rationale and Scientific Defense

### Defect Identified in Phase 3
Phase 3 achieved strong positive retrieval Recall@5 (0.3191) with clause-aware chunking and two-stage query processing, but had a critical abstention defect:
- Correct negative abstentions: **0/19 (0.0%)**
- The system returned spurious passages for all 19 unanswerable training questions because raw BM25 scores scale monotonically with query length. A 12-word unanswerable question naturally amasses raw BM25 scores of 20–43 by accumulating partial matches on generic legal boilerplate ("agreement", "contract", "parties", "hereunder"), whereas a 2-word answerable query scored only 7–12.

### Selected Abstention Strategy: Normalized Score & Evidence Coverage Guardrail
1. **Query-Length Normalized BM25 Relevance**:
   Evaluates $\text{score}_{\text{norm}} = \frac{\text{score}_{\text{top1}}}{\max(1, |\text{content\_words}|)}$.
   If $\text{score}_{\text{norm}} < 2.40$, the query term density in the top-ranked passage is insufficient to support an affirmative answer.
2. **Evidence Coverage Guardrail**:
   When a question specifies $\ge 5$ distinct content terms (e.g. out-of-domain questions about Amazon rainfall, Neapolitan pizza, or entities outside the corpus), the system abstains if $\le 1$ content word appears in the top retrieved passage.

### Compliance with Frozen Guardrails
- **Negative Abstention Improvement**: Improved from **0/19 (0.0%)** to **10/19 (52.6%)** (out-of-fold cross-validation: 51.7%).
- **False Abstention Limit**: 5/140 (**3.6%**), well within the pre-declared $\le 5.0\%$ ceiling (7 queries).
- **Recall@5 Preservation**: Recall@5 dropped by only 2.97% (from 0.3191 to 0.3096), strictly respecting the maximum allowable 5% relative reduction (floor: 0.3031).
- **Inference Latency**: 27.81 ms mean latency, virtually identical to Phase 3 (28 ms) and well within the 3x guardrail.

---

## 3. Strict Pre-Test Invariant Declaration

1. **Held-out test split has NEVER been accessed, evaluated, or peeked at.**
2. The 54 answerable test queries and 13 unanswerable test queries remain locked.
3. Once Stage D is initiated, `scripts/final_evaluation.py` will run **exactly once**.
4. No hyperparameter adjustments, threshold tuning, or model selection will take place after the single test evaluation.
