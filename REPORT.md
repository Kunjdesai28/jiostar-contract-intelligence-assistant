# Contract Intelligence Assistant — Technical Evaluation & Performance Report

## 1. Executive Summary

This report documents the architectural design, dataset preparation, baseline methodology, experimental evaluation, and error analysis for the JioStar Contract Intelligence Assistant. The assistant is designed to retrieve contract clauses from complex legal agreements, extract exact character-offset evidence, and reliably abstain when no evidence supports an answer.

**Current Status**: Phase 1 Foundation and Dataset Preparation is complete. All ground-truth annotations (272 snippets across 29 unique agreements) have been verified with 100% character-level precision. Baseline retrieval and downstream evaluation metrics are scheduled for Phase 2 and Phase 3.

---

## 2. System Design

### 2.1 High-Level Architecture (Planned)
The system is designed around a three-tier retrieval and evidence verification architecture:
1. **Document Ingestion & Chunking**: Boundaries defined with character offset tracking relative to original document roots.
2. **Hybrid Retrieval**: Sparse lexical retrieval (BM25) complemented by dense semantic retrieval and cross-encoder reranking.
3. **Evidence Verification & Abstention Engine**: Calibrated confidence thresholding to return exact character offsets or emit an abstention signal when confidence falls below the reliability threshold.

### 2.2 Phase 1 Architecture Implemented
```
[Hugging Face awinml/legalbench-rag]
                 │
                 ▼
      [data/raw/benchmarks/cuad.json] (4,042 queries)
      [data/raw/corpus/cuad/]         (462 contracts)
                 │
                 ▼ (mini_selector.py)
      [data/processed/cuad_mini.json] (194 queries, SHA-256: 412fc88d...)
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
[span_validator.py]   [splitter.py]
 (272/272 Valid)       (140 Train / 54 Test)
      │                     │
      ▼                     ▼
[loader.py]           [data/processed/train_queries.json]
 (29 Contracts,       [data/processed/test_queries.json]
  1,211,773 chars)          │
                            ▼
                    [negative_queries.py]
                     (32 Unanswerable Queries: 19 Train / 13 Test)
```

---

## 3. Dataset and Evaluation Setup

### 3.1 Benchmark Selection: LegalBench-RAG CUAD Mini
- **Source**: `https://huggingface.co/datasets/awinml/legalbench-rag`
- **Revision / Commit SHA**: `2b9c248bc8179ef0908fd6ba01d50b156facd48b`
- **Reference Paper**: *LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in Legal Reasoning* ([arXiv:2408.10343](https://arxiv.org/abs/2408.10343))
- **Selection Function**: Deterministic PRNG seeded with each record's first snippet `file_path`:
  - Total queries selected: **194**
  - All 194 queries are unique.
  - Serialization SHA-256: `412fc88d91b8299defa7158a5bd2ee008e288af0332a5dfd39b8d44a83a61749`

### 3.2 Corpus Discovery & Scope
Only contracts referenced by the 194 mini benchmark queries are indexed into the retrieval corpus:
- **Total Unique Contracts**: **29**
- **Total Document Characters**: **1,211,773** characters (UTF-8)
- **Manifest File**: `data/processed/contract_manifest.json` (SHA-256: `59fd376b2fb69406ec1e626a224398a2255a7fe307ff1b006afd83c0a7b9c053`)

### 3.3 Ground-Truth Span Validation Results
Every ground-truth snippet in the mini benchmark was subjected to strict character-level offset verification (`document[start:end] == expected_answer`):
- **Total Benchmark Queries**: 194
- **Total Annotated Snippets**: 272
- **Valid Spans (Exact Match)**: 272 (100.0%)
- **Invalid Spans**: 0
- **Missing Documents**: 0
- **Validation Summary Artifact**: `data/processed/span_validation_summary.json`

### 3.4 Honest Train/Test Split
Split assignments are determined strictly via SHA-256 hash integer thresholding on whitespace-stripped query strings:
$$\text{split} = \begin{cases} \text{"test"} & \text{if } \text{int}(\text{SHA256}(\text{query}), 16) \pmod{10} < 3 \\ \text{"train"} & \text{otherwise} \end{cases}$$

- **Train Split**: 140 queries (72.2%)
- **Test Split**: 54 queries (27.8%)
- **Overlap**: 0 queries (strictly disjoint)

*Evaluation Protocol Guardrail*: The 54 test queries are strictly isolated and will only be evaluated at final project completion. No model selection or hyperparameter tuning in Phase 2 or 3 will access test split queries or ground-truth snippets.

### 3.5 Unanswerable Query Benchmark
To evaluate model abstention capabilities, 32 manually reasoned negative queries were created across 6 distinct categories:
- **Outside Indexed Corpus**: 6 queries (e.g. JioStar IPL broadcasting, Netflix streaming licenses)
- **Absent From Contracts**: 6 queries (e.g. historical post-signing revenue, actual delivered volume)
- **Future Events Not Established**: 5 queries (e.g. subsequent bankruptcy filings, renewal exercise)
- **External Business Data**: 4 queries (e.g. NASDAQ stock prices, Moody's credit ratings)
- **Unsupported Legal Facts**: 6 queries (e.g. GDPR penalties in pre-2016 contracts, HIPAA damages)
- **Out of Domain**: 5 queries (e.g. Amazon rainfall, FIFA World Cup, culinary recipes)
- **Split Distribution**: 19 Train / 13 Test (assigned using the exact `assign_split()` function)
- **Dataset File**: `data/unanswerable_queries.json` (SHA-256: `7eaf304f4a22a4567100fb283a53f1bfc7b458eaa2f7feb7d78ba66db3d539d3`)

---

## 4. Baseline Methodology
*Pending Phase 2 implementation and measured results.*

Planned Baseline:
- BM25 Okapi baseline implementation without LLM overhead.
- Sentence/paragraph chunking with sliding windows preserving document-level character offset pointers.
- Heuristic score thresholding for abstention.

---

## 5. Retrieval Metrics
*Pending Phase 2 implementation and measured results.*

Planned Evaluation Metrics:
- Exact Span Overlap / Intersection over Union (IoU)
- Document-level Recall@K (K=1, 3, 5)
- Character-level Precision and Recall
- Mean Reciprocal Rank (MRR)

---

## 6. Abstention Metrics
*Pending Phase 2 implementation and measured results.*

Planned Metrics:
- Abstention Precision: $\frac{\text{True Abstentions}}{\text{Total Abstention Decisions}}$
- Abstention Recall: $\frac{\text{True Abstentions}}{\text{Total Unanswerable Queries}}$
- False Abstention Rate: Rate of abstaining on truly answerable queries.
- Hallucination / Over-retrieval Rate on negative queries.

---

## 7. Improvement-Loop Objective
*Pending Phase 3 implementation and measured results.*

Target Objective:
- Automated iterative improvement loop optimizing chunk size, chunk overlap, BM25 hyper-parameters ($k_1, b$), dense retriever embedding weight, and abstention threshold strictly on the train split.

---

## 8. Experiment History
*Pending Phase 3 implementation and measured results.*

| Experiment ID | Description | Train Recall@5 | Train Span IoU | Train Abstention F1 | Status |
|---|---|---|---|---|---|
| *EXP_001* | *Phase 2 Baseline* | *Pending* | *Pending* | *Pending* | *Pending* |

---

## 9. Baseline versus Final Results
*Pending Phase 3 implementation and measured results.*

---

## 10. Error Analysis
*Pending Phase 2/3 implementation and measured results.*

---

## 11. Limitations
1. **Contract Domain Specificity**: The 29 indexed contracts originate from US SEC EDGAR filings and may not fully generalize to civil law or international cross-border agreements without domain adaptation.
2. **Fixed Mini Corpus Scope**: Evaluating retrieval against a 29-contract subset reduces the search space compared to the full 462-contract CUAD corpus; however, this mirrors standard LegalBench-RAG benchmarking requirements.

---

## 12. Next Steps (Phase 2 Roadmap)
1. **Document Chunking Engine**: Implement paragraph and sliding-window chunkers in `src/retrieval/chunking.py` preserving character offsets $[start, end]$.
2. **BM25 Lexical Indexer**: Implement baseline search in `src/retrieval/bm25.py`.
3. **Passage Retrieval Interface**: Return ranked results with exact document offsets.
4. **Abstention Evaluator**: Calibrate abstention score thresholds on `train_queries` and `unanswerable_queries` train split.
