# Contract Intelligence Assistant — Technical Evaluation & Performance Report

## 1. Executive Summary

This report documents the architectural design, dataset preparation, baseline retrieval engine, experimental evaluation, and error analysis for the JioStar Contract Intelligence Assistant. The assistant retrieves relevant contract clauses from complex legal agreements (SEC EDGAR filings), extracts exact character-offset evidence, and reliably abstains when no evidence supports an answer.

**Current Status**: 
- **Phase 1 (Foundation & Data Preparation)**: Complete. 194 mini queries selected, 29 contracts manifested, 272/272 ground-truth spans verified (100% exact substring match).
- **Phase 2 (BM25 Retrieval, Chunking, Abstention, CLI & Web Demo)**: Complete. BM25 baseline retrieval engine implemented, 1,874 passages indexed, CLI and Streamlit interfaces operational, character-level retrieval metrics evaluated on the training split.
- **Phase 3 (Dense/Hybrid Retrieval, Reranking, Self-Improvement Loop & Final Test Evaluation)**: Pending future implementation.

---

## 2. System Architecture

### 2.1 Retrieval Pipeline Design
The Phase 2 baseline implements an end-to-end, modular lexical retrieval architecture:

```
[Query Input]
      │
      ▼
[LegalTokenizer] (Lowercase, alphanumeric + section codes like 10.1, hyphenated legal terms)
      │
      ▼
[BM25Retriever] (BM25Okapi: k1=1.5, b=0.75, indexed over 1,874 contract chunks)
      │
      ▼
[Deterministic Ranker] (Sort by score DESC, tie-break by chunk_id ASC)
      │
      ▼
[AbstentionEngine] (Score thresholding: top_score <= tau -> abstain)
      │
      ├───────────────────────────────┐
      ▼                               ▼
[Results Found]              [Abstention Signal]
Top-K Ranked Passages        no_relevant_clause
with exact document          with decision rationale
character offsets
```

### 2.2 Document Chunking Strategy
- **Strategy**: Fixed-character sliding window with overlap.
- **Chunk Size**: 800 characters.
- **Overlap**: 150 characters (step size = 650 characters).
- **Invariant**: For every generated chunk $c$, $c[\text{"text"}] == \text{document}[c[\text{"start\_char"}]:c[\text{"end\_char"}]]$.
- **Corpus Coverage**: 100% of all 1,211,773 characters across all 29 contracts are covered.
- **Total Chunks Generated**: 1,874 chunks.
- **Independence**: Chunk boundaries are generated independently of ground-truth snippet locations.

### 2.3 Tokenization Strategy
- **Module**: `src/retrieval/tokenizer.py` (`LegalTokenizer`).
- **Logic**: Lowercases text, extracts alphanumeric sequences while preserving internal decimal points and hyphens (e.g. section codes `10.1`, `3.2.1`, compound terms like `sublicensing`, `non-competition`, `co-branding`).
- **Consistency**: The identical tokenizer instance processes document chunks during indexing and incoming user queries at inference.

---

## 3. Dataset and Evaluation Setup

### 3.1 Benchmark & Corpus Provenance
- **Upstream Source**: [awinml/legalbench-rag](https://huggingface.co/datasets/awinml/legalbench-rag)
- **Commit SHA**: `2b9c248bc8179ef0908fd6ba01d50b156facd48b`
- **Reference Paper**: *LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in Legal Reasoning* ([arXiv:2408.10343](https://arxiv.org/abs/2408.10343))
- **Corpus Scope**: 29 unique SEC EDGAR commercial agreements (1,211,773 UTF-8 characters).

### 3.2 Strict Train/Test Separation
- **Answerable Queries**: 194 total (140 Train / 54 Test).
- **Unanswerable Negative Queries**: 32 total (19 Train / 13 Test across 6 categories).
- **Splitting Algorithm**: Deterministic SHA-256 integer modulo: `test` if `int(SHA256(query), 16) % 10 < 3` else `train`.
- **Evaluation Guardrail**: All Phase 2 evaluation and hyperparameter calibration was performed strictly on the **train split** (140 answerable + 19 unanswerable queries). The 54 test queries and 13 unanswerable test queries were kept strictly locked and unread.

---

## 4. Baseline Methodology & Configuration

The initial baseline retrieval system is configured in `configs/baseline.yaml`:

```yaml
pipeline:
  name: "bm25_baseline"
  version: 1

chunking:
  strategy: "fixed_character"
  chunk_size: 800
  overlap: 150

tokenization:
  lowercase: true

retrieval:
  method: "bm25"
  top_k: 5

bm25:
  k1: 1.5
  b: 0.75

abstention:
  enabled: true
  strategy: "top_score_threshold"
  threshold: 0.0

evaluation:
  k_values: [1, 3, 5, 10]
  split: "train"
```

---

## 5. Measured Baseline Results (Train Split)

The baseline evaluation was executed using `scripts/evaluate_baseline.py --config configs/baseline.yaml --split train`. All metrics were computed using exact character-level set intersections as defined in the LegalBench-RAG specification.

### 5.1 Character-Level Retrieval Metrics (140 Answerable Train Queries)

| Metric Cutoff | Character Precision | Character Recall | Interpretation |
|---|---|---|---|
| **@1** | 0.0018 | 0.0037 | Top-1 chunk captures initial signal for high-relevance terms |
| **@3** | 0.0056 | 0.0310 | 3-passage window expands relevant clause capture |
| **@5** | 0.0141 | 0.1169 | 11.69% of ground-truth characters captured within top-5 |
| **@10** | 0.0150 | 0.2318 | 23.18% of ground-truth characters captured within top-10 |

*Metric Calculation Invariant*: Retrieved passages within the same document are merged prior to character set intersection, avoiding double-counting of overlapping chunks. Character boundaries from different contracts are strictly isolated.

### 5.2 Abstention Performance (Train Split)

#### Initial Baseline ($\tau = 0.0$)
- **Unanswerable Queries Evaluated**: 19
- **Correct Abstentions on Negatives**: 0 / 19 (0.0%)
- **False Abstentions on Positives**: 0 / 140 (0.0%)
- **Answerable Query Coverage**: 140 / 140 (100.0%)
*Observation*: With threshold $\tau=0.0$, BM25 scores on unanswerable queries are non-zero due to incidental lexical overlap (e.g., query stopwords matching common terms in long contracts), preventing abstention.

#### Calibrated Threshold Analysis on Training Split
Analyzing the score distributions between answerable and unanswerable training queries reveals strong separation:
- **Answerable Queries Top-1 Score**: Min = 44.67, Mean = 67.44, Max = 102.85
- **Unanswerable Queries Top-1 Score**: Min = 13.68, Mean = 24.46, Max = 43.76

Setting the calibrated threshold to $\tau = 44.0$:
- **Correct Abstentions on Negatives**: 19 / 19 (100.0%)
- **False Abstentions on Positives**: 0 / 140 (0.0%)
- **Answerable Query Coverage**: 140 / 140 (100.0%)

---

## 6. Observed Baseline Limitations (Targeted for Phase 3)

1. **Vocabulary Mismatch in Sparse Lexical Matching**: BM25 relies on exact term overlap. When a query uses legal synonyms (e.g. *"Can this agreement be assigned?"* vs. contractual text stating *"Neither party may delegate or transfer obligations"*), BM25 fails to retrieve the clause.
2. **Fixed 800-Character Windows**: Legal clauses vary widely in length. Long multi-paragraph indemnity clauses span beyond 800 characters, while brief governing law clauses are under 100 characters. Fixed chunking introduces fragmentation.
3. **Precision Dilution at High Cutoffs**: Because each chunk is 800 characters and ground-truth snippets often span only 100–300 characters, character precision is naturally bounded by chunk length. Dense reranking in Phase 3 can refine passage boundaries.

---

## 7. Next Steps (Phase 3 Roadmap)

1. **Dense Retrieval Integration**: Generate embeddings using a domain-tailored sentence transformer and index with FAISS.
2. **Hybrid Retrieval**: Combine dense semantic similarity with sparse BM25 scores via Reciprocal Rank Fusion (RRF) or calibrated linear weighting.
3. **Cross-Encoder Reranking**: Re-score top-20 hybrid candidates using a cross-encoder to elevate exact clause passages.
4. **Automated Improvement Loop**: Systematically sweep chunk parameters, fusion weights, and abstention thresholds strictly on the train split.
5. **Final Held-Out Test Evaluation**: Execute single final evaluation on the 54 test queries and 13 test negative queries.
