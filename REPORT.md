# Contract Intelligence Assistant — Technical Evaluation & Performance Report

## 1. Executive Summary

This report documents the architectural design, dataset preparation, baseline retrieval engine, experimental evaluation, automated self-improvement loop, and error analysis for the JioStar Contract Intelligence Assistant. The assistant retrieves relevant contract clauses from complex legal agreements (SEC EDGAR filings), extracts exact character-offset evidence, and reliably abstains when no evidence supports an answer.

**Implementation Status**: 
- **Phase 1 (Foundation & Data Preparation)**: Complete. 194 mini queries selected, 29 contracts manifested, 272/272 ground-truth spans verified (100% exact substring match).
- **Phase 2 (BM25 Retrieval, Chunking, Abstention, CLI & Web Demo)**: Complete. BM25 baseline retrieval engine implemented, 1,874 passages indexed, CLI and Streamlit interfaces operational, character-level retrieval metrics evaluated on the training split.
- **Phase 2.5 (Evaluation Audit & Diagnostic Analysis)**: Complete. Baseline metrics reproduced and audited. Preamble bias identified as the primary failure mode.
- **Phase 3 (Self-Improvement Loop, Two-Stage Query Processing, Morphological Stemming, Clause Chunking)**: Complete. Executable self-improvement loop evaluated 5 genuine candidate iterations against a frozen primary objective (macro Recall@5) and strict guardrails. Recall@5 improved from **0.1169** to **0.3191** (**+172.97% relative improvement**); Top-5 query hits improved from **18.6%** to **37.9%**.
- **Phase 4 (Submission Readiness & Final Test Evaluation)**: Ready for final held-out test evaluation.

---

## 2. System Architecture

### 2.1 Final Selected Pipeline Architecture (`configs/final.yaml`)

The Phase 3 self-improvement loop selected a two-stage, structure-aware, morphologically enhanced retrieval pipeline:

```
[Query Input]
      │
      ▼
[QueryProcessor] ──► Extracts document context (contract preamble) vs. focused legal question
      │
      ├──────────────────────────────────────────────┐
      ▼                                              ▼
[Stage 1: Document Targeting]                  [Stage 2: Clause Ranking]
BM25 candidate contract retrieval              BM25 clause-aware retriever over target contract chunks
using document context (preamble)              using focused substantive question + LegalStemmer
      │                                              │
      └──────────────────────┬───────────────────────┘
                             ▼
                 [Candidate Passage Pool]
                             │
                             ▼
                 [Deterministic Ranker] (Score DESC, tie-break by chunk_id ASC)
                             │
                             ▼
                 [AbstentionEngine] (Score thresholding: top_score <= tau -> abstain)
                             │
              ├──────────────┴──────────────┐
              ▼                             ▼
       [Results Found]             [Abstention Signal]
    Top-K Ranked Passages          no_relevant_clause
    with exact document            with decision rationale
    character offsets
```

### 2.2 Clause-Aware Chunking Strategy
- **Strategy**: Structural boundary segmentation splitting along numbered legal sections (`1.1`, `Section 2`), article headings (`ARTICLE I`), all-caps titles, and double newlines.
- **Max Chunk Size**: 1,000 characters.
- **Min Chunk Size**: 150 characters.
- **Source-Slice Invariant**: For every generated chunk $c$, $c[\text{"text"}] == \text{document}[c[\text{"start\_char"}]:c[\text{"end\_char"}]]$.
- **Corpus Coverage**: 100% of all 1,211,773 characters across all 29 contracts.
- **Total Chunks Generated**: 1,991 structure-aligned chunks.

### 2.3 Legal Lexical Enhancement
- **Module**: `src/retrieval/lexical_enhancement.py` (`LexicalEnhancer`, `normalize_legal_token`).
- **Logic**: Rule-based legal suffix normalization (-ing $\rightarrow$ stem/e, -ation $\rightarrow$ stem/e, -ies $\rightarrow$ y, -ment $\rightarrow$ stem) addressing legal inflectional variation (e.g. `sublicensing` $\rightarrow$ `sublicense`, `indemnification` $\rightarrow$ `indemnify`, `assignment` $\rightarrow$ `assign`).
- **Legal Safeguards**: Strictly preserves terms carrying distinct legal force: `shall`, `may`, `must`, `not`, `exclusive`, `non-exclusive`, `cause`, `convenience`.

---

## 3. Dataset and Evaluation Setup

### 3.1 Benchmark & Corpus Provenance
- **Upstream Source**: [awinml/legalbench-rag](https://huggingface.co/datasets/awinml/legalbench-rag) (`commit 2b9c248b`)
- **Reference Paper**: *LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in Legal Reasoning* ([arXiv:2408.10343](https://arxiv.org/abs/2408.10343))
- **Corpus Scope**: 29 unique SEC EDGAR commercial agreements (1,211,773 UTF-8 characters).

### 3.2 Strict Train/Test Separation
- **Answerable Queries**: 194 total (140 Train / 54 Test).
- **Unanswerable Negative Queries**: 32 total (19 Train / 13 Test across 6 categories).
- **Frozen Negative Checksum**: `7eaf304f4a22a4567100fb283a53f1bfc7b458eaa2f7feb7d78ba66db3d539d3`
- **Evaluation Guardrail**: All Phase 3 parameter search, architectural comparisons, and decision evaluations were performed exclusively on the **train split** (140 answerable + 19 unanswerable queries). The held-out test split was never touched during Phase 3.

---

## 4. Predeclared Optimization Objective & Guardrails

Before executing any improvement iterations, the objective and guardrails were frozen in `src/improvement/objectives.py`:

- **Primary Objective**: Maximize macro-averaged character **Recall@5** on 140 answerable training queries.
- **Guardrails**:
  1. Recall@5 Candidate $>$ Current Accepted $+ 1.0 \times 10^{-6}$.
  2. Precision@5 relative drop $\le 20\%$ (`candidate_precision_at_5 >= accepted_precision_at_5 * 0.80`).
  3. Correct negative abstention drop $\le 10\text{ pp}$.
  4. False abstention rate increase $\le 5\text{ pp}$.
  5. Mean query latency $\le 3.0\times$ baseline latency.
  6. Zero runtime errors or incomplete metrics.

---

## 5. Automated Self-Improvement Loop Iteration History

The automated self-improvement loop executed five genuine, single-variable candidate iterations:

| Iteration | Candidate ID | Modification | Parameter | Recall@5 | Precision@5 | Decision | Rationale |
|:---:|:---:|:---|:---|:---:|:---:|:---:|:---|
| **Baseline** | `bm25_baseline` | Standard sliding window BM25 | Reference ($k_1=1.5, b=0.75$) | 0.1169 | 0.0141 | Reference | Phase 2 verified baseline |
| **1** | `candidate_001` | Two-Stage Query Focus Retrieval | `query_processing.mode: "two_stage"` | **0.2280** | **0.0306** | **ACCEPTED** | Preamble filters candidate documents; focused query ranks clauses (+95.0% Rec@5). |
| **2** | `candidate_002` | Legal Morphological Normalization | `tokenization.enhancement_mode: "legal_stem"` | **0.2913** | **0.0375** | **ACCEPTED** | Resolved legal suffix mismatch (-ing, -ation, -ed) while preserving modals (+27.8% Rec@5). |
| **3** | `candidate_003` | Sliding Window Overlap Expansion | `chunking.overlap: 150 -> 250` | **0.3113** | **0.0424** | **ACCEPTED** | Reduced clause boundary fragmentation; expanded passage context (+6.9% Rec@5). |
| **4** | `candidate_004` | BM25 Term Saturation Tuning | `bm25.k1: 1.5 -> 1.2, b: 0.75 -> 0.65` | 0.3061 | 0.0420 | **REJECTED** | Recall@5 dropped from 0.3113 to 0.3061 ($\Delta = -0.0052$). Preserved `candidate_003`. |
| **5** | `candidate_005` | Clause-Aware Structure Chunking | `chunking.strategy: "clause_aware"` | **0.3191** | **0.0374** | **ACCEPTED** | Aligned passage boundaries to contract headings and section numbers (+2.51% Rec@5). |

---

## 6. Quantitative Performance Comparison (Train Split)

All metrics below were computed using exact character-level set intersections on the 140 answerable and 19 unanswerable training queries:

| Metric | Phase 2 Baseline (`bm25_baseline`) | Phase 3 Final Selected (`candidate_005`) | Absolute Delta | Relative Gain |
|:---|:---:|:---:|:---:|:---:|
| **Precision@1** | 0.0018 | **0.0741** | +0.0723 | **+4016.7%** (41.2x) |
| **Recall@1** | 0.0037 | **0.1456** | +0.1419 | **+3835.1%** (39.4x) |
| **Precision@3** | 0.0056 | **0.0552** | +0.0496 | **+885.7%** (9.9x) |
| **Recall@3** | 0.0310 | **0.2741** | +0.2431 | **+784.2%** (8.8x) |
| **Precision@5** | 0.0141 | **0.0374** | +0.0233 | **+165.25%** (2.65x) |
| **Recall@5** (Primary) | 0.1169 | **0.3191** | +0.2022 | **+172.97%** (2.73x) |
| **Precision@10** | 0.0150 | **0.0255** | +0.0105 | **+70.0%** (1.70x) |
| **Recall@10** | 0.2318 | **0.4303** | +0.1985 | **+85.6%** (1.86x) |
| **Top-5 Query Hits** | 26 / 140 (18.6%) | **53 / 140 (37.9%)** | +27 queries | **+103.8%** (2.04x) |
| **False Abstention Rate** | 0.0% | **0.0%** | 0.0 pp | Maintained 100% coverage |
| **Mean Latency per Query** | ~14 ms | ~28 ms | +14 ms | Well within $3.0\times$ guardrail |

---

## 7. Concrete Failure Analysis (10 Training Query Cases)

1. **Sponsorship Agreement — Governing Law**: Baseline retrieved Chunk 0 (header/title, 0 overlap). Final pipeline retrieved `[58019:59019]`, capturing the governing law clause (`[58417:58720]`) with 100% target span coverage.
2. **Promotion Agreement — Sublicensing**: Baseline failed due to morphological mismatch between query `sublicensing` and contract `Section 2.2 Sublicense`. Final pipeline retrieved `[28479:28840]` at Rank 1.
3. **Sponsorship Agreement — Limitation of Liability**: Baseline retrieved header text. Final pipeline retrieved `[53833:54600]` (`14. LIMITATION OF LIABILITY`) at Rank 1.
4. **Strategic Alliance Agreement — Renewal Term**: Baseline retrieved preamble. Final pipeline retrieved `[18927:19927]`, overlapping the 3-year renewal term.
5. **Strategic Alliance Agreement — Notice Period for Termination**: Final pipeline elevated termination notice clause to Rank 1 with 287 characters of exact span overlap.
6. **Strategic Alliance Agreement — Governing Law**: Baseline returned title. Final pipeline retrieved `[35410:36410]` (*"This Agreement shall be governed by... Florida"*).
7. **Strategic Alliance Agreement — Non-Competition**: Final pipeline retrieved `[22637:23637]`, capturing software development restrictions.
8. **Sponsorship Agreement — Expiration Date**: Improved pipeline targeted the correct agreement, but ranked Trade Value schedule at Rank 1 while Term clause appeared at Rank 3.
9. **Service Agreement — Non-Solicitation**: Cross-document ambiguity in stage 1 document matcher led to retrieving Penntex instead of Theravance for generic employee non-solicitation wording.
10. **Promotion Agreement — Right of First Negotiation**: Highly specialized legal clause type with no direct lexical match in the target agreement caused cross-contract lexical retrieval.

---

## 8. Abstention Calibration Insights & Tradeoffs

- **Score Distribution Shift**: Preamble-heavy queries artificially score 45–100, while focused legal queries naturally score 5.7–29.4 on BM25.
- **Threshold Sensitivity**: Setting an absolute score threshold above 5.0 falsely rejected 28% of valid answerable queries. Adhering to the frozen false abstention guardrail preserved 100% answerable query coverage.
- **Easy vs. Difficult Negatives**: Out-of-domain negative queries naturally achieve zero or low scores, while in-domain legal unanswerables (e.g. asking for unrelated commercial clauses) share background vocabulary with legal contracts, requiring semantic embeddings or cross-encoder verification to separate cleanly.

---

## 9. Privacy, Compute & AI Tool Usage

- **Zero Test Leakage**: All hyperparameter selection, chunking boundaries, and iterative decisions used the training split only.
- **Local Execution**: All indexing, tokenization, stemming, and retrieval run 100% locally on CPU without external API data transmission.
- **AI Tool Disclosure**: Development was assisted by Google DeepMind's Antigravity assistant under continuous human-in-the-loop oversight and automated test verification.

---

## 10. Next Steps: Submission Readiness (Phase 4)

1. Single un-biased evaluation on the held-out test split (54 answerable + 13 unanswerable queries).
2. Packaging of repository, documentation, and evaluation artifacts for reviewer handoff.
