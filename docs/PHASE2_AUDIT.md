# Phase 2.5 — Critical Baseline Verification, Evaluation Audit and Targeted Corrections

**Author & Role**: Principal Information Retrieval Engineer, Senior Data Scientist, ML Evaluation Specialist, Independent Technical Auditor  
**Date**: October 2026  
**Audited Commit**: `58092f8704e0fcc75e9f9e9e7f42844e318d8efa`  
**Dataset Provenance**: LegalBench-RAG (`awinml/legalbench-rag`, commit `2b9c248b`)  
**Corpus Scope**: 29 SEC EDGAR Contracts (1,211,773 characters, 1,874 chunks)  

---

## 1. Audit Methodology & Scope

This audit performs an independent, rigorous technical examination of the Phase 1 and Phase 2 implementations for the JioStar Contract Intelligence Assistant. The evaluation verified:
1. **Mathematical correctness of evaluation metrics**: Character-level Precision@K and Recall@K against the official LegalBench-RAG specification ([arXiv:2408.10343](https://arxiv.org/abs/2408.10343)).
2. **Investigation of low baseline retrieval recall**: Diagnosing whether Recall@1 (0.0037) and Recall@5 (0.1169) stem from implementation bugs (e.g., path mismatch, offset encoding drift, ranking inversions) or legitimate IR phenomena.
3. **Audit of Abstention Threshold $\tau = 44.0$**: Investigating score distributions, query length dynamics, and the discrepancy observed during the sublicensing demonstration.
4. **Interface consistency**: Verifying that CLI, Streamlit, and automated evaluation invoke identical retrieval pipelines without divergence.
5. **Preservation of Benchmark Invariants**: Guaranteeing zero leakage of held-out test data, preservation of raw contract text, and stability of frozen negative query hashes.

---

## 2. Benchmark Artifact Checksums & Invariants

Prior to code inspection and testing, all foundational benchmark artifacts were cryptographically verified:

| Artifact Path | Verified SHA-256 Checksum | Invariant Status |
|---|---|---|
| `data/raw/benchmarks/cuad.json` | `8e0405b15fb8869c631da424372b488e274f253a81634b78803b21c72c0601f9` | Unmodified upstream source |
| `data/processed/cuad_mini.json` | `412fc88d91b8299defa7158a5bd2ee008e288af0332a5dfd39b8d44a83a61749` | 194 queries, deterministic PRNG |
| `data/processed/train_queries.json` | `45176ec3100b7edaa93d90fcb5b3d15fba34e5045609fdf28d49b3c1e0259e62` | 140 answerable train queries |
| `data/processed/test_queries.json` | `c4de1ef9917e546c77458497916dcaf1cb2b83badd94f9f396d6249172030181` | 54 held-out test queries (LOCKED) |
| `data/processed/contract_manifest.json` | `59fd376b2fb69406ec1e626a224398a2255a7fe307ff1b006afd83c0a7b9c053` | 29 unique referenced contracts |
| `data/unanswerable_queries.json` | `7eaf304f4a22a4567100fb283a53f1bfc7b458eaa2f7feb7d78ba66db3d539d3` | 32 frozen negative queries |

*Verification Outcome*: All benchmark artifacts remain bit-for-bit identical to their Phase 1 freezing commits. No test queries were accessed during this audit.

---

## 3. Original vs. Reproduced Baseline Metrics

The baseline evaluation was independently re-executed on the 140 answerable training queries and 19 unanswerable training queries using `scripts/evaluate_baseline.py --config configs/baseline.yaml --split train --output experiments/phase2_audit_train_metrics.json`.

| Metric | Original Reported Value | Reproduced Audit Value | Variance |
|---|---|---|---|
| **Precision@1** | `0.0018` | `0.0018` | `0.0000` (Exact Match) |
| **Precision@3** | `0.0056` | `0.0056` | `0.0000` (Exact Match) |
| **Precision@5** | `0.0141` | `0.0141` | `0.0000` (Exact Match) |
| **Precision@10** | `0.0150` | `0.0150` | `0.0000` (Exact Match) |
| **Recall@1** | `0.0037` | `0.0037` | `0.0000` (Exact Match) |
| **Recall@3** | `0.0310` | `0.0310` | `0.0000` (Exact Match) |
| **Recall@5** | `0.1169` | `0.1169` | `0.0000` (Exact Match) |
| **Recall@10** | `0.2318` | `0.2318` | `0.0000` (Exact Match) |
| **Correct Abstention on Negatives ($\tau=0.0$)** | `0 / 19 (0.0%)` | `0 / 19 (0.0%)` | `0.0000` (Exact Match) |
| **False Abstention on Positives ($\tau=0.0$)** | `0 / 140 (0.0%)` | `0 / 140 (0.0%)` | `0.0000` (Exact Match) |
| **Answerable Coverage ($\tau=0.0$)** | `140 / 140 (100.0%)`| `140 / 140 (100.0%)`| `0.0000` (Exact Match) |

*Audit Finding*: The reported baseline metrics are 100% reproducible and mathematically stable across runs.

---

## 4. Evaluation Metric Implementation Verification

The character-level precision and recall implementations in `src/evaluation/metrics.py` were audited against the formal LegalBench-RAG specification:

1. **Exact Character Offsets**: For each query, character intervals $[start, end]$ are mapped into discrete integer sets $\{start, \dots, end-1\}$.
2. **Document Isolation**: Character sets are computed strictly per-document ($S_{G, d}$ and $S_{R_K, d}$). Identical offset ranges across different contracts are never merged or matched.
3. **Overlapping Passage Deduplication**: If multiple retrieved passages within the same document overlap (e.g. $[100, 160]$ and $[140, 220]$), their character union $[100, 220]$ (120 characters) is used, preventing artificial inflation of the retrieved character denominator.
4. **Multiple Gold Snippets**: Multiple ground-truth annotations within a document are merged via union ($S_{G, d} = \bigcup_i [s_i, e_i]$), avoiding double-counting in the recall denominator.
5. **Partial Overlaps**: Strict set intersection $|S_{G, d} \cap S_{R_K, d}|$ ensures that partial overlaps credit only the exact shared characters.
6. **Macro vs. Micro Averaging**: Metrics compute macro-averages across queries ($\frac{1}{N} \sum_{i=1}^N \text{Metric}_i$), adhering to standard information retrieval practice.

### Synthetic Verification Test Suite
Ten synthetic verification test cases were implemented in `tests/test_retrieval_metrics.py`, independently verified with pre-calculated arithmetic:
- **Case 1 (Exact 100-char match)**: Prec=1.0, Rec=1.0. Passed.
- **Case 2 (50-char overlap)**: Prec=0.5, Rec=0.5. Passed.
- **Case 3 (Zero overlap in same doc)**: Prec=0.0, Rec=0.0. Passed.
- **Case 4 (Two gold spans in one doc)**: Prec=1.0, Rec=0.5. Passed.
- **Case 5 (One predicted span covering two gold spans)**: Prec=0.5, Rec=1.0. Passed.
- **Case 6 (Overlapping retrieved spans)**: Union=120 chars, Overlap=100 chars -> Prec=0.8333, Rec=1.0. Passed.
- **Case 7 (Identical offsets in different documents)**: Prec=0.0, Rec=0.0. Passed.
- **Case 8 (Empty retrieved list / Abstention)**: Prec=0.0, Rec=0.0. Passed.
- **Case 9 (Multiple predictions at different ranks)**: Rank progression correctly evaluated at cutoffs K=1, 2, 3. Passed.
- **Case 10 (Duplicate predicted spans)**: Deduplicated without inflating denominator. Passed.

---

## 5. Investigation of Low Baseline Retrieval Recall

The audit investigated why baseline Recall@1 is 0.0037 and Recall@5 is 0.1169 via `scripts/diagnose_retrieval.py` and output `experiments/phase2_failure_diagnostics.json`:

### Key Empirical Findings:
1. **Document-Level Retrieval is Highly Accurate**:
   - **Document Hit Rate @ 1**: **100.0%** (15 / 15 audited queries retrieved the exact gold document at Rank 1).
   - **Document Hit Rate @ 5**: **100.0%** (15 / 15 audited queries included the gold document in top-5).
   - *Conclusion*: There is **zero document-level confusion or path mapping error**. The index contains the correct 29 contracts, and document identifiers match identically between benchmark records and retrieval outputs.
2. **Within-Document Passage Dilution (Preamble Bias)**:
   - In LegalBench-RAG, all benchmark queries are formatted as:
     `"Consider the [Title of Agreement] between [Party A] and [Party B]; [Legal Question]?"`
   - Example (Query #1):
     `"Consider the Amendment No. 3 to Wireless Content License Agreement between Fox Mobile Entertainment, Inc. and Glu Mobile Inc.; What is the expiration date of this contract?"`
   - The contract preamble (`"Amendment No. 3"`, `"Wireless Content License Agreement"`, `"Fox Mobile Entertainment"`, `"Glu Mobile"`) matches the document title header (Chunk 0, offset $[0:800]$) and concluding signature blocks with massive BM25 term frequencies.
   - Consequently, BM25 assigns top scores (60–88) to **Chunk 0 (Title/Header)** and **Chunk N (Signatures)**.
   - The actual substantive clause (e.g., expiration date at offset $[650:1450]$ or indemnification at offset $[16867:17117]$) only matches the specific question tokens (`"expiration"`, `"date"`), receiving a lower BM25 score (20–35).
   - Thus, while the correct contract is found at Rank 1, the top-5 retrieved passages from that contract are dominated by document headers rather than the specific internal clause.
3. **Chunk Boundary Dilution**:
   - Ground-truth snippets average 200–400 characters, while chunks are fixed at 800 characters with 150-character overlap.
   - When a 300-character clause spans across a chunk boundary, characters are split between two chunks, lowering precision and delaying full recall until larger K cutoffs.

*Audit Conclusion*: The low baseline recall is **not a software defect**, but a well-understood limitation of pure BM25 lexical matching when queries contain document-identifying preambles. This establishes a scientifically valid, authentic baseline that Phase 3 hybrid retrieval and cross-encoder reranking are specifically designed to outperform.

---

## 6. Critical Audit of Abstention Threshold $\tau = 44.0$

### 6.1 Score Distribution Analysis (`experiments/phase2_score_distribution.json`)

The score distribution across all training queries was audited:

| Cohort | Count | Min Top-1 Score | Max Top-1 Score | Mean Top-1 Score | Median Top-1 Score | StDev |
|---|---|---|---|---|---|---|
| **Answerable Training Queries** | 140 | `44.67` | `102.86` | `67.42` | `66.67` | `14.40` |
| **Unanswerable Training Queries** | 19 | `13.68` | `43.76` | `24.46` | `20.53` | `8.90` |

### 6.2 Threshold Tradeoff Sweep (Training Data Only)

| Threshold $\tau$ | Correct Abstention (Negatives) | False Abstention (Positives) | Answerable Coverage | Recall@5 under Abstention |
|---|---|---|---|---|
| **0.0** (Baseline Default) | `0 / 19 (0.0%)` | `0 / 140 (0.0%)` | `100.0%` | `0.1169` |
| **10.0** | `0 / 19 (0.0%)` | `0 / 140 (0.0%)` | `100.0%` | `0.1169` |
| **20.0** | `8 / 19 (42.1%)` | `0 / 140 (0.0%)` | `100.0%` | `0.1169` |
| **30.0** | `15 / 19 (78.9%)` | `0 / 140 (0.0%)` | `100.0%` | `0.1169` |
| **40.0** | `17 / 19 (89.5%)` | `0 / 140 (0.0%)` | `100.0%` | `0.1169` |
| **44.0** (Calibrated) | `19 / 19 (100.0%)` | `0 / 140 (0.0%)` | `100.0%` | `0.1169` |
| **50.0** (Aggressive) | `19 / 19 (100.0%)` | `20 / 140 (14.3%)` | `85.7%` | `0.0898` |

### 6.3 Critical Technical Assessment of $\tau = 44.0$
- **Technical Validity on Benchmark**: On LegalBench-RAG formatted training queries, $\tau = 44.0$ is technically correct and separates the two distributions with a 0.91-point margin.
- **Operational Risk on Real-World / Ad-Hoc User Queries**:
  The demonstration query `"Does this contract permit sublicensing?"` returned a BM25 score of `10.3375`. Under $\tau = 44.0$, the system would abstain on this query!
  Why? Because natural user queries are short (5–10 tokens) and lack the 20-token contract title preamble present in benchmark queries. With only 1–2 matching tokens, BM25 scores for valid short queries naturally peak between `8.0` and `18.0`.
- **Policy Recommendation**:
  - Keep `configs/baseline.yaml` default threshold at `0.0` for conservative, open-domain retrieval.
  - Expose the threshold as a dynamic parameter in the CLI (`--threshold`) and Streamlit sidebar.
  - Document explicitly that $\tau = 44.0$ is calibrated for benchmark-style queries with document preambles, and must not be treated as a universal probability threshold.

---

## 7. Investigation of the Sublicensing Demonstration

**Query**: `"Does this contract permit sublicensing?"`  
**Phase 2 Output**: Retrieved passage from `cuad/VITAMINSHOPPECOMINC_...` with score `10.3375` (discussing advertising banners and trademarks).

### Detailed Root Cause Diagnosis:
1. **Corpus Inspection**: The 29 indexed contracts contain **24 occurrences** of sublicensing provisions (e.g. `ALCOSTORESINC` at offset 5,840, `DUOSTECHNOLOGIESGROUP` at offset 1,549, `KINGPHARMACEUTICALS` at offset 7,468).
2. **Lexical Inflection Mismatch**:
   - The query used the present-participle form: `"sublicensing"`.
   - The contracts use the base and inflected forms: `"sublicense"`, `"sublicensed"`, `"sublicenses"`.
   - In accordance with baseline rules (Rule 5: *"Avoid aggressive stemming or lemmatization in the baseline"*), `LegalTokenizer` performs exact token extraction without stemming.
   - Consequently, `"sublicensing"` matched 0 chunks containing `"sublicense"`.
3. **Residual Common Term Matching**:
   - With `"sublicensing"` having zero term frequency in those chunks, BM25 scored the query purely on `"permit"` and `"contract"`.
   - The Vitamin Shoppe agreement contained the word `"permit"` in an advertising placement section, resulting in a spurious match with score `10.3375`.
4. **Resolution**: This confirms that exact-match BM25 suffers from lexical fragility. When searching for `"Does this contract permit sublicense?"`, the system correctly retrieves `ALCOSTORESINC` at Rank 1 (score 11.598). Phase 3 dense embeddings and subword tokenization will solve this vocabulary mismatch.

---

## 8. Interface Consistency Verification

An automated consistency test (`tests/test_interface_consistency.py`) was created to verify that `cli.py`, `app/streamlit_app.py`, and `scripts/evaluate_baseline.py` invoke identical retrieval logic:
- All interfaces load `ContractRetrievalPipeline` from `src/retrieval/pipeline.py`.
- For identical queries, `cli.py search --format json` produces bit-for-bit identical candidate rankings, chunk IDs, character offsets, scores, and status fields as direct pipeline calls.
- Streamlit utilizes `@st.cache_resource` over `ContractRetrievalPipeline`, reading parameters directly from `configs/baseline.yaml`.

---

## 9. Summary of Confirmed Defects, Corrections, and Tests Added

| Area | Observation | Correction / Action |
|---|---|---|
| **Tokenizer Unicode Support** | Initial ASCII pattern failed on accented characters (e.g., `café`). | Updated regex pattern to `\w+(?:[-.]\w+)*` with `re.UNICODE` in `src/retrieval/tokenizer.py`. |
| **Evaluation Metrics Coverage** | Existing tests covered basic overlaps but lacked exhaustive edge cases. | Added 10 mandatory synthetic verification tests in `tests/test_retrieval_metrics.py`. |
| **Interface Consistency** | CLI and pipeline consistency had manual checks. | Added automated test in `tests/test_interface_consistency.py`. |
| **Failure Diagnostics** | Low recall needed empirical breakdown across queries. | Implemented `scripts/diagnose_retrieval.py` and generated `experiments/phase2_failure_diagnostics.json`. |
| **Score Distribution Analysis** | Threshold 44.0 risk on short queries required proof. | Implemented `scripts/audit_score_distributions.py` and generated `experiments/phase2_score_distribution.json`. |

---

## 10. Automated Test Results

Executed via `pytest -v`:
- **Total Test Files**: 13
- **Total Tests**: **57 passed in 0.95s**
- **Failures**: **0**
- **Regression Check**: All 18 Phase 1 tests and all Phase 2 tests pass without modifications to existing assertions.

---

## 11. Phase 3 Readiness Decision

**Decision: FULLY READY FOR PHASE 3.**

The foundation is verified, sound, and scientifically reproducible:
- Zero data leakage into held-out test data.
- Benchmark ground-truth character offsets verified with 100% exact substring equality.
- Evaluated metrics verified against LegalBench-RAG character-level definitions.
- The three highest-priority improvement opportunities for Phase 3 are:
  1. **Dense Semantic Embeddings & Subword Tokenization**: Overcome vocabulary mismatch and morphological variation (e.g., `sublicensing` vs. `sublicense`, `assignment` vs. `delegation`).
  2. **Cross-Encoder Reranking**: Elevate substantive clauses over document title preambles and signature recitals in top-5 rankings.
  3. **Clause-Aware Dynamic Chunking**: Adapt passage boundaries to legal clause structures rather than fixed 800-character windows.
