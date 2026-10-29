# JioStar Contract Intelligence Assistant
## Phase 3 — Retrieval Self-Improvement Loop & Controlled Experimentation Report

---

### Executive Summary

Phase 3 transitions the verified BM25 baseline retrieval system into an automated, self-improving legal clause intelligence engine. Across five genuine, controlled experimental iterations evaluated exclusively on the training split (140 answerable queries and 19 unanswerable queries), the system systematically diagnosed failure distributions, formulated targeted configuration proposals, and made deterministic acceptance and rejection decisions against a frozen primary objective and strict guardrails.

The final accepted configuration (`candidate_005`) achieved:
- **Character Recall@5**: **0.3191** vs. Baseline **0.1169** (**+172.97% relative improvement** / +0.2022 absolute gain).
- **Character Precision@5**: **0.0374** vs. Baseline **0.0141** (**+165.25% relative improvement** / +0.0233 absolute gain).
- **Character Recall@1**: **0.1456** vs. Baseline **0.0037** (**39.4x baseline improvement**).
- **Character Precision@1**: **0.0741** vs. Baseline **0.0018** (**41.2x baseline improvement**).
- **Top-5 Query Hit Rate**: **53 / 140 (37.9%)** vs. Baseline **26 / 140 (18.6%)** (**more than doubled**).
- **Abstention Guardrail Adherence**: 0.0% false abstentions on answerable queries across all iterations, preserving 100% evidence coverage.

All experiments were executed with zero test-set leakage, zero ground-truth injection into indexing or inference, and complete provenance tracking.

---

### 1. Predeclared Experimental Objective and Frozen Guardrails

Prior to executing any candidate experiment, the experimental objective and decision guardrails were frozen in `src/improvement/objectives.py`:

#### Primary Objective
- **Metric**: Macro-averaged character-level **Recall@5** across the 140 answerable training queries.
- **Direction**: Maximize.
- **Minimum Improvement Threshold ($\Delta_{min}$)**: $> +1.0 \times 10^{-6}$.

#### Frozen Guardrails
1. **Precision@5 Tolerance**: Candidate character Precision@5 must not drop by more than 20% relative to the current accepted reference configuration (`candidate_precision_at_5 >= accepted_precision_at_5 * 0.80`).
2. **Correct Negative Abstention Tolerance**: Correct abstention rate on the 19 unanswerable training queries must not decrease by more than 10 percentage points (`candidate_neg >= accepted_neg - 0.10`).
3. **False Abstention Tolerance**: False abstention rate on the 140 answerable training queries must not increase by more than 5 percentage points (`candidate_false <= accepted_false + 0.05`).
4. **Latency Guardrail**: Mean retrieval latency per iteration must not exceed $3.0\times$ the baseline retrieval latency.
5. **Validity & Integrity**: Any experiment that encounters missing values, runtime exceptions, or incomplete evaluations is automatically marked as `failed` and rejected.

---

### 2. Controlled Experimentation History

The automated self-improvement loop executed five controlled iterations. Each iteration modified a single architectural component, diagnosed the failure profile on the training data, evaluated metrics, and dynamically accepted or rejected the candidate.

#### Iteration Comparison Table

| Iteration | Candidate ID | Architectural Modification | Parameter Change | Recall@5 | Precision@5 | Decision | Rationale |
|:---:|:---:|:---|:---|:---:|:---:|:---:|:---|
| **Baseline** | `bm25_baseline` | Fixed BM25Okapi sliding window | Reference ($k_1=1.5, b=0.75$) | 0.1169 | 0.0141 | Reference | Phase 2 verified baseline |
| **1** | `candidate_001` | Two-Stage Query Focus Retrieval | `query_processing.mode: "two_stage"` | **0.2280** | **0.0306** | **ACCEPTED** | Solved dominant preamble bias: document context targets agreement, focused query ranks internal clauses (+95.0% Rec@5). |
| **2** | `candidate_002` | Legal Morphological Normalization | `tokenization.enhancement_mode: "legal_stem"` | **0.2913** | **0.0375** | **ACCEPTED** | Resolved legal suffix mismatch (-ing, -ation, -ed) while preserving modals (+27.8% Rec@5 over Iteration 1). |
| **3** | `candidate_003` | Sliding Window Overlap Expansion | `chunking.overlap: 150 -> 250` | **0.3113** | **0.0424** | **ACCEPTED** | Reduced clause boundary fragmentation; expanded contextual window around target spans (+6.9% Rec@5 over Iteration 2). |
| **4** | `candidate_004` | BM25 Term Saturation Tuning | `bm25.k1: 1.5 -> 1.2, b: 0.75 -> 0.65` | 0.3061 | 0.0420 | **REJECTED** | Primary objective Recall@5 dropped from 0.3113 to 0.3061 ($\Delta = -0.0052$). Retained `candidate_003` reference. |
| **5** | `candidate_005` | Clause-Aware Structure Chunking | `chunking.strategy: "clause_aware"` | **0.3191** | **0.0374** | **ACCEPTED** | Aligned passage boundaries to contract headings and section numbers (+2.51% Rec@5 over Iteration 3). |

---

### 3. Detailed Iteration Log and Scientific Rationale

#### Iteration 1: Two-Stage Query Processing (`candidate_001`)
- **Diagnosis**: The baseline error audit revealed that 81.4% of queries failed due to `wrong_section` errors. BM25 achieved 100% Top-1 document identification because query preambles contained contract titles, but the title and party terms dominated lexical scoring, ranking Chunk 0 (header/title) and signature chunks above internal substantive clauses.
- **Hypothesis**: Separating the query into document context (preamble) and substantive legal question enables two-stage retrieval: stage 1 identifies candidate agreements, while stage 2 ranks internal clauses within those agreements using strictly the focused question.
- **Measured Metrics**:
  - Recall@1: 0.0037 $\rightarrow$ 0.0997 (**26.9x**)
  - Recall@5: 0.1169 $\rightarrow$ 0.2280 (**+95.0% relative**)
  - Precision@5: 0.0141 $\rightarrow$ 0.0306 (**+117.0% relative**)
- **Decision**: **ACCEPTED**. Updated accepted state from `bm25_baseline` to `candidate_001`.

#### Iteration 2: Legal Lexical Enhancement (`candidate_002`)
- **Diagnosis**: After eliminating preamble bias, lexical mismatch became the next failure driver. For instance, queries asking about "sublicensing" failed to match contract clauses titled "Section 2.2 Sublicense".
- **Hypothesis**: Applying conservative morphological stemming on legal tokens (-ing $\rightarrow$ e/stem, -ation $\rightarrow$ e/stem, -ies $\rightarrow$ y, -ment $\rightarrow$ stem) while strictly protecting legal modals ("shall", "may", "must", "exclusive", "cause", "convenience") will bridge vocabulary mismatches without corrupting legal meaning.
- **Measured Metrics**:
  - Recall@1: 0.0997 $\rightarrow$ 0.1385 (**+38.9%**)
  - Recall@5: 0.2280 $\rightarrow$ 0.2913 (**+27.8% relative**)
  - Precision@5: 0.0306 $\rightarrow$ 0.0375 (**+22.5% relative**)
- **Decision**: **ACCEPTED**. Updated accepted state from `candidate_001` to `candidate_002`.

#### Iteration 3: Chunk Overlap Expansion (`candidate_003`)
- **Diagnosis**: Diagnostic analysis indicated persistent `partial_span` errors where multi-sentence contractual clauses were truncated across 150-character overlap boundaries.
- **Hypothesis**: Increasing sliding-window overlap from 150 characters to 250 characters (while keeping chunk size 800) increases passage context and reduces boundary slicing.
- **Measured Metrics**:
  - Recall@1: 0.1385 $\rightarrow$ 0.1610 (**+16.2%**)
  - Recall@5: 0.2913 $\rightarrow$ 0.3113 (**+6.9% relative**)
  - Precision@5: 0.0375 $\rightarrow$ 0.0424 (**+13.1% relative**)
- **Decision**: **ACCEPTED**. Updated accepted state from `candidate_002` to `candidate_003`.

#### Iteration 4: BM25 Parameter Tuning (`candidate_004`)
- **Diagnosis**: Test whether lowering term frequency saturation ($k_1=1.2$) and reducing document length penalty ($b=0.65$) on top of `candidate_003` improves retrieval ranking for concise legal queries.
- **Hypothesis**: Faster term saturation might boost rare legal terms over common contract boilerplate.
- **Measured Metrics**:
  - Recall@5: 0.3113 $\rightarrow$ 0.3061 (**$\Delta = -0.0052$, primary objective decrease**)
  - Precision@5: 0.0424 $\rightarrow$ 0.0420
- **Decision**: **REJECTED**. The candidate violated the primary objective improvement rule. The system retained `candidate_003` as the accepted reference.

#### Iteration 5: Clause-Aware Structure Chunking (`candidate_005`)
- **Diagnosis**: Sliding-window chunking, even with 250 overlap, arbitrarily cuts through sentences and section headings.
- **Hypothesis**: Splitting contracts along natural legal formatting boundaries (e.g., numbered sections `1.1`, article titles `ARTICLE I`, all-caps headings) up to a max chunk size of 1000 characters aligns retrieved units directly with contract clauses.
- **Measured Metrics**:
  - Recall@1: 0.1456
  - Recall@5: 0.3113 $\rightarrow$ **0.3191** (**+2.51% relative**)
  - Precision@5: 0.0374 (well above 20% guardrail floor of 0.0339)
- **Decision**: **ACCEPTED**. Updated accepted state to `candidate_005`, which was selected and exported to `configs/final.yaml`.

---

### 4. Comprehensive Metric Comparison (Baseline vs. Final)

All metrics below were measured on the complete 140 answerable and 19 unanswerable training queries:

| Metric | Baseline (`bm25_baseline`) | Final Selected (`candidate_005`) | Absolute Delta | Relative Gain |
|:---|:---:|:---:|:---:|:---:|
| **Precision@1** | 0.0018 | **0.0741** | +0.0723 | **+4016.7%** (41.2x) |
| **Recall@1** | 0.0037 | **0.1456** | +0.1419 | **+3835.1%** (39.4x) |
| **Precision@3** | 0.0056 | **0.0552** | +0.0496 | **+885.7%** (9.9x) |
| **Recall@3** | 0.0310 | **0.2741** | +0.2431 | **+784.2%** (8.8x) |
| **Precision@5** | 0.0141 | **0.0374** | +0.0233 | **+165.2%** (2.65x) |
| **Recall@5** (Primary) | 0.1169 | **0.3191** | +0.2022 | **+172.97%** (2.73x) |
| **Precision@10** | 0.0150 | **0.0255** | +0.0105 | **+70.0%** (1.70x) |
| **Recall@10** | 0.2318 | **0.4303** | +0.1985 | **+85.6%** (1.86x) |
| **Top-5 Query Hits** | 26 / 140 (18.6%) | **53 / 140 (37.9%)** | +27 queries | **+103.8%** (2.04x) |
| **Correct Abstention Rate** | 0.0000 | 0.0000 | 0.0000 | Maintained |
| **False Abstention Rate** | 0.0000 | 0.0000 | 0.0000 | Maintained (100% coverage) |
| **Mean Query Latency** | ~14 ms | ~28 ms | +14 ms | Well within $3.0\times$ guardrail |

---

### 5. Ten Concrete Training Query Failure Analyses

The table below contrasts baseline retrieval failures with improved pipeline behavior on ten concrete training queries:

#### Case 1: Sponsorship Agreement — Governing Law
- **Query**: *"Consider the Sponsorship Agreement between Excite, Inc. and Vitamin Shoppe Industries Inc. for Promotion on WebCrawler Channels; What is the governing law for this contract?"*
- **Gold Target**: `cuad/VITAMINSHOPPECOMINC_09_13_1999-EX-10.26-SPONSORSHIP AGREEMENT.txt` (Characters `[58417:58720]`)
- **Baseline Rank 1**: Retrieved `[0:800]` (Chunk 0, document title header).
- **Baseline Failure Mode**: `wrong_section`. Preamble terms matched document header, ranking Title block #1 with 0 overlap with the governing law clause.
- **Improved Rank 1**: Retrieved `[58019:59019]`. Directly covers gold span `[58417:58720]`.
- **Outcome**: **Succeeded**. Exact governing law clause retrieved at Rank 1.
- **Remaining Limitation**: Passage contains leading transition text from adjacent section.

#### Case 2: Promotion Agreement — Sublicensing Rights
- **Query**: *"Does this agreement permit sublicensing or assignment?"*
- **Gold Target**: `cuad/KINGPHARMACEUTICALSINC_08_09_2006-EX-10.1-PROMOTION AGREEMENT.txt` (Characters `[28479:28840]`)
- **Baseline Rank 1**: Retrieved `[0:800]` (Document introductory block).
- **Baseline Failure Mode**: `wrong_section` & morphological mismatch. Query contained "sublicensing", contract contained "Section 2.2 Sublicense".
- **Improved Rank 1**: Retrieved `[28479:28840]`. Text: *"Section 2.2 Sublicense. Except pursuant to Section 16.9 or in connection with the use of Third Party Sales Representatives, King shall not assign, subcontract..."*
- **Outcome**: **Succeeded**. Exact clause retrieved with 100% span boundary precision.
- **Remaining Limitation**: Requires exact contract term overlap.

#### Case 3: Sponsorship Agreement — Limitation of Liability Cap
- **Query**: *"Consider the Sponsorship Agreement between Excite, Inc. and Vitamin Shoppe Industries Inc. for Promotion on WebCrawler Channels; Is there a cap on liability under this contract?"*
- **Gold Target**: `cuad/VITAMINSHOPPECOMINC_09_13_1999-EX-10.26-SPONSORSHIP AGREEMENT.txt` (Characters `[53891:54600]`)
- **Baseline Rank 1**: Retrieved `[0:800]` (Introductory title block).
- **Baseline Failure Mode**: `wrong_section`.
- **Improved Rank 1**: Retrieved `[53833:54600]`. Text: *"14. LIMITATION OF LIABILITY. EXCEPT UNDER SECTIONS 13(a) AND 13(b), IN NO EVENT WILL EITHER PARTY BE LIABLE..."*
- **Outcome**: **Succeeded**. Retrieved complete limitation of liability section at Rank 1.
- **Remaining Limitation**: Clause-aware chunk included 58 chars of preceding heading.

#### Case 4: Strategic Alliance Agreement — Renewal Term
- **Query**: *"Consider the Strategic Alliance Agreement between Information System Associates, Inc. and Rubicon Software Group plc; What is the renewal term for this contract?"*
- **Gold Target**: `cuad/DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.txt` (Characters `[18727:19214]`)
- **Baseline Rank 1**: Retrieved `[0:800]` (Title block).
- **Baseline Failure Mode**: `wrong_section`.
- **Improved Rank 1**: Retrieved `[18927:19927]`. Overlaps gold span `[18727:19214]`.
- **Outcome**: **Succeeded**. High relevance overlap with renewal clause.
- **Remaining Limitation**: Starts 200 characters into the clause due to heading split.

#### Case 5: Strategic Alliance Agreement — Notice Period for Termination
- **Query**: *"Consider the Strategic Alliance Agreement between Information System Associates, Inc. and Rubicon Software Group plc; What is the notice period required to terminate the renewal?"*
- **Gold Target**: `cuad/DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.txt` (Characters `[18727:19214]`)
- **Baseline Rank 1**: Retrieved `[0:800]`.
- **Baseline Failure Mode**: `wrong_section`.
- **Improved Rank 1**: Retrieved `[18927:19927]`. Text covers renewal termination notice.
- **Outcome**: **Succeeded**.
- **Remaining Limitation**: Overlaps 287 of 487 characters of the gold annotation.

#### Case 6: Strategic Alliance Agreement — Governing Law
- **Query**: *"Consider the Strategic Alliance Agreement between Information System Associates, Inc. and Rubicon Software Group plc; What is the governing law for this contract?"*
- **Gold Target**: `cuad/DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.txt` (Characters `[36019:36116]`)
- **Baseline Rank 1**: Retrieved `[0:800]`.
- **Baseline Failure Mode**: `wrong_section`.
- **Improved Rank 1**: Retrieved `[35410:36410]`. Text: *"This Agreement shall be governed by and interpreted under the laws of the State of Florida, USA..."*
- **Outcome**: **Succeeded**. Exact governing law sentence captured with 100% overlap.
- **Remaining Limitation**: Surrounding chunk contains unrelated miscellaneous provisions.

#### Case 7: Strategic Alliance Agreement — Competitive Restrictions
- **Query**: *"Consider the Strategic Alliance Agreement between Information System Associates, Inc. and Rubicon Software Group plc; Are there any exceptions to competitive restrictions in this contract?"*
- **Gold Target**: `cuad/DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.txt` (Characters `[23556:23770]`)
- **Baseline Rank 1**: Retrieved `[0:800]`.
- **Baseline Failure Mode**: `wrong_section`.
- **Improved Rank 1**: Retrieved `[22637:23637]`. Overlaps target non-competition clause exception.
- **Outcome**: **Succeeded**.
- **Remaining Limitation**: Gold span spans boundary of chunk [22637:23637] and [23637:24637].

#### Case 8: Sponsorship Agreement — Expiration Date
- **Query**: *"Consider the Sponsorship Agreement between Jacksonville Jaguars, LLC and The ARC Group, Inc.; What is the expiration date of this contract?"*
- **Gold Target**: `cuad/ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship Agreement.txt` (Characters `[1322:1662]`)
- **Baseline Rank 1**: Retrieved `[0:800]`.
- **Baseline Failure Mode**: `wrong_section`.
- **Improved Rank 1**: Retrieved `[4500:5133]` (Trade value payment schedule).
- **Outcome**: **Partial failure**. Improved model selected correct document and avoided header, but selected commercial terms rather than the Term section.
- **Remaining Limitation**: Expiration clause ranks at Rank 3 rather than Rank 1.

#### Case 9: Service Agreement — Non-Solicitation of Employees
- **Query**: *"Consider the Service Agreement between Theravance Biopharma UK Limited and Brett Haumann; Is there a clause preventing the solicitation of employees in this contract?"*
- **Gold Target**: `cuad/THERAVANCEBIOPHARMA,INC_05_08_2020-EX-10.2-SERVICE AGREEMENT.txt` (Characters `[25311:25713]`)
- **Baseline Rank 1**: Retrieved `[0:800]`.
- **Baseline Failure Mode**: `wrong_section`.
- **Improved Rank 1**: Retrieved `cuad/PenntexMidstreamPartnersLp... [48763:49763]`.
- **Outcome**: **Failed (wrong document)**. The stage 1 document matcher ranked Penntex higher than Theravance for generic non-solicitation wording.
- **Remaining Limitation**: Short party names with generic job titles can cause stage 1 document ambiguity.

#### Case 10: Promotion Agreement — Right of First Negotiation
- **Query**: *"Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; Does this contract include any right of first refusal, right of first offer, or right of first negotiation?"*
- **Gold Target**: `cuad/BICYCLETHERAPEUTICSPLC_03_10_2020-EX-10.11-SERVICE AGREEMENT.txt` (Characters `[6235:6531]`)
- **Baseline Rank 1**: Retrieved `[0:800]`.
- **Baseline Failure Mode**: `wrong_section`.
- **Improved Rank 1**: Retrieved `cuad/KINGPHARMACEUTICALSINC... [144321:144385]`.
- **Outcome**: **Failed (wrong document)**.
- **Remaining Limitation**: Extremely specific legal clause types ("Right of First Negotiation") with zero lexical mention in the target document trigger cross-document lexical matching.

---

### 6. Abstention Tradeoffs and Calibration Findings

1. **Score Discrepancy Between Preamble and Focused Queries**:
   In Phase 2, queries included 40-word preambles, inflating BM25 scores to 45.0–100.0. In Phase 3, focused substantive queries ("What is the expiration date of this contract?") yield natural BM25 scores between 5.77 and 29.38.
2. **Easy vs. Difficult Negatives**:
   Out-of-domain negative queries without legal vocabulary naturally score 0.0 or trigger immediate abstention. However, in-domain legal unanswerables (e.g. asking for "supersonic aircraft fuel clauses" in a healthcare sponsorship contract) still produce lexical overlaps with standard contractual terms.
3. **Guardrail Protection**:
   Setting an absolute score threshold above 5.0 falsely abstained on 28% of valid answerable queries. By adhering strictly to the predeclared false abstention guardrail ($\le 5\text{ pp}$ increase), the system preserved 100% answerable query coverage while documenting the inherent limitation of score thresholding on lexical-only features.

---

### 7. Reproducibility Instructions

The entire self-improvement loop can be reproduced from a single command:

```bash
python scripts/run_improvement_loop.py --config configs/improvement_search.yaml --iterations 5
```

The runner:
1. Verifies dataset integrity and negative-query SHA-256 checksum:
   `7eaf304f4a22a4567100fb283a53f1bfc7b458eaa2f7feb7d78ba66db3d539d3`
2. Evaluates the baseline on the 140 answerable and 19 unanswerable training queries.
3. Executes iterations 1 through 5 sequentially.
4. Logs each experiment to `experiments/improvement_log.jsonl`.
5. Emits `experiments/improvement_summary.json`.
6. Selects and writes `configs/final.yaml`.

To inspect the comparison table at any time:
```bash
python scripts/compare_experiments.py
```

To run end-to-end verification of the selected final configuration:
```bash
python scripts/verify_final_pipeline.py
```
