# JioStar Contract Intelligence Assistant

A production-grade Legal Agreement Retrieval and Character-Span Evidence Extraction System.

This repository implements the **JioStar Contract Intelligence Assistant** using the official **LegalBench-RAG-mini CUAD** benchmark.

---

## 1. Project Overview & Current Status

The Contract Intelligence Assistant retrieves relevant contractual clauses from complex legal agreements (SEC EDGAR filings), extracts exact character-offset spans as verifiable evidence, and reliably abstains when requested evidence is not present.

### Implementation Status:
- [x] **Phase 1: Project Foundation, Dataset Preparation, Git Setup, and Validation** (COMPLETED)
  - Official LegalBench-RAG CUAD dataset acquisition with verified provenance (`commit 2b9c248b`).
  - Deterministic 194-query mini benchmark selection.
  - SHA-256 train/test split partitioning (140 train / 54 test).
  - 100% verified ground-truth character offset spans (272/272 snippets valid).
  - 29 referenced contracts discovered and manifested (1,211,773 characters).
  - 32 unanswerable negative queries benchmark frozen (SHA-256: `7eaf304f4a22a4567100fb283a53f1bfc7b458eaa2f7feb7d78ba66db3d539d3`).
- [x] **Phase 2: BM25 Retrieval, Clause-Aware Chunking, Abstention, Streamlit Web App, CLI & Baseline Evaluation** (COMPLETED)
  - Fixed-character sliding-window chunker (800 chars, 150 overlap) with exact offset verification.
  - Lightweight deterministic legal tokenizer (`LegalTokenizer`).
  - Production BM25 Okapi retrieval engine (`BM25Retriever`).
  - Score-threshold abstention decision engine (`AbstentionEngine`).
  - Unified search pipeline (`ContractRetrievalPipeline`).
  - Interactive Streamlit web interface (`app/streamlit_app.py`).
  - Command-line interface (`cli.py`) supporting text and JSON output.
  - Character-level precision/recall and abstention metrics (`src/evaluation/metrics.py`).
  - Scientific baseline evaluation executed on training split (`experiments/baseline_train_metrics.json`).
- [x] **Phase 2.5: Critical Baseline Verification & Technical Evaluation Audit** (COMPLETED)
  - Verified and reproduced baseline training metrics: Prec@5: 0.0141, Rec@5: 0.1169.
  - Identified header/preamble bias: BM25 frequently ranks document titles and party preambles above internal contractual clauses.
  - Documented abstention score sensitivity and morphological mismatch.
  - 57 automated tests passing. Audit report committed at `026efe8`.
- [x] **Phase 3: Automated Self-Improvement Loop & Controlled Retrieval Optimization** (COMPLETED)
  - Fully executable self-improvement loop running 5 controlled iterations on training data.
  - Predeclared primary objective: Macro-averaged character **Recall@5** on 140 answerable training queries.
  - Frozen decision guardrails: Precision@5 drop $\le 20\%$, correct abstention drop $\le 10\text{pp}$, false abstention increase $\le 5\text{pp}$, latency factor $\le 3.0\times$.
  - Automated failure diagnostics categorizing errors (`wrong_document`, `wrong_section`, `partial_span`, `incorrect_abstention`, `failed_abstention`, `successful_retrieval`).
  - Two-Stage Query Processing (`QueryProcessor`) separating contract-context preambles from substantive questions.
  - Legal morphological normalization (`LexicalEnhancer`) bridging vocabulary variations (e.g. sublicensing/sublicense) while preserving legal modals.
  - Clause-aware structure chunking (`chunk_contract_by_clauses`) splitting along numbered sections and article headings.
  - Append-only experiment logging (`experiments/improvement_log.jsonl`) and summary (`experiments/improvement_summary.json`).
  - Automatic export of final selected configuration to `configs/final.yaml`.
  - Streamlit and CLI support for selecting between Final and Baseline configurations.
  - 79 automated tests passing across 15 test suites.

---

## 2. Verified Performance Metrics (Training Split)

Evaluated exclusively across the 140 answerable and 19 unanswerable training queries (zero test-set leakage):

| Metric | Phase 2 Baseline (`bm25_baseline`) | Phase 3 Final Selected (`candidate_005`) | Absolute Delta | Relative Improvement |
|:---|:---:|:---:|:---:|:---:|
| **Precision@1** | 0.0018 | **0.0741** | +0.0723 | **+4016.7%** (41.2x) |
| **Recall@1** | 0.0037 | **0.1456** | +0.1419 | **+3835.1%** (39.4x) |
| **Precision@3** | 0.0056 | **0.0552** | +0.0496 | **+885.7%** (9.9x) |
| **Recall@3** | 0.0310 | **0.2741** | +0.2431 | **+784.2%** (8.8x) |
| **Precision@5** | 0.0141 | **0.0374** | +0.0233 | **+165.25%** (2.65x) |
| **Recall@5** (Primary Objective) | 0.1169 | **0.3191** | +0.2022 | **+172.97%** (2.73x) |
| **Precision@10** | 0.0150 | **0.0255** | +0.0105 | **+70.0%** (1.70x) |
| **Recall@10** | 0.2318 | **0.4303** | +0.1985 | **+85.6%** (1.86x) |
| **Top-5 Query Hits** | 26 / 140 (18.6%) | **53 / 140 (37.9%)** | +27 queries | **+103.8%** (2.04x) |
| **False Abstention Rate** | 0.0% | **0.0%** | 0.0 pp | Maintained 100% coverage |

---

## 3. Technology Requirements & Installation

- **Python**: 3.11 (`3.11.9` verified)
- **Environment**: Cross-platform (Windows PowerShell / POSIX bash)
- **Dependencies**: `rank-bm25`, `streamlit`, `requests`, `huggingface_hub`, `PyYAML`, `tqdm`, `pytest`

### Step 1: Clone Repository
```bash
git clone https://github.com/Kunjdesai28/jiostar-contract-intelligence-assistant.git
cd jiostar-contract-intelligence-assistant
```

### Step 2: Initialize Virtual Environment
On Windows (PowerShell):
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. End-to-End Workflow & Execution Commands

### 1. Download & Prepare Benchmark Data
```bash
python scripts/download_dataset.py
python scripts/prepare_dataset.py
python scripts/validate_dataset.py
```

### 2. Execute the Automated Retrieval Self-Improvement Loop
Runs the 5 controlled optimization iterations on the training split:
```bash
python scripts/run_improvement_loop.py --config configs/improvement_search.yaml --iterations 5
```
- Logs all iterations to `experiments/improvement_log.jsonl`.
- Writes full experiment summary to `experiments/improvement_summary.json`.
- Exports winning configuration to `configs/final.yaml`.

### 3. Display Experiment Iteration Comparison Table
Prints the scientific progression and decisions across all iterations:
```bash
python scripts/compare_experiments.py
```

### 4. Verify Final Selected Pipeline
Runs end-to-end verification of `configs/final.yaml`:
```bash
python scripts/verify_final_pipeline.py
```

### 5. Run the Command-Line Interface (CLI)
Search for legal clauses directly from the terminal using the improved configuration:
```bash
# Text format with final selected configuration
python cli.py search --query "Does this agreement permit sublicensing or assignment?" --top-k 5 --config configs/final.yaml

# Machine-readable JSON output
python cli.py search --query "Does this agreement permit sublicensing or assignment?" --top-k 5 --format json --config configs/final.yaml

# Baseline comparison
python cli.py search --query "Does this agreement permit sublicensing or assignment?" --top-k 5 --config configs/baseline.yaml
```

### 6. Launch the Streamlit Web Application
Starts the interactive legal clause retrieval interface:
```bash
python -m streamlit run app/streamlit_app.py
```
Open `http://localhost:8501`. Use the sidebar dropdown to toggle between the **Final Selected Pipeline** (`configs/final.yaml`) and the **Baseline Pipeline** (`configs/baseline.yaml`).

### 7. Run Automated Tests
Execute the complete test suite (Phase 1 + Phase 2 + Phase 3):
```bash
pytest -v
```
All 79 tests pass in ~1.1 seconds.

---

## 5. Directory Structure

```text
jiostar-contract-intelligence-assistant/
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py           # Streamlit web app with config switching
│
├── configs/
│   ├── baseline.yaml              # Immutable Phase 2 baseline configuration
│   ├── improvement_search.yaml    # Search space, objective, and guardrails
│   └── final.yaml                 # Best accepted configuration from Phase 3 loop
│
├── docs/
│   ├── DATASET.md                 # Benchmark manifest and split documentation
│   ├── PHASE2_AUDIT.md            # Technical audit of Phase 2 baseline
│   └── PHASE3_EXPERIMENTS.md      # Full Phase 3 experiment report & 10 failure analyses
│
├── experiments/
│   ├── baseline_train_metrics.json    # Phase 2 baseline metrics (immutable reference)
│   ├── improvement_log.jsonl          # Append-only JSONL log of every iteration
│   └── improvement_summary.json       # Complete end-to-end experiment run summary
│
├── scripts/
│   ├── download_dataset.py        # HuggingFace benchmark acquisition
│   ├── prepare_dataset.py         # 194-query mini selection & contract manifest
│   ├── validate_dataset.py        # Character offset & split validation
│   ├── run_improvement_loop.py    # Self-improvement loop entrypoint
│   ├── compare_experiments.py     # Experiment comparison table printer
│   └── verify_final_pipeline.py   # Final pipeline smoke test
│
├── src/
│   ├── config.py                  # Project-root aware YAML configuration loader
│   ├── data/                      # Dataset splitting, loading, and validation
│   ├── retrieval/
│   │   ├── query_processor.py     # Preamble separation & focused query extraction
│   │   ├── lexical_enhancement.py # Legal morphological stemmer & normalizer
│   │   ├── clause_chunker.py      # Structure-aware heading chunker
│   │   ├── bm25.py                # BM25 Okapi retriever
│   │   ├── semantic.py            # Dense embedding retriever interface
│   │   ├── hybrid.py              # Reciprocal Rank Fusion (RRF) combiner
│   │   ├── abstention.py          # Score thresholding engine
│   │   └── pipeline.py            # Unified retrieval pipeline interface
│   │
│   ├── evaluation/
│   │   ├── metrics.py             # Character-level precision, recall, abstention
│   │   ├── baseline.py            # Baseline evaluation runner
│   │   └── diagnostics.py         # Automated failure classification per query
│   │
│   └── improvement/
│       ├── objectives.py          # Frozen primary objective and guardrails
│       ├── proposer.py            # Rule-based diagnostic proposal engine
│       ├── decision.py            # Deterministic acceptance/rejection engine
│       ├── experiment_logger.py   # Append-only logger and summary emitter
│       └── runner.py              # End-to-end self-improvement loop orchestrator
│
└── tests/                         # 79 automated tests across 15 test suites
```

---

## 6. Privacy & Ethics Disclosure

- **Zero Test Set Leakage**: All parameter tuning, chunking decisions, and model evaluations were executed strictly on the training split (140 answerable + 19 unanswerable queries). The held-out test split was never touched during Phase 3.
- **Contract Privacy**: Contract text is processed locally. No confidential agreements or user queries are transmitted to external commercial LLM APIs.
- **AI Tool Usage**: Code scaffolding and experimental orchestration were assisted by Google DeepMind's Antigravity agentic coding assistant under strict human supervision, continuous verification against ground-truth labels, and deterministic regression testing.
