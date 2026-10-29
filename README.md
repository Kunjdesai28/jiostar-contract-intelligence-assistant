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
  - 32 unanswerable negative queries benchmark frozen.
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
  - 52 automated tests passing across 12 test suites.
- [ ] **Phase 3: Dense Retrieval, Hybrid Fusion, Reranking & Self-Improvement Loop** (PENDING)

---

## 2. Technology Requirements

- **Python**: 3.11 (`3.11.9` verified)
- **Environment**: Cross-platform (Windows PowerShell / POSIX bash)
- **Dependencies**: `rank-bm25`, `streamlit`, `requests`, `huggingface_hub`, `PyYAML`, `tqdm`, `pytest`

---

## 3. Environment Setup & Dependency Installation

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

## 4. End-to-End Workflow & Commands

### 1. Download Official Dataset
Downloads the raw CUAD benchmark and corpus files from Hugging Face:
```bash
python scripts/download_dataset.py
```

### 2. Prepare Dataset & Manifest
Selects the 194-query mini benchmark, discovers referenced contracts, verifies all 272 spans, assigns train/test splits, and freezes unanswerable queries:
```bash
python scripts/prepare_dataset.py
```

### 3. Validate Dataset Integrity
Validates character-level offsets, disjoint splits, and negative query schema:
```bash
python scripts/validate_dataset.py
```

### 4. Build the BM25 Index
Chunks the 29 contracts into 1,874 passages and fits the BM25 index:
```bash
python scripts/build_index.py
```

### 5. Run the Command-Line Interface (CLI)
Search for clauses directly from the terminal:
```bash
# Human-readable formatted search
python cli.py search --query "Does this contract permit sublicensing?" --top-k 5

# Machine-readable JSON output
python cli.py search --query "Does this contract permit sublicensing?" --top-k 5 --format json
```

### 6. Launch the Streamlit Web Application
Starts the interactive legal clause retrieval interface:
```bash
python -m streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### 7. Run Baseline Evaluation
Evaluates character-level Precision@K and Recall@K on the training split:
```bash
python scripts/evaluate_baseline.py --config configs/baseline.yaml --split train
```
Results are saved to `experiments/baseline_train_metrics.json`.

### 8. Run Automated Tests
Execute the complete test suite (Phase 1 + Phase 2):
```bash
pytest -v
```

---

## 5. Directory Structure

```text
jiostar-contract-intelligence-assistant/
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py           # Streamlit web application
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Project-root aware YAML configuration loader
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py            # HF dataset acquisition & provenance recording
│   │   ├── mini_selector.py       # Prescribed deterministic mini selection algorithm
│   │   ├── splitter.py            # Deterministic SHA-256 train/test query splitting
│   │   ├── loader.py              # Safe corpus discovery & contract manifest generation
│   │   ├── span_validator.py      # Character-offset slice verification
│   │   ├── negative_queries.py    # Unanswerable query dataset builder & validator
│   │   └── prepare.py             # End-to-end data preparation pipeline
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── chunker.py             # Character-offset sliding window chunker
│   │   ├── tokenizer.py           # Legal text tokenizer preserving section numbers
│   │   ├── bm25.py                # BM25 Okapi retriever
│   │   ├── ranker.py              # Candidate ranker with stable tie-breaking
│   │   ├── abstention.py          # Score thresholding abstention engine
│   │   └── pipeline.py            # Unified retrieval pipeline interface
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Character-level precision/recall & abstention metrics
│   │   └── baseline.py            # Baseline evaluation runner for train split
│   │
│   └── improvement/
│       └── __init__.py            # Phase 3 self-improvement placeholder
│
├── configs/
│   ├── data.yaml                  # Dataset paths and benchmark configuration
│   └── baseline.yaml              # BM25 baseline retrieval and evaluation parameters
│
├── data/
│   ├── raw/                       # Downloaded benchmark and corpus (.gitignored)
│   ├── processed/                 # Processed mini dataset and manifest (.gitignored)
│   └── unanswerable_queries.json  # Frozen 32 unanswerable negative queries
│
├── experiments/
│   ├── .gitkeep
│   └── baseline_train_metrics.json # Measured baseline evaluation results
│
├── scripts/
│   ├── download_dataset.py        # Dataset downloader
│   ├── prepare_dataset.py         # Pipeline executor
│   ├── validate_dataset.py        # Integrity validator
│   ├── inspect_dataset.py         # Metrics inspector
│   ├── build_index.py             # Index builder
│   ├── evaluate_baseline.py       # Baseline evaluator
│   └── inspect_retrieval.py       # Interactive retrieval inspection utility
│
├── tests/
│   ├── test_mini_selector.py
│   ├── test_splitter.py
│   ├── test_span_validator.py
│   ├── test_data_loader.py
│   ├── test_negative_queries.py
│   ├── test_preparation.py
│   ├── test_chunker.py
│   ├── test_tokenizer.py
│   ├── test_bm25.py
│   ├── test_abstention.py
│   ├── test_retrieval_pipeline.py
│   └── test_retrieval_metrics.py
│
├── docs/
│   └── DATASET.md                 # Dataset provenance and schema specifications
├── cli.py                         # Unified CLI search interface
├── .gitignore
├── .env.example
├── requirements.txt
├── README.md
├── REPORT.md
└── pyproject.toml
```

---

## 6. Baseline Performance Summary (Train Split)

Measured results on 140 answerable queries + 19 unanswerable queries:

| Metric | Measured Baseline Value |
|---|---|
| **Indexed Contracts** | 29 agreements |
| **Indexed Chunks** | 1,874 passages (800 chars, 150 overlap) |
| **Character Recall@1** | 0.0037 |
| **Character Recall@3** | 0.0310 |
| **Character Recall@5** | 0.1169 |
| **Character Recall@10** | 0.2318 |
| **Character Precision@5** | 0.0141 |
| **Uncalibrated Abstention ($\tau=0.0$)** | 0.0% Correct Abstention on Negatives, 100.0% Answerable Coverage |
| **Calibrated Abstention ($\tau=44.0$)** | 100.0% Correct Abstention on Negatives, 100.0% Answerable Coverage |

---

## 7. AI Coding Tool Usage Disclosure

In compliance with project guidelines, this project was developed with assistance from **Google Antigravity** (Advanced Agentic AI Assistant). All code, mathematical metric calculations, test suites, and empirical evaluation statistics were independently verified through automated terminal execution and real logs.
