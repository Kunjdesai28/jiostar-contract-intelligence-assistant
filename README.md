# JioStar Contract Intelligence Assistant

A production-grade Legal Agreement Retrieval and Character-Span Evidence Extraction System.

This project implements the **Phase 1: Project Foundation, Dataset Preparation, Git Setup, and Validation** milestones for legal contract intelligence using the official **LegalBench-RAG-mini CUAD** benchmark.

---

## 1. Project Overview & Phase 1 Scope

The objective of the Contract Intelligence Assistant is to retrieve relevant contractual clauses from complex corporate agreements (SEC EDGAR filings), extract exact character-offset spans as verifiable evidence, and reliably abstain when requested contractual evidence is not present.

### Phase 1 Status: COMPLETED
- [x] Portable, modular Python 3.11 project scaffold.
- [x] Official LegalBench-RAG CUAD benchmark and corpus acquisition with verified provenance.
- [x] Exact deterministic 194-query mini benchmark selection.
- [x] Deterministic SHA-256 train/test query splitting (140 train / 54 test).
- [x] Ground-truth character offset span validation across all 272 snippets (100% exact match).
- [x] Strict referenced-contract discovery and manifest generation (29 unique contracts, 1,211,773 characters).
- [x] Curated 32 unanswerable negative queries benchmark with balanced categorization and deterministic splits.
- [x] Automated test suite (18 unit, integration, and reproducibility tests passing).
- [x] Git version control initialization and negative query dataset freezing.

*Note: Phase 1 strictly excludes BM25 indexing, dense embeddings, FAISS, reranking, LLM integrations, and Streamlit interfaces, which are scheduled for Phase 2.*

---

## 2. Technology Requirements

- **Python**: 3.11 (`3.11.9` verified)
- **Environment**: Cross-platform (Windows PowerShell / POSIX bash)
- **Core Libraries**: `requests`, `huggingface_hub`, `PyYAML`, `tqdm`, `pytest`
- **Zero Heavyweight Retrieval/LLM Overhead**: No LangChain, LlamaIndex, or vector databases in Phase 1.

---

## 3. Environment Setup & Dependency Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/Kunjdesai28/jiostar-contract-intelligence-assistant.git
cd jiostar-contract-intelligence-assistant
```

### Step 2: Initialize Python 3.11 Virtual Environment
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
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Dataset Acquisition & Provenance

The system uses the peer-reviewed **LegalBench-RAG** benchmark ([arXiv:2408.10343](https://arxiv.org/abs/2408.10343)).

### Download Official CUAD Assets
Downloads strictly the CUAD components (`benchmarks/cuad.json` and `corpus/cuad/*.txt`) from the official Hugging Face mirror:
```bash
python scripts/download_dataset.py
```
- **Hugging Face Mirror**: `https://huggingface.co/datasets/awinml/legalbench-rag`
- **Pinned Commit SHA**: `2b9c248bc8179ef0908fd6ba01d50b156facd48b`
- **Storage Paths**:
  - Benchmark: `data/raw/benchmarks/cuad.json`
  - Corpus: `data/raw/corpus/cuad/` (462 text files)
  - Provenance: `data/raw/provenance.json`

---

## 5. Dataset Preparation & Validation

### Run Preparation Pipeline
Selects the deterministic 194-query mini subset, discovers contracts, validates ground-truth offsets, computes SHA-256 splits, and freezes unanswerable queries:
```bash
python scripts/prepare_dataset.py
```

### Run Validation Checks
Validates character-level span integrity, split determinism, and unanswerable query schema:
```bash
python scripts/validate_dataset.py
```

### Live Inspection Report
Displays live dataset metrics and cryptographic checksums computed from actual disk files:
```bash
python scripts/inspect_dataset.py
```

---

## 6. Project Directory Structure

```
jiostar-contract-intelligence-assistant/
│
├── app/
│   └── __init__.py                # Web application package (Phase 2/3)
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
│   ├── retrieval/                 # Retrieval package placeholder (Phase 2)
│   │   └── __init__.py
│   ├── evaluation/                # Evaluation metrics package placeholder (Phase 2/3)
│   │   └── __init__.py
│   └── improvement/               # Iterative improvement loop placeholder (Phase 3)
│       └── __init__.py
│
├── configs/
│   └── data.yaml                  # Core dataset paths and benchmark configurations
│
├── data/
│   ├── raw/                       # Downloaded benchmark and corpus (.gitignored)
│   │   └── .gitkeep
│   ├── processed/                 # Processed mini dataset and manifest (.gitignored)
│   │   └── .gitkeep
│   └── unanswerable_queries.json  # Frozen 32 unanswerable negative queries
│
├── scripts/
│   ├── download_dataset.py        # Dataset downloader
│   ├── prepare_dataset.py         # Pipeline executor
│   ├── validate_dataset.py        # Integrity validator
│   └── inspect_dataset.py         # Metrics inspector
│
├── tests/
│   ├── __init__.py
│   ├── fixtures/
│   ├── test_mini_selector.py      # Mini selection determinism & count tests
│   ├── test_splitter.py           # Split hash thresholding tests
│   ├── test_span_validator.py     # Synthetic & real ground-truth offset tests
│   ├── test_data_loader.py        # Path security & document loader tests
│   ├── test_negative_queries.py   # Unanswerable query invariant tests
│   └── test_preparation.py        # End-to-end pipeline reproducibility tests
│
├── experiments/                   # Experiment artifacts placeholder
│   └── .gitkeep
├── docs/
│   └── DATASET.md                 # Detailed dataset schema and provenance specification
├── .gitignore
├── .env.example
├── requirements.txt
├── README.md
├── REPORT.md
└── pyproject.toml
```

---

## 7. Automated Testing

Run the full pytest test suite:
```bash
pytest -v
```

All 18 automated tests cover:
- Selection determinism and preservation of the original 194 records.
- Exact SHA-256 split partitioning and disjointness.
- Synthetic span validation edge cases (negative start, out-of-bounds end, empty span, multiline text, unicode, mismatches).
- Real CUAD mini integration span validation (272/272 spans matching).
- Directory traversal security and contract deduplication.
- Unanswerable query validation and idempotency.
- End-to-end pipeline reproducibility and checksum stability across repeated runs.

---

## 8. Dataset Statistics Summary

| Metric | Measured Value | Notes |
|---|---|---|
| Full CUAD Benchmark Queries | 4,042 | Official LegalBench-RAG |
| Selected Mini Benchmark Queries | 194 | Deterministic seeded selection |
| Mini Train Queries | 140 (72.2%) | Deterministic SHA-256 split |
| Mini Test Queries | 54 (27.8%) | Deterministic SHA-256 split |
| Unique Referenced Contracts | 29 | Discovered from mini snippets |
| Total Contract Characters | 1,211,773 chars | Unstripped UTF-8 text |
| Total Ground-Truth Snippets | 272 | 100% exact substring match |
| Invalid Spans / Missing Docs | 0 / 0 | Full validation passed |
| Unanswerable Negative Queries | 32 | 19 train / 13 test (6 categories) |

---

## 9. Documented Assumptions & Guardrails

1. **Newline and Text Semantics**: Contracts are read in Python UTF-8 text mode without line-ending normalization or text stripping. This exactly matches the raw character indices in the benchmark.
2. **Corpus Scope**: Only contracts referenced by the selected 194 benchmark records (29 unique files) are indexed. The remaining 433 files in `corpus/cuad/` are not indexed to prevent corpus pollution.
3. **Train/Test Integrity**: The 54 test queries and 13 negative test queries are strictly reserved for final Phase 3 reporting and will never be accessed during retrieval tuning.
4. **Path Traversal Security**: All relative paths in snippet annotations are validated against the corpus root using `resolve_safe_corpus_path()` to prevent directory traversal attacks.

---

## 10. AI Coding Tool Usage Disclosure

In compliance with project guidelines, this implementation was assisted by **Google Antigravity** (Advanced Agentic AI Assistant). All generated code, tests, dataset hashes, and statistical distributions were independently validated through automated execution and real-world execution logs.
