# LegalBench-RAG-mini CUAD Dataset Specification & Provenance

This document provides technical documentation of dataset provenance, selection algorithms, partitioning schemes, and schemas used in the JioStar Contract Intelligence Assistant.

---

## 1. Upstream Benchmark Provenance

- **Repository**: [awinml/legalbench-rag](https://huggingface.co/datasets/awinml/legalbench-rag)
- **Official Paper**: *LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in Legal Reasoning* ([arXiv:2408.10343](https://arxiv.org/abs/2408.10343))
- **Verified Commit SHA**: `2b9c248bc8179ef0908fd6ba01d50b156facd48b`
- **License**: Public legal benchmark / CC-BY / CUAD License

---

## 2. Deterministic Mini Selection Logic

The full CUAD benchmark in LegalBench-RAG contains 4,042 test queries. The official evaluation prescribes a 194-query mini benchmark selected deterministically using the following function:

```python
import json
import random
from pathlib import Path

def mini(tests, n=194):
    if len(tests) <= n:
        return tests

    def key(t):
        random.seed(t["snippets"][0]["file_path"])
        return random.random()

    return sorted(tests, key=key)[:n]
```

### Deterministic Invariants
- Input records: 4,042
- Output records: 194
- Duplicate queries: 0
- Mini Benchmark Checksum (SHA-256): `412fc88d91b8299defa7158a5bd2ee008e288af0332a5dfd39b8d44a83a61749`

---

## 3. Train / Test Splitting Protocol

Splits are determined strictly via SHA-256 hash integer thresholding on whitespace-stripped query strings:

```python
import hashlib

def assign_split(query: str) -> str:
    h = int(
        hashlib.sha256(
            query.strip().encode("utf-8")
        ).hexdigest(),
        16
    )
    return "test" if h % 10 < 3 else "train"
```

### Split Distribution
- **Train Split**: 140 queries (72.2%)
- **Test Split**: 54 queries (27.8%)
- **Integrity**: Disjoint, zero record loss, reproducible across environments.

---

## 4. Indexed Contract Corpus Scope

To ensure search efficiency and prevent irrelevant distractors, only contracts directly referenced by the 194 mini queries are indexed:
- **Total Unique Contracts**: 29 agreements
- **Corpus Size**: 1,211,773 UTF-8 characters
- **Manifest Location**: `data/processed/contract_manifest.json`

Each manifest entry contains:
```json
{
  "doc_id": "DOC_001_ALCOSTORESINC_12_14_2005-EX-10.26-AGENCY AGREEMENT",
  "benchmark_relative_path": "cuad/ALCOSTORESINC_12_14_2005-EX-10.26-AGENCY AGREEMENT.txt",
  "local_file_path": ".../data/raw/corpus/cuad/ALCOSTORESINC_12_14_2005-EX-10.26-AGENCY AGREEMENT.txt",
  "char_length": 45892,
  "content_sha256": "..."
}
```

---

## 5. Ground-Truth Character Span Verification

Every annotated ground-truth snippet in `data/processed/cuad_mini.json` was validated against the local corpus file using direct slicing:
$$\text{document}[\text{span}[0]:\text{span}[1]] == \text{answer}$$

- Total snippets: **272**
- Valid snippets: **272** (100%)
- Invalid snippets: **0**
- Missing files: **0**

---

## 6. Unanswerable Query Benchmark

Located at `data/unanswerable_queries.json`, this dataset consists of 32 curated negative queries across 6 categories:
1. `outside_indexed_corpus` (6 queries)
2. `absent_from_contracts` (6 queries)
3. `future_events_not_established` (5 queries)
4. `external_business_data` (4 queries)
5. `unsupported_legal_facts` (6 queries)
6. `out_of_domain` (5 queries)

Split distribution (assigned via `assign_split()`):
- Train: 19 queries
- Test: 13 queries
- Checksum: `7eaf304f4a22a4567100fb283a53f1bfc7b458eaa2f7feb7d78ba66db3d539d3`
