"""Automated Submission Readiness Verification Script.

Performs comprehensive validation of artifacts, checksums, integrity,
and invariants without re-running the held-out test evaluation.

Usage:
    python scripts/verify_submission.py
"""

import hashlib
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.pipeline import ContractRetrievalPipeline

FROZEN_NEGATIVE_CHECKSUM = "7eaf304f4a22a4567100fb283a53f1bfc7b458eaa2f7feb7d78ba66db3d539d3"
FROZEN_CONFIG_CHECKSUM = "6448998c99d0c236dd0644dd3abd6c62b679a0749187b3ce8f14ff210caa7718"


def main():
    print("=" * 70)
    print("JIOSTAR CONTRACT INTELLIGENCE ASSISTANT — SUBMISSION VERIFICATION")
    print("=" * 70)

    checks = []

    def record_check(name: str, passed: bool, detail: str = ""):
        status_str = "[PASS]" if passed else "[FAIL]"
        checks.append((name, passed, detail))
        print(f"{status_str} {name}")
        if detail:
            print(f"       -> {detail}")

    # 1. Required Files Exist
    required_files = [
        "README.md",
        "REPORT.md",
        "cli.py",
        "requirements.txt",
        "configs/final.yaml",
        "configs/baseline.yaml",
        "configs/final_phase3.yaml",
        "docs/FINAL_EVALUATION.md",
        "docs/PHASE4_FREEZE.md",
        "docs/SUBMISSION_EMAIL.md",
        "docs/SUBMISSION_CHECKLIST.md",
        "experiments/baseline_train_metrics.json",
        "experiments/final_test_metrics.json",
        "experiments/improvement_log.jsonl",
        "experiments/improvement_summary.json",
    ]
    missing = [f for f in required_files if not Path(f).exists()]
    record_check("Required files existence", len(missing) == 0, f"Missing: {missing}" if missing else "All 15 key files present")

    # 2. Frozen Configuration Checksum
    cfg_path = Path("configs/final.yaml")
    if cfg_path.exists():
        cfg_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        record_check(
            "Final configuration checksum",
            cfg_hash == FROZEN_CONFIG_CHECKSUM,
            f"Expected: {FROZEN_CONFIG_CHECKSUM[:16]}... | Actual: {cfg_hash[:16]}..."
        )
    else:
        record_check("Final configuration checksum", False, "configs/final.yaml not found")

    # 3. Frozen Negative Queries Checksum
    neg_path = Path("data/unanswerable_queries.json")
    if neg_path.exists():
        neg_hash = hashlib.sha256(neg_path.read_bytes()).hexdigest()
        record_check(
            "Frozen negative queries checksum",
            neg_hash == FROZEN_NEGATIVE_CHECKSUM,
            f"Expected: {FROZEN_NEGATIVE_CHECKSUM[:16]}... | Actual: {neg_hash[:16]}..."
        )
    else:
        record_check("Frozen negative queries checksum", False, "data/unanswerable_queries.json not found")

    # 4. Dataset Counts
    try:
        train_pos = json.load(open("data/processed/train_queries.json", encoding="utf-8"))["tests"]
        test_pos = json.load(open("data/processed/test_queries.json", encoding="utf-8"))["tests"]
        all_negs = json.load(open("data/unanswerable_queries.json", encoding="utf-8"))
        train_neg = [q for q in all_negs if q.get("split") == "train"]
        test_neg = [q for q in all_negs if q.get("split") == "test"]

        counts_ok = (len(train_pos) == 140 and len(test_pos) == 54 and len(train_neg) == 19 and len(test_neg) == 13)
        record_check(
            "Dataset sample counts",
            counts_ok,
            f"Train: {len(train_pos)} pos / {len(train_neg)} neg | Test: {len(test_pos)} pos / {len(test_neg)} neg"
        )
    except Exception as e:
        record_check("Dataset sample counts", False, f"Error: {e}")

    # 5. Improvement Loop Completeness
    imp_log_path = Path("experiments/improvement_log.jsonl")
    if imp_log_path.exists():
        lines = [line.strip() for line in imp_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        record_check("Improvement loop logged iterations", len(lines) >= 5, f"{len(lines)} iterations logged in improvement_log.jsonl")
    else:
        record_check("Improvement loop logged iterations", False, "improvement_log.jsonl not found")

    # 6. Final Test Result Artifact Integrity
    test_res_path = Path("experiments/final_test_metrics.json")
    if test_res_path.exists():
        try:
            test_data = json.load(open(test_res_path, encoding="utf-8"))
            has_split = test_data.get("split") == "test"
            has_cfg = test_data.get("configuration_sha256") == FROZEN_CONFIG_CHECKSUM
            has_metrics = "recall@5" in test_data.get("character_level_retrieval_metrics", {})
            test_valid = has_split and has_cfg and has_metrics
            record_check(
                "Final test result artifact integrity",
                test_valid,
                f"Recall@5 = {test_data.get('character_level_retrieval_metrics', {}).get('recall@5')}, "
                f"Negative Abstentions = {test_data.get('abstention_metrics', {}).get('correct_negative_abstentions')}"
            )
        except Exception as e:
            record_check("Final test result artifact integrity", False, f"JSON parse error: {e}")
    else:
        record_check("Final test result artifact integrity", False, "final_test_metrics.json not found")

    # 7. Substring Offset Verification
    try:
        pipeline = ContractRetrievalPipeline(config_path="configs/final.yaml")
        res = pipeline.search("What is the governing law of this agreement?", top_k=3)
        all_match = True
        for p in res.get("passages", []):
            doc = pipeline.contracts[p["document"]]["text"]
            sliced = doc[p["start_char"]:p["end_char"]]
            if sliced != p["text"]:
                all_match = False
                break
        record_check("Exact passage offset integrity", all_match, "100% exact substring match verified against raw files")
    except Exception as e:
        record_check("Exact passage offset integrity", False, f"Retrieval error: {e}")

    # Summary
    all_passed = all(p for _, p, _ in checks)
    print("=" * 70)
    if all_passed:
        print("RESULT: ALL 7 SUBMISSION INTEGRITY CHECKS PASSED.")
        print("The repository satisfies all technical prerequisites for submission.")
    else:
        print("RESULT: INTEGRITY CHECKS FAILED. Please review failed checks above.")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
