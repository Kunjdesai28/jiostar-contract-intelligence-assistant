"""Deterministic train/test split assignment using SHA-256 hash thresholding."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def assign_split(query: str) -> str:
    """Prescribed deterministic train/test splitting function.

    Computes SHA-256 of whitespace-stripped query string.
    Returns 'test' if the resulting integer modulo 10 is < 3, else 'train'.

    DO NOT MODIFY THIS ALGORITHM OR THRESHOLD.
    """
    h = int(
        hashlib.sha256(
            query.strip().encode("utf-8")
        ).hexdigest(),
        16
    )
    return "test" if h % 10 < 3 else "train"


def split_benchmark(
    tests: List[Dict[str, Any]],
    train_output_path: Optional[Path | str] = None,
    test_output_path: Optional[Path | str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Partition benchmark records into train and test splits deterministically.

    Preserves all original record data, including query text and ground-truth snippets.
    Verifies that every record belongs to exactly one split, with no lost or duplicated records.

    Args:
        tests: List of benchmark test dictionaries.
        train_output_path: Optional destination path for train_queries.json.
        test_output_path: Optional destination path for test_queries.json.

    Returns:
        Tuple of (train_records, test_records, stats_dict).
    """
    train_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    for t in tests:
        q = t["query"]
        split = assign_split(q)

        # Clone record and attach split field explicitly
        record = dict(t)
        record["split"] = split

        if split == "train":
            train_records.append(record)
        elif split == "test":
            test_records.append(record)
        else:
            raise ValueError(f"Unexpected split value '{split}' for query: {q}")

    # Integrity verification
    total_in = len(tests)
    total_out = len(train_records) + len(test_records)
    if total_in != total_out:
        raise ValueError(f"Partition record count mismatch! Input: {total_in}, Output: {total_out}")

    train_queries = {t["query"] for t in train_records}
    test_queries = {t["query"] for t in test_records}
    overlap = train_queries.intersection(test_queries)
    if overlap:
        raise ValueError(f"Critical error: Overlap detected between train and test queries: {overlap}")

    stats = {
        "total_queries": total_in,
        "train_count": len(train_records),
        "test_count": len(test_records),
        "train_fraction": round(len(train_records) / total_in, 4) if total_in > 0 else 0.0,
        "test_fraction": round(len(test_records) / total_in, 4) if total_in > 0 else 0.0,
    }

    logger.info(f"Split complete: {stats['train_count']} train queries ({stats['train_fraction']*100:.1f}%), "
                f"{stats['test_count']} test queries ({stats['test_fraction']*100:.1f}%)")

    # Serialize outputs deterministically if paths provided
    if train_output_path:
        out_train = Path(train_output_path)
        out_train.parent.mkdir(parents=True, exist_ok=True)
        serialized_train = json.dumps({"split": "train", "tests": train_records}, indent=2, ensure_ascii=False)
        out_train.write_text(serialized_train, encoding="utf-8")
        stats["train_sha256"] = hashlib.sha256(serialized_train.encode("utf-8")).hexdigest()

    if test_output_path:
        out_test = Path(test_output_path)
        out_test.parent.mkdir(parents=True, exist_ok=True)
        serialized_test = json.dumps({"split": "test", "tests": test_records}, indent=2, ensure_ascii=False)
        out_test.write_text(serialized_test, encoding="utf-8")
        stats["test_sha256"] = hashlib.sha256(serialized_test.encode("utf-8")).hexdigest()

    return train_records, test_records, stats
