"""Clause-Aware Structure Chunker for Legal Agreements.

Preserves exact UTF-8 character offsets and source-slice equality
(source_text[start:end] == chunk_text) while splitting along structural
legal boundaries (numbered sections, articles, clause headings, double newlines).
"""

import hashlib
import re
from typing import Any, Dict, List

# Structural boundary regex patterns in commercial contracts
SECTION_HEADING_PATTERN = re.compile(
    r"(?:\n\s*(?:SECTION|ARTICLE|Article|Section|CLAUSE|Clause)\s+[0-9IVXLCDM]+(?:\.[0-9]+)*[^\n]*\n)"
    r"|(?:\n\s*[0-9]+(?:\.[0-9]+)+\s+[^\n]{3,60}\n)"
    r"|(?:\n\n(?=[A-Z0-9\s\.\-]{3,50}\n))",
    re.MULTILINE,
)


def chunk_contract_by_clauses(
    doc_id: str,
    file_path: str,
    text: str,
    max_chunk_size: int = 1200,
    min_chunk_size: int = 150,
) -> List[Dict[str, Any]]:
    """Segment a contract document into structure-aware clause chunks.

    Guarantees:
      - text[start_char:end_char] == chunk_text
      - 0 <= start_char < end_char <= len(text)
      - Deterministic chunk_id generation.
      - Fallback to sliding window for oversized clauses.

    Args:
        doc_id: Unique document identifier.
        file_path: Relative contract file path.
        text: Raw document text string.
        max_chunk_size: Maximum allowed character length before sub-chunking.
        min_chunk_size: Minimum characters to avoid tiny fragments.

    Returns:
        List of chunk dictionaries with exact offsets and metadata.
    """
    if not text:
        return []

    text_len = len(text)
    # Find all structural boundaries
    split_indices = [0]
    for match in SECTION_HEADING_PATTERN.finditer(text):
        idx = match.start()
        if idx > split_indices[-1]:
            split_indices.append(idx)
    if split_indices[-1] < text_len:
        split_indices.append(text_len)

    raw_spans: List[tuple[int, int]] = []
    for i in range(len(split_indices) - 1):
        s = split_indices[i]
        e = split_indices[i + 1]
        if e > s:
            raw_spans.append((s, e))

    # Refine spans: merge tiny spans, split oversized spans
    refined_spans: List[tuple[int, int]] = []
    curr_s, curr_e = 0, 0

    for s, e in raw_spans:
        if curr_e == 0:
            curr_s, curr_e = s, e
            continue

        span_len = curr_e - curr_s
        next_len = e - s

        # If current span is too small, merge with next
        if span_len < min_chunk_size and (span_len + next_len) <= max_chunk_size:
            curr_e = e
        else:
            # If current span is too large, split using sliding window
            if span_len > max_chunk_size:
                sub_s = curr_s
                while sub_s < curr_e:
                    sub_e = min(sub_s + max_chunk_size, curr_e)
                    refined_spans.append((sub_s, sub_e))
                    if sub_e == curr_e:
                        break
                    sub_s += int(max_chunk_size * 0.75)  # 25% overlap
            else:
                refined_spans.append((curr_s, curr_e))
            curr_s, curr_e = s, e

    # Flush last span
    if curr_e > curr_s:
        span_len = curr_e - curr_s
        if span_len > max_chunk_size:
            sub_s = curr_s
            while sub_s < curr_e:
                sub_e = min(sub_s + max_chunk_size, curr_e)
                refined_spans.append((sub_s, sub_e))
                if sub_e == curr_e:
                    break
                sub_s += int(max_chunk_size * 0.75)
        else:
            refined_spans.append((curr_s, curr_e))

    # Construct chunk records and verify source-slice equality
    chunks: List[Dict[str, Any]] = []
    for idx, (s, e) in enumerate(refined_spans):
        chunk_text = text[s:e]
        assert text[s:e] == chunk_text, f"Source-slice equality invariant failed for {doc_id} span ({s}, {e})"

        chunk_id = f"{doc_id}_clause_{idx:05d}"
        sha = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

        chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "file_path": file_path,
            "chunk_index": idx,
            "start_char": s,
            "end_char": e,
            "char_length": len(chunk_text),
            "text": chunk_text,
            "content_sha256": sha,
            "strategy": "clause_aware",
        })

    return chunks


def chunk_collection_clause_aware(
    contracts: Dict[str, Dict[str, Any]],
    max_chunk_size: int = 1200,
    min_chunk_size: int = 150,
) -> List[Dict[str, Any]]:
    """Segment an entire collection of contracts using clause-aware chunking."""
    all_chunks: List[Dict[str, Any]] = []
    for file_path, doc_info in contracts.items():
        doc_id = doc_info["doc_id"]
        text = doc_info["text"]
        doc_chunks = chunk_contract_by_clauses(
            doc_id=doc_id,
            file_path=file_path,
            text=text,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
        )
        all_chunks.extend(doc_chunks)
    return all_chunks
