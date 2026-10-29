"""Unanswerable queries dataset generation, validation, and freezing utility."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.data.splitter import assign_split

logger = logging.getLogger(__name__)

# Curated, manually reasoned benchmark of 32 unanswerable questions spanning 6 categories
NEGATIVE_QUERY_DEFINITIONS: List[Dict[str, str]] = [
    # Category 1: outside_indexed_corpus
    {
        "id": "NEG_001",
        "query": "What are the broadcast syndication rights granted to JioStar for Indian Premier League cricket matches?",
        "category": "outside_indexed_corpus",
        "rationale": "The indexed corpus contains SEC EDGAR filings for US entities; no JioStar or sports broadcasting agreements are included.",
    },
    {
        "id": "NEG_002",
        "query": "What is the royalty percentage paid by Netflix to Warner Bros for digital streaming rights?",
        "category": "outside_indexed_corpus",
        "rationale": "Neither Netflix nor Warner Bros streaming license agreements are present in the 29 indexed contracts.",
    },
    {
        "id": "NEG_003",
        "query": "Under Apple's supplier agreement, what is Foxconn's defect tolerance rate for iPhone assembly?",
        "category": "outside_indexed_corpus",
        "rationale": "Apple and Foxconn manufacturing agreements are entirely absent from the indexed contract collection.",
    },
    {
        "id": "NEG_004",
        "query": "What are Microsoft's service level agreement commitments for Azure OpenAI uptime guarantee?",
        "category": "outside_indexed_corpus",
        "rationale": "Microsoft Azure enterprise cloud agreements are not part of the indexed contract corpus.",
    },
    {
        "id": "NEG_005",
        "query": "What are Pfizer's revenue-sharing obligations with BioNTech for COVID-19 vaccine distribution in Europe?",
        "category": "outside_indexed_corpus",
        "rationale": "Pfizer/BioNTech vaccine commercialization contracts are outside the indexed corpus.",
    },
    {
        "id": "NEG_006",
        "query": "What are the exclusivity terms governing Spotify's podcast distribution agreement with Joe Rogan?",
        "category": "outside_indexed_corpus",
        "rationale": "Celebrity podcast distribution agreements for Spotify are not present in the indexed contracts.",
    },
    # Category 2: absent_from_contracts (post-execution operational metrics)
    {
        "id": "NEG_007",
        "query": "What was the exact dollar amount of endorsement fees paid to Shaquille O'Neal by Papa John's in calendar year 2021?",
        "category": "absent_from_contracts",
        "rationale": "The Papa John's contract establishes endorsement fee terms and stock grant formulas, but actual realized post-signing 2021 payment totals are operational accounting records absent from the contract.",
    },
    {
        "id": "NEG_008",
        "query": "How many tons of chemical proppant did Flotek Industries actually deliver to customers during the fourth quarter of 2020?",
        "category": "absent_from_contracts",
        "rationale": "The Flotek supply agreement defines supply terms and pricing mechanisms, not historical post-execution delivery volume figures.",
    },
    {
        "id": "NEG_009",
        "query": "What was the total repair cost incurred by Holiday RV Superstores for vehicle maintenance in fiscal year 2003?",
        "category": "absent_from_contracts",
        "rationale": "Internal operational expenditures incurred post-signing are not contained within the endorsement contract text.",
    },
    {
        "id": "NEG_010",
        "query": "What were the actual quarterly advertising impressions delivered by Deltathree under the co-branding agreement?",
        "category": "absent_from_contracts",
        "rationale": "The Deltathree agreement outlines co-branding obligations and revenue shares, but does not record post-execution web traffic or ad impression statistics.",
    },
    {
        "id": "NEG_011",
        "query": "How many dental clinics joined the Union Dental Holdings network following the execution of the affiliate agreement?",
        "category": "absent_from_contracts",
        "rationale": "Subsequent affiliate enrollment numbers are operational business metrics not recorded in the static agreement.",
    },
    {
        "id": "NEG_012",
        "query": "What was the total volume of natural gas transported by PennTex Midstream Partners through the Arcadia plant in 2017?",
        "category": "absent_from_contracts",
        "rationale": "Operational pipeline throughput volumes post-signing are not established in the static transportation contract text.",
    },
    # Category 3: future_events_not_established
    {
        "id": "NEG_013",
        "query": "Did any party to the Veoneer joint venture agreement file for Chapter 11 bankruptcy reorganization in 2023?",
        "category": "future_events_not_established",
        "rationale": "The contract defines bankruptcy as an event of default, but subsequent external filings in 2023 cannot be determined from the contract text.",
    },
    {
        "id": "NEG_014",
        "query": "Was the Bon-Ton Stores liquidation agency agreement terminated early due to court intervention in 2019?",
        "category": "future_events_not_established",
        "rationale": "Subsequent court proceedings and early termination events post-dating execution are not recorded in the agreement.",
    },
    {
        "id": "NEG_015",
        "query": "Did Cano Petroleum successfully renew its sponsorship agreement upon expiration of the primary term in 2012?",
        "category": "future_events_not_established",
        "rationale": "The agreement outlines renewal options, but whether the parties exercised those options in 2012 is external historical fact.",
    },
    {
        "id": "NEG_016",
        "query": "Did Western Copper and Gold Corporation ever initiate formal arbitration against the covenantor under the non-competition agreement?",
        "category": "future_events_not_established",
        "rationale": "Post-execution dispute filings and arbitration outcomes are external events not recorded in the document.",
    },
    {
        "id": "NEG_017",
        "query": "Did Loop Industries achieve commercial PET recycling production milestones before December 2022 under its marketing agreement?",
        "category": "future_events_not_established",
        "rationale": "Whether commercial targets were achieved years after execution is an external operational outcome absent from the contract.",
    },
    # Category 4: external_business_data
    {
        "id": "NEG_018",
        "query": "What was the closing stock price of Dynamex Inc. on NASDAQ on June 6, 1996?",
        "category": "external_business_data",
        "rationale": "Public stock market closing prices are external market data not contained in commercial contract texts.",
    },
    {
        "id": "NEG_019",
        "query": "What credit rating was assigned to King Pharmaceuticals by Moody's at the time the promotion agreement was signed?",
        "category": "external_business_data",
        "rationale": "External rating agency credit opinions are not stated within the promotion agreement.",
    },
    {
        "id": "NEG_020",
        "query": "What was the market share of Vitamin Shoppe in the online nutritional supplement retail sector in 1999?",
        "category": "external_business_data",
        "rationale": "Macroeconomic and sector market share figures are external business data absent from the sponsorship contract.",
    },
    {
        "id": "NEG_021",
        "query": "Who succeeded as Chief Executive Officer of Sucampo Pharmaceuticals when the strategic alliance agreement expired?",
        "category": "external_business_data",
        "rationale": "Executive leadership succession history is corporate governance data not established in the agreement.",
    },
    # Category 5: unsupported_legal_facts (plausible legal queries absent from specific contracts)
    {
        "id": "NEG_022",
        "query": "What is the penalty interest rate if Duos Technologies fails to comply with European Union GDPR data erasure requests?",
        "category": "unsupported_legal_facts",
        "rationale": "The 2009 Duos Technologies agreement predates the GDPR regulation and contains no GDPR compliance or penalty clauses.",
    },
    {
        "id": "NEG_023",
        "query": "What are the liquidated damages specified if Bicycle Therapeutics breaches clinical trial participant privacy under HIPAA?",
        "category": "unsupported_legal_facts",
        "rationale": "The Bicycle Therapeutics agreement is governed by English law and contains no liquidated damages clause for US HIPAA violations.",
    },
    {
        "id": "NEG_024",
        "query": "What specific royalty rate is payable by Glu Mobile for exploiting licensed content in virtual reality metaverse applications?",
        "category": "unsupported_legal_facts",
        "rationale": "The 2007 Glu Mobile content license agreement does not contemplate or define virtual reality metaverse rights.",
    },
    {
        "id": "NEG_025",
        "query": "Under the Arc Group sponsorship agreement, what liquidated damages apply if a hurricane cancels the sponsored event?",
        "category": "unsupported_legal_facts",
        "rationale": "The agreement includes standard force majeure excuse provisions but specifies no liquidated damages payable for weather cancellations.",
    },
    {
        "id": "NEG_026",
        "query": "What is the required severance compensation if Theravance Biopharma terminates an executive director without cause?",
        "category": "unsupported_legal_facts",
        "rationale": "The Theravance Biopharma contract in the corpus is a corporate service agreement, not an executive employment agreement.",
    },
    {
        "id": "NEG_027",
        "query": "What is the required margin deposit if Rgc Resources experiences a credit downgrade below investment grade?",
        "category": "unsupported_legal_facts",
        "rationale": "The franchise agreement with municipal authorities contains no collateral margin or trading credit downgrade terms.",
    },
    # Category 6: out_of_domain
    {
        "id": "NEG_028",
        "query": "What is the average annual rainfall in the Amazon rainforest basin?",
        "category": "out_of_domain",
        "rationale": "Geographical and meteorological data has no relation to commercial legal contracts.",
    },
    {
        "id": "NEG_029",
        "query": "Which national team won the FIFA Men's World Cup football tournament in 2022?",
        "category": "out_of_domain",
        "rationale": "International sports tournament records are unrelated to the indexed contract collection.",
    },
    {
        "id": "NEG_030",
        "query": "What is the chemical formula and molecular weight of acetylsalicylic acid?",
        "category": "out_of_domain",
        "rationale": "Chemical formula and molecular properties are outside the contract retrieval domain.",
    },
    {
        "id": "NEG_031",
        "query": "What is the airspeed velocity of an unladen European swallow?",
        "category": "out_of_domain",
        "rationale": "Pop-culture trivia and ornithological questions are completely outside the legal contract domain.",
    },
    {
        "id": "NEG_032",
        "query": "What is the recommended oven temperature and cooking time for traditional Neapolitan pizza?",
        "category": "out_of_domain",
        "rationale": "Culinary recipes and baking instructions are entirely unsupported by corporate agreements.",
    },
]


def build_unanswerable_dataset() -> List[Dict[str, Any]]:
    """Construct full unanswerable query records with deterministic splits and verification metadata."""
    dataset: List[Dict[str, Any]] = []
    for item in NEGATIVE_QUERY_DEFINITIONS:
        q = item["query"]
        split = assign_split(q)
        record = {
            "id": item["id"],
            "query": q,
            "answerable": False,
            "split": split,
            "category": item["category"],
            "rationale": item["rationale"],
            "verification_status": "verified_unanswerable",
        }
        dataset.append(record)
    return dataset


def validate_negative_queries(
    records: List[Dict[str, Any]] | Path | str,
    min_count: int = 30,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate negative query dataset schema, determinism, and answerability invariants.

    Checks:
      1. At least min_count records (>= 30).
      2. Unique IDs.
      3. Non-empty query string.
      4. answerable is explicitly False.
      5. split matches assign_split(query) exactly.
      6. category is specified and valid.
      7. rationale is non-empty.
      8. verification_status is 'verified_unanswerable' or 'pending_review'.

    Args:
        records: List of record dicts or path to JSON file.
        min_count: Minimum required questions (default 30).

    Returns:
        Tuple of (is_valid, validation_report).
    """
    if isinstance(records, (Path, str)):
        p = Path(records)
        if not p.exists():
            return False, {"error": f"Negative queries file not found: {p}"}
        records = json.loads(p.read_text(encoding="utf-8"))

    if not isinstance(records, list):
        return False, {"error": "Negative queries dataset must be a list of records."}

    errors: List[str] = []
    seen_ids = set()
    train_count = 0
    test_count = 0
    categories: Dict[str, int] = {}
    pending_review: List[str] = []

    if len(records) < min_count:
        errors.append(f"Insufficient negative queries: found {len(records)}, required at least {min_count}.")

    for idx, r in enumerate(records):
        qid = r.get("id")
        if not qid:
            errors.append(f"Record {idx} missing 'id'.")
        elif qid in seen_ids:
            errors.append(f"Duplicate id '{qid}' at record {idx}.")
        else:
            seen_ids.add(qid)

        q = r.get("query")
        if not q or not isinstance(q, str) or not q.strip():
            errors.append(f"Record {qid or idx} has empty or non-string query.")
            continue

        if r.get("answerable") is not False:
            errors.append(f"Record {qid} must have 'answerable' set to False, got: {r.get('answerable')}")

        expected_split = assign_split(q)
        actual_split = r.get("split")
        if actual_split != expected_split:
            errors.append(f"Record {qid} split mismatch: expected '{expected_split}', got '{actual_split}'.")

        if actual_split == "train":
            train_count += 1
        elif actual_split == "test":
            test_count += 1

        cat = r.get("category", "unspecified")
        categories[cat] = categories.get(cat, 0) + 1

        rat = r.get("rationale")
        if not rat or not isinstance(rat, str) or not rat.strip():
            errors.append(f"Record {qid} has missing or empty rationale.")

        v_status = r.get("verification_status")
        if v_status == "pending_review":
            pending_review.append(qid)
        elif v_status != "verified_unanswerable":
            errors.append(f"Record {qid} has invalid verification_status '{v_status}'.")

    is_valid = len(errors) == 0

    report = {
        "is_valid": is_valid,
        "total_queries": len(records),
        "train_count": train_count,
        "test_count": test_count,
        "categories": categories,
        "pending_review_count": len(pending_review),
        "pending_review_ids": pending_review,
        "error_count": len(errors),
        "errors": errors,
    }

    return is_valid, report


def freeze_negative_queries(
    output_path: Path | str,
    overwrite_existing: bool = False,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """Generate, validate, and serialize unanswerable queries dataset to destination.

    Will not overwrite an existing frozen file unless overwrite_existing is True.

    Args:
        output_path: Destination path (e.g. data/unanswerable_queries.json).
        overwrite_existing: If False and file exists, reads and validates existing file.

    Returns:
        Tuple of (records, sha256_checksum, validation_report).
    """
    out_p = Path(output_path)

    if out_p.exists() and not overwrite_existing:
        logger.info(f"Existing unanswerable queries file found at {out_p}. Verifying integrity...")
        raw_text = out_p.read_text(encoding="utf-8")
        records = json.loads(raw_text)
        checksum = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        is_valid, report = validate_negative_queries(records)
        if not is_valid:
            raise ValueError(f"Existing negative queries failed validation: {report['errors']}")
        return records, checksum, report

    records = build_unanswerable_dataset()
    is_valid, report = validate_negative_queries(records)
    if not is_valid:
        raise ValueError(f"Generated negative queries failed validation: {report['errors']}")

    serialized = json.dumps(records, indent=2, ensure_ascii=False)
    checksum = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(serialized, encoding="utf-8")
    logger.info(f"Saved frozen negative queries to {out_p} (SHA-256: {checksum})")

    return records, checksum, report
