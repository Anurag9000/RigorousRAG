from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pytest

from tools.evidence_graph_claim_extraction import (
    extract_scientific_claim_proposals,
)


@dataclass
class Section:
    content: str
    page_number: int | None = None


@dataclass
class Document:
    id: str
    text: str
    sections: list[Section]
    metadata: dict


def document() -> Document:
    text = "Drug A reduced mortality in the randomized cohort. Further work is needed."
    return Document(
        id="doc1",
        text=text,
        sections=[Section(text, 1)],
        metadata={"content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()},
    )


def output(*, supersedes: str | None = None) -> dict:
    text = document().sections[0].content
    claim = {
        "claim_key": "claim-1" if supersedes is None else "claim-2",
        "claim_text": "Drug A reduced mortality.",
        "claim_type": "finding",
        "modality": "asserted",
        "section_index": 0,
        "page_number": 1,
        "char_start": text.index("Drug A"),
        "char_end": text.index(" in the"),
        "confidence": 0.9,
    }
    if supersedes is not None:
        claim["supersedes_proposal_id"] = supersedes
    return {"schema_version": 1, "claims": [claim]}


def extract(raw, *, now=1.0):
    return extract_scientific_claim_proposals(
        document(),
        raw,
        owner_id="alice",
        generation=7,
        profile_fingerprint="b" * 64,
        proposer_id="claim-extractor",
        extractor_name="scientific-claims",
        extractor_version="1.0",
        now=now,
    )


def test_extraction_is_deterministic_generation_bound_and_non_mutating():
    first = extract(output(), now=1.0)
    second = extract(output(), now=9.0)

    assert first.proposals[0].proposal_id == second.proposals[0].proposal_id
    assert first.batch_digest == second.batch_digest
    assert first.proposals[0].owner_id == "alice"
    assert first.proposals[0].doc_id == "doc1"
    assert first.proposals[0].generation == 7
    assert first.proposals[0].locator.evidence_sha256 == hashlib.sha256(
        b"Drug A reduced mortality"
    ).hexdigest()
    assert first.proposals[0].metadata["evidence_length"] == 24
    assert not hasattr(first, "graph_mutation_performed")


def test_extractor_json_is_closed_and_duplicate_keys_fail_closed():
    raw = output()
    raw["claims"][0]["unsupported"] = True
    with pytest.raises(ValueError, match="schema"):
        extract(raw)

    duplicate = (
        '{"schema_version":1,"schema_version":1,"claims":[]}'
    )
    with pytest.raises(ValueError, match="duplicate"):
        extract(duplicate)


def test_taxonomy_duplicate_claim_keys_and_nonfinite_confidence_refuse():
    raw = output()
    raw["claims"][0]["claim_type"] = "invented"
    with pytest.raises(ValueError, match="taxonomy"):
        extract(raw)

    raw = output()
    raw["claims"].append(dict(raw["claims"][0]))
    with pytest.raises(ValueError, match="duplicate claim keys"):
        extract(raw)

    raw = output()
    raw["claims"][0]["confidence"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        extract(raw)


def test_locator_page_and_content_identity_are_revalidated():
    raw = output()
    raw["claims"][0]["page_number"] = 2
    with pytest.raises(ValueError, match="page_number"):
        extract(raw)

    raw = output()
    raw["claims"][0]["char_end"] = 999
    with pytest.raises(ValueError, match="between"):
        extract(raw)

    doc = document()
    doc.metadata["content_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="content hash"):
        extract_scientific_claim_proposals(
            doc,
            output(),
            owner_id="alice",
            generation=7,
            profile_fingerprint="b" * 64,
            proposer_id="claim-extractor",
            extractor_name="scientific-claims",
            extractor_version="1.0",
        )


def test_correction_lineage_changes_identity_and_is_digest_bound():
    original = extract(output()).proposals[0]
    corrected = extract(output(supersedes=original.proposal_id), now=2.0).proposals[0]

    assert corrected.proposal_id != original.proposal_id
    assert corrected.supersedes_proposal_id == original.proposal_id
    assert corrected.proposal_digest != original.proposal_digest
