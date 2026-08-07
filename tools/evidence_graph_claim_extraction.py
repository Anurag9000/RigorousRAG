"""Closed-schema adapter from reviewed extractor output to claim proposals only."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tools.evidence_graph_claim_contracts import (
    CLAIM_MODALITIES,
    CLAIM_TYPES,
    ClaimEvidenceLocator,
    ScientificClaimProposal,
    _digest,
    _identifier,
    _integer,
    _metadata,
    _sha256,
    _timestamp,
)
from tools.security import normalize_owner_id

_MAX_OUTPUT_BYTES = 5_000_000
_MAX_CLAIMS = 10_000
_ALLOWED_CLAIM_KEYS = frozenset(
    {
        "claim_key",
        "claim_text",
        "claim_type",
        "modality",
        "section_index",
        "page_number",
        "char_start",
        "char_end",
        "confidence",
        "supersedes_proposal_id",
        "metadata",
    }
)
_REQUIRED_CLAIM_KEYS = _ALLOWED_CLAIM_KEYS - {
    "page_number",
    "supersedes_proposal_id",
    "metadata",
}


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("extractor output contains a duplicate JSON key.")
        result[key] = value
    return result


def _parse_output(value: str | bytes | bytearray | Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if isinstance(value, Mapping):
        raw = dict(value)
        try:
            payload = json.dumps(
                raw,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except ValueError as exc:
            raise ValueError(
                "extractor output contains a non-finite or invalid JSON value."
            ) from exc
    else:
        payload = bytes(value) if isinstance(value, (bytes, bytearray)) else None
        if payload is None:
            if not isinstance(value, str):
                raise ValueError("extractor output must be JSON text, bytes or an object.")
            payload = value.encode("utf-8")
        if not 1 <= len(payload) <= _MAX_OUTPUT_BYTES:
            raise ValueError("extractor output size is invalid.")
        try:
            raw = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
        except ValueError as exc:
            if "duplicate JSON key" in str(exc):
                raise
            raise ValueError("extractor output JSON is invalid.") from exc
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("extractor output JSON is invalid.") from exc
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "claims"}:
        raise ValueError("extractor output schema is invalid.")
    if raw["schema_version"] != 1:
        raise ValueError("extractor output schema is unsupported.")
    claims = raw["claims"]
    if (
        isinstance(claims, (str, bytes, bytearray))
        or not isinstance(claims, Sequence)
        or not 1 <= len(claims) <= _MAX_CLAIMS
    ):
        raise ValueError("claims must be a bounded non-empty array.")
    return raw, hashlib.sha256(payload).hexdigest()


def _section_data(section: Any, index: int) -> tuple[str, int | None]:
    if hasattr(section, "model_dump") and callable(section.model_dump):
        raw = section.model_dump()
    elif isinstance(section, Mapping):
        raw = dict(section)
    else:
        raw = {
            "content": getattr(section, "content", None),
            "page_number": getattr(section, "page_number", None),
        }
    if not isinstance(raw, Mapping):
        raise ValueError(f"section {index} is not object-like.")
    content = raw.get("content")
    if not isinstance(content, str) or not content.strip() or len(content) > 5_000_000:
        raise ValueError(f"section {index} content is invalid.")
    page = raw.get("page_number")
    if page is not None:
        page = _integer(page, "section page_number", 1, 1_000_000)
    return content.strip(), page


@dataclass(frozen=True)
class ScientificClaimExtractionBatch:
    owner_id: str
    doc_id: str
    generation: int
    content_sha256: str
    profile_fingerprint: str
    extractor_name: str
    extractor_version: str
    proposer_id: str
    output_digest: str
    proposals: tuple[ScientificClaimProposal, ...]
    created_at: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id", 200))
        object.__setattr__(self, "generation", _integer(self.generation, "generation", 1, 2**63 - 1))
        object.__setattr__(self, "content_sha256", _digest(self.content_sha256, "content_sha256"))
        object.__setattr__(
            self,
            "profile_fingerprint",
            _digest(self.profile_fingerprint, "profile_fingerprint"),
        )
        object.__setattr__(self, "extractor_name", _identifier(self.extractor_name, "extractor_name", 200))
        object.__setattr__(
            self,
            "extractor_version",
            _identifier(self.extractor_version, "extractor_version", 200),
        )
        object.__setattr__(self, "proposer_id", _identifier(self.proposer_id, "proposer_id", 200))
        object.__setattr__(self, "output_digest", _digest(self.output_digest, "output_digest"))
        if (
            not isinstance(self.proposals, tuple)
            or not 1 <= len(self.proposals) <= _MAX_CLAIMS
            or any(not isinstance(value, ScientificClaimProposal) for value in self.proposals)
        ):
            raise ValueError("proposals must be a bounded non-empty tuple.")
        if len({value.proposal_id for value in self.proposals}) != len(self.proposals):
            raise ValueError("claim extraction batch contains duplicate proposal IDs.")
        for value in self.proposals:
            if (
                value.owner_id != self.owner_id
                or value.doc_id != self.doc_id
                or value.generation != self.generation
                or value.content_sha256 != self.content_sha256
                or value.profile_fingerprint != self.profile_fingerprint
                or value.extractor_name != self.extractor_name
                or value.extractor_version != self.extractor_version
                or value.proposer_id != self.proposer_id
            ):
                raise ValueError("claim proposal escaped extraction batch scope.")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        if self.schema_version != 1:
            raise ValueError("claim extraction batch schema is unsupported.")

    @property
    def batch_digest(self) -> str:
        return _sha256(
            {
                "scope": "rigorousrag-scientific-claim-extraction-batch-v1",
                "owner_id": self.owner_id,
                "doc_id": self.doc_id,
                "generation": self.generation,
                "content_sha256": self.content_sha256,
                "profile_fingerprint": self.profile_fingerprint,
                "extractor_name": self.extractor_name,
                "extractor_version": self.extractor_version,
                "proposer_id": self.proposer_id,
                "output_digest": self.output_digest,
                "proposal_digests": [value.proposal_digest for value in self.proposals],
            }
        )


def extract_scientific_claim_proposals(
    document: Any,
    extractor_output: str | bytes | bytearray | Mapping[str, Any],
    *,
    owner_id: str,
    generation: int,
    profile_fingerprint: str,
    proposer_id: str,
    extractor_name: str,
    extractor_version: str,
    now: float | None = None,
) -> ScientificClaimExtractionBatch:
    """Validate extractor output against exact finalized sections; perform no persistence."""

    owner = normalize_owner_id(owner_id)
    doc_id = _identifier(getattr(document, "id", None), "document.id", 200)
    text = getattr(document, "text", None)
    if not isinstance(text, str) or not text.strip() or len(text) > 50_000_000:
        raise ValueError("document.text is invalid.")
    finalized_text = text.strip()
    content_sha256 = hashlib.sha256(finalized_text.encode("utf-8")).hexdigest()
    metadata = getattr(document, "metadata", {})
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("document.metadata must be a mapping.")
    declared = (metadata or {}).get("content_sha256")
    if declared is not None and declared != content_sha256:
        raise ValueError("document content hash does not match finalized text.")

    raw_sections = getattr(document, "sections", None)
    if (
        isinstance(raw_sections, (str, bytes, bytearray))
        or not isinstance(raw_sections, Sequence)
        or not 1 <= len(raw_sections) <= 10_000
    ):
        raise ValueError("document must contain bounded finalized sections.")
    sections = tuple(_section_data(value, index) for index, value in enumerate(raw_sections))

    raw, output_digest = _parse_output(extractor_output)
    created_at = time.time() if now is None else _timestamp(now, "now")
    profile = _digest(profile_fingerprint, "profile_fingerprint")
    selected_proposer = _identifier(proposer_id, "proposer_id", 200)
    selected_name = _identifier(extractor_name, "extractor_name", 200)
    selected_version = _identifier(extractor_version, "extractor_version", 200)

    proposals: list[ScientificClaimProposal] = []
    claim_keys: set[str] = set()
    for index, claim in enumerate(raw["claims"]):
        if not isinstance(claim, Mapping) or not _REQUIRED_CLAIM_KEYS <= set(claim) <= _ALLOWED_CLAIM_KEYS:
            raise ValueError(f"claim {index} schema is invalid.")
        claim_key = _identifier(claim["claim_key"], "claim_key", 2_000)
        if claim_key in claim_keys:
            raise ValueError("extractor output contains duplicate claim keys.")
        claim_keys.add(claim_key)
        claim_type = _identifier(claim["claim_type"], "claim_type", 50)
        modality = _identifier(claim["modality"], "modality", 50)
        if claim_type not in CLAIM_TYPES or modality not in CLAIM_MODALITIES:
            raise ValueError("extractor claim taxonomy is unsupported.")
        section_index = _integer(claim["section_index"], "section_index", 0, len(sections) - 1)
        section_text, section_page = sections[section_index]
        char_start = _integer(claim["char_start"], "char_start", 0, len(section_text))
        char_end = _integer(claim["char_end"], "char_end", 1, len(section_text))
        if char_end <= char_start:
            raise ValueError("claim evidence span is empty.")
        evidence = section_text[char_start:char_end]
        if not evidence.strip():
            raise ValueError("claim evidence span contains no text.")
        page_number = claim.get("page_number")
        if page_number is not None:
            page_number = _integer(page_number, "page_number", 1, 1_000_000)
        if section_page is not None and page_number is not None and section_page != page_number:
            raise ValueError("claim page_number differs from finalized section provenance.")
        resolved_page = section_page if page_number is None else page_number
        locator = ClaimEvidenceLocator(
            section_index=section_index,
            page_number=resolved_page,
            char_start=char_start,
            char_end=char_end,
            evidence_sha256=hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
        )
        proposal_metadata = {
            "extractor_output_digest": output_digest,
            "evidence_length": len(evidence),
            **_metadata(claim.get("metadata")),
        }
        proposals.append(
            ScientificClaimProposal.create(
                owner_id=owner,
                doc_id=doc_id,
                generation=generation,
                content_sha256=content_sha256,
                profile_fingerprint=profile,
                claim_key=claim_key,
                claim_text=claim["claim_text"],
                claim_type=claim_type,
                modality=modality,
                locator=locator,
                proposer_kind="model",
                proposer_id=selected_proposer,
                extractor_name=selected_name,
                extractor_version=selected_version,
                confidence=claim["confidence"],
                supersedes_proposal_id=claim.get("supersedes_proposal_id"),
                metadata=proposal_metadata,
                created_at=created_at,
            )
        )
    return ScientificClaimExtractionBatch(
        owner_id=owner,
        doc_id=doc_id,
        generation=generation,
        content_sha256=content_sha256,
        profile_fingerprint=profile,
        extractor_name=selected_name,
        extractor_version=selected_version,
        proposer_id=selected_proposer,
        output_digest=output_digest,
        proposals=tuple(proposals),
        created_at=created_at,
    )


__all__ = [
    "ScientificClaimExtractionBatch",
    "extract_scientific_claim_proposals",
]
