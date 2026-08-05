"""Canonical execution boundary for active governed scientific claim extractors."""

from __future__ import annotations

from typing import Any

from tools.evidence_graph_claim_contracts import ScientificClaimProposal, _identifier
from tools.evidence_graph_claim_extraction import (
    ScientificClaimExtractionBatch,
    extract_scientific_claim_proposals,
)
from tools.evidence_graph_claim_extractor_registry import (
    ScientificClaimExtractorRegistry,
    _allows,
)


def extract_governed_scientific_claim_proposals(
    document: Any,
    extractor_output: Any,
    *,
    owner_id: str,
    generation: int,
    profile_fingerprint: str,
    proposer_id: str,
    extractor_name: str,
    extractor_version: str,
    language: str,
    registry: ScientificClaimExtractorRegistry,
    now: float | None = None,
) -> ScientificClaimExtractionBatch:
    """Execute the closed adapter and bind proposals to the active registry record."""

    if not isinstance(registry, ScientificClaimExtractorRegistry):
        raise ValueError("registry must be ScientificClaimExtractorRegistry.")
    record = registry.require_active(
        owner_id=owner_id,
        extractor_name=extractor_name,
        extractor_version=extractor_version,
    )
    selected_language = _identifier(language, "language", 100).casefold()
    if not _allows(record.supported_languages, selected_language):
        raise PermissionError("extractor is not registered for this language.")

    raw_batch = extract_scientific_claim_proposals(
        document,
        extractor_output,
        owner_id=owner_id,
        generation=generation,
        profile_fingerprint=profile_fingerprint,
        proposer_id=proposer_id,
        extractor_name=extractor_name,
        extractor_version=extractor_version,
        now=now,
    )
    proposals = tuple(
        ScientificClaimProposal.create(
            owner_id=value.owner_id,
            doc_id=value.doc_id,
            generation=value.generation,
            content_sha256=value.content_sha256,
            profile_fingerprint=value.profile_fingerprint,
            claim_key=value.claim_key,
            claim_text=value.claim_text,
            claim_type=value.claim_type,
            modality=value.modality,
            locator=value.locator,
            proposer_kind=record.extractor_kind,
            proposer_id=value.proposer_id,
            extractor_name=value.extractor_name,
            extractor_version=value.extractor_version,
            confidence=value.confidence,
            supersedes_proposal_id=value.supersedes_proposal_id,
            metadata={
                **dict(value.metadata),
                "extractor_registry_record_digest": record.record_digest,
                "extractor_implementation_sha256": record.implementation_sha256,
                "extractor_configuration_sha256": record.configuration_sha256,
                "extractor_output_schema_sha256": record.output_schema_sha256,
                "extractor_language": selected_language,
            },
            created_at=value.created_at,
        )
        for value in raw_batch.proposals
    )
    for proposal in proposals:
        if not _allows(record.supported_claim_types, proposal.claim_type):
            raise PermissionError("extractor emitted an unregistered claim type.")
        if not _allows(record.supported_modalities, proposal.modality):
            raise PermissionError("extractor emitted an unregistered modality.")
    return ScientificClaimExtractionBatch(
        owner_id=raw_batch.owner_id,
        doc_id=raw_batch.doc_id,
        generation=raw_batch.generation,
        content_sha256=raw_batch.content_sha256,
        profile_fingerprint=raw_batch.profile_fingerprint,
        extractor_name=raw_batch.extractor_name,
        extractor_version=raw_batch.extractor_version,
        proposer_id=raw_batch.proposer_id,
        output_digest=raw_batch.output_digest,
        proposals=proposals,
        created_at=raw_batch.created_at,
    )


__all__ = ["extract_governed_scientific_claim_proposals"]
