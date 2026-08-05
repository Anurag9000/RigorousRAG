from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace

import pytest

from tools.evidence_graph_claim_extractor_registry import (
    SCIENTIFIC_CLAIM_OUTPUT_SCHEMA_SHA256,
    ClaimExtractorAdministratorGrant,
    ClaimExtractorGovernancePolicy,
    GovernedScientificClaimExtractorService,
    ScientificClaimExtractorRecord,
    ScientificClaimExtractorRegistry,
)
from tools.evidence_graph_claim_registered_extraction import (
    extract_governed_scientific_claim_proposals,
)
from tools.evidence_graph_relation_actor import ReviewActorBinding


@dataclass
class Section:
    content: str
    page_number: int = 1


@dataclass
class Document:
    id: str
    text: str
    sections: list[Section]
    metadata: dict


def actor(value="extractor-admin"):
    return ReviewActorBinding.create(
        actor_id=value,
        binding_method="process_environment",
        loaded_at=1.0,
    )


def policy(*, expires_at=None):
    return ClaimExtractorGovernancePolicy(
        administrators=(
            ClaimExtractorAdministratorGrant(
                administrator_id="extractor-admin",
                owners=("alice",),
                extractor_names=("claims",),
                actions=("register", "retire"),
                expires_at=expires_at,
            ),
        )
    )


def registration(service, *, kind="model", version="1"):
    return service.register(
        actor=actor(),
        owner_id="alice",
        extractor_name="claims",
        extractor_version=version,
        extractor_kind=kind,
        implementation_sha256="a" * 64,
        configuration_sha256="b" * 64,
        supported_claim_types=("finding", "limitation"),
        supported_modalities=("asserted", "uncertain"),
        supported_languages=("en",),
    )


def document():
    text = "Drug A reduced mortality in the randomized cohort."
    return Document(
        id="doc1",
        text=text,
        sections=[Section(text)],
        metadata={"content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()},
    )


def output(*, kind="finding", modality="asserted"):
    text = document().text
    return {
        "schema_version": 1,
        "claims": [
            {
                "claim_key": "claim-1",
                "claim_text": "Drug A reduced mortality.",
                "claim_type": kind,
                "modality": modality,
                "section_index": 0,
                "page_number": 1,
                "char_start": 0,
                "char_end": text.index(" in the"),
                "confidence": 0.9,
            }
        ],
    }


def test_record_identity_registration_and_replay_are_deterministic(tmp_path):
    registry = ScientificClaimExtractorRegistry(tmp_path / "extractors.sqlite3")
    service = GovernedScientificClaimExtractorService(
        registry=registry,
        policy=policy(),
        clock=lambda: 2.0,
    )
    first = registration(service)
    second = registration(service)

    assert first == second
    assert first.state == "active"
    assert first.output_schema_sha256 == SCIENTIFIC_CLAIM_OUTPUT_SCHEMA_SHA256
    assert len(first.record_digest) == 64
    assert registry.require_active(
        owner_id="alice", extractor_name="claims", extractor_version="1"
    ) == first


def test_registration_scope_policy_expiry_and_actor_identity_fail_closed(tmp_path):
    registry = ScientificClaimExtractorRegistry(tmp_path / "extractors.sqlite3")
    expired = GovernedScientificClaimExtractorService(
        registry=registry,
        policy=policy(expires_at=1.0),
        clock=lambda: 2.0,
    )
    with pytest.raises(PermissionError, match="grant"):
        registration(expired)

    service = GovernedScientificClaimExtractorService(
        registry=registry,
        policy=policy(),
        clock=lambda: 2.0,
    )
    with pytest.raises(PermissionError, match="not authorized"):
        service.register(
            actor=actor("other-admin"),
            owner_id="alice",
            extractor_name="claims",
            extractor_version="1",
            extractor_kind="model",
            implementation_sha256="a" * 64,
            configuration_sha256="b" * 64,
            supported_claim_types=("finding",),
            supported_modalities=("asserted",),
            supported_languages=("en",),
        )


def test_retirement_is_monotonic_and_retired_versions_cannot_execute(tmp_path):
    registry = ScientificClaimExtractorRegistry(tmp_path / "extractors.sqlite3")
    service = GovernedScientificClaimExtractorService(
        registry=registry,
        policy=policy(),
        clock=lambda: 2.0,
    )
    active = registration(service)
    retired = service.retire(
        actor=actor(),
        owner_id="alice",
        extractor_name="claims",
        extractor_version="1",
    )
    assert retired.state == "retired"
    assert retired.retired_at == 2.0
    assert service.retire(
        actor=actor(),
        owner_id="alice",
        extractor_name="claims",
        extractor_version="1",
    ) == retired
    with pytest.raises(PermissionError, match="retired"):
        registry.require_active(
            owner_id="alice", extractor_name="claims", extractor_version="1"
        )
    with pytest.raises(RuntimeError, match="may not be reactivated"):
        registry.register(active)


def test_governed_rule_extraction_binds_registry_provenance(tmp_path):
    registry = ScientificClaimExtractorRegistry(tmp_path / "extractors.sqlite3")
    service = GovernedScientificClaimExtractorService(
        registry=registry,
        policy=policy(),
        clock=lambda: 2.0,
    )
    record = registration(service, kind="rule")
    batch = extract_governed_scientific_claim_proposals(
        document(),
        output(),
        owner_id="alice",
        generation=1,
        profile_fingerprint="c" * 64,
        proposer_id="rule-extractor",
        extractor_name="claims",
        extractor_version="1",
        language="EN",
        registry=registry,
        now=3.0,
    )

    claim = batch.proposals[0]
    assert claim.proposer_kind == "rule"
    assert claim.metadata["extractor_registry_record_digest"] == record.record_digest
    assert claim.metadata["extractor_implementation_sha256"] == "a" * 64
    assert claim.metadata["extractor_configuration_sha256"] == "b" * 64
    assert claim.metadata["extractor_language"] == "en"


def test_language_taxonomy_and_schema_capabilities_are_enforced(tmp_path):
    registry = ScientificClaimExtractorRegistry(tmp_path / "extractors.sqlite3")
    service = GovernedScientificClaimExtractorService(
        registry=registry,
        policy=policy(),
        clock=lambda: 2.0,
    )
    registration(service)

    with pytest.raises(PermissionError, match="language"):
        extract_governed_scientific_claim_proposals(
            document(),
            output(),
            owner_id="alice",
            generation=1,
            profile_fingerprint="c" * 64,
            proposer_id="extractor",
            extractor_name="claims",
            extractor_version="1",
            language="fr",
            registry=registry,
        )
    with pytest.raises(PermissionError, match="claim type"):
        extract_governed_scientific_claim_proposals(
            document(),
            output(kind="hypothesis"),
            owner_id="alice",
            generation=1,
            profile_fingerprint="c" * 64,
            proposer_id="extractor",
            extractor_name="claims",
            extractor_version="1",
            language="en",
            registry=registry,
        )

    active = registry.require_active(
        owner_id="alice", extractor_name="claims", extractor_version="1"
    )
    with pytest.raises(ValueError, match="output schema"):
        replace(active, output_schema_sha256="f" * 64)


def test_registry_payload_and_file_identity_tampering_fail_closed(tmp_path):
    path = tmp_path / "extractors.sqlite3"
    registry = ScientificClaimExtractorRegistry(path)
    service = GovernedScientificClaimExtractorService(
        registry=registry,
        policy=policy(),
        clock=lambda: 2.0,
    )
    registration(service)

    with registry._lock, registry._connect() as connection:
        connection.execute(
            "UPDATE scientific_claim_extractors SET record_digest=?",
            ("f" * 64,),
        )
    with pytest.raises(RuntimeError, match="columns"):
        registry.list(owner_id="alice")

    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    with pytest.raises(RuntimeError, match="identity changed"):
        registry.list(owner_id="alice")
