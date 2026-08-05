from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace

import pytest

from tools.evidence_graph_claim_contracts import (
    ClaimReviewDecision,
    ClaimReviewPolicy,
    ClaimReviewerGrant,
)
from tools.evidence_graph_claim_extraction import extract_scientific_claim_proposals
from tools.evidence_graph_claim_review import (
    GovernedScientificClaimReviewService,
    approved_claim_annotations,
)
from tools.evidence_graph_claim_store import ScientificClaimReviewStore
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


def document() -> Document:
    text = "Drug A reduced mortality in the randomized cohort."
    return Document(
        id="doc1",
        text=text,
        sections=[Section(text)],
        metadata={"content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()},
    )


def raw(*, claim_key="claim-1", claim_text="Drug A reduced mortality.", supersedes=None):
    text = document().text
    value = {
        "claim_key": claim_key,
        "claim_text": claim_text,
        "claim_type": "finding",
        "modality": "asserted",
        "section_index": 0,
        "page_number": 1,
        "char_start": 0,
        "char_end": text.index(" in the"),
        "confidence": 0.9,
    }
    if supersedes is not None:
        value["supersedes_proposal_id"] = supersedes
    return {"schema_version": 1, "claims": [value]}


def proposal(*, proposer="model-1", version="1", supersedes=None, generation=1):
    return extract_scientific_claim_proposals(
        document(),
        raw(
            claim_key="claim-1" if supersedes is None else "claim-2",
            claim_text=(
                "Drug A reduced mortality."
                if supersedes is None
                else "Drug A reduced mortality in the cohort."
            ),
            supersedes=supersedes,
        ),
        owner_id="alice",
        generation=generation,
        profile_fingerprint="b" * 64,
        proposer_id=proposer,
        extractor_name="claims",
        extractor_version=version,
        now=float(generation),
    ).proposals[0]


def policy(*, expires_at=None):
    return ClaimReviewPolicy(
        reviewers=(
            ClaimReviewerGrant(
                reviewer_id="reviewer",
                owners=("alice",),
                doc_ids=("doc1",),
                decisions=("approved", "rejected", "superseded"),
                expires_at=expires_at,
            ),
        )
    )


def binding(actor="reviewer"):
    return ReviewActorBinding.create(
        actor_id=actor,
        binding_method="process_environment",
        loaded_at=1.0,
    )


def decision(value, kind, *, replacement=None, decided_at=3.0):
    return ClaimReviewDecision.create(
        proposal_id=value.proposal_id,
        owner_id=value.owner_id,
        decision=kind,
        reviewer_id="reviewer",
        reason_code="reviewed",
        replacement_proposal_id=replacement,
        decided_at=decided_at,
    )


def test_store_is_atomic_idempotent_and_detects_correction_branching(tmp_path):
    store = ScientificClaimReviewStore(tmp_path / "claims.sqlite3")
    original = proposal()
    corrected = proposal(
        proposer="model-2", version="2", supersedes=original.proposal_id
    )
    assert store.submit_many((original, corrected)) == (original, corrected)
    assert store.submit_many((original, corrected)) == (original, corrected)

    competing = replace(
        corrected,
        proposal_id="f" * 64,
        claim_key="claim-3",
    )
    with pytest.raises(ValueError, match="proposal_id"):
        store.submit(competing)

    other = proposal(
        proposer="model-3", version="3", supersedes=original.proposal_id
    )
    with pytest.raises(RuntimeError, match="different correction successor"):
        store.submit(other)


def test_cross_generation_correction_and_reviewer_self_approval_refuse(tmp_path):
    store = ScientificClaimReviewStore(tmp_path / "claims.sqlite3")
    original = proposal()
    store.submit(original)
    cross_generation = proposal(
        proposer="model-2",
        version="2",
        supersedes=original.proposal_id,
        generation=2,
    )
    with pytest.raises(PermissionError, match="same document generation"):
        store.submit(cross_generation)

    authored = proposal(proposer="reviewer")
    store = ScientificClaimReviewStore(tmp_path / "self.sqlite3")
    store.submit(authored)
    service = GovernedScientificClaimReviewService(
        store=store, policy=policy(), clock=lambda: 3.0
    )
    with pytest.raises(PermissionError, match="own proposal"):
        service.decide(decision(authored, "approved"), actor_binding=binding())


def test_policy_expiry_and_process_owned_actor_are_enforced(tmp_path):
    value = proposal()
    store = ScientificClaimReviewStore(tmp_path / "claims.sqlite3")
    store.submit(value)

    expired = GovernedScientificClaimReviewService(
        store=store, policy=policy(expires_at=2.0), clock=lambda: 3.0
    )
    with pytest.raises(PermissionError, match="grant"):
        expired.decide(decision(value, "approved"), actor_binding=binding())

    active = GovernedScientificClaimReviewService(
        store=store, policy=policy(), clock=lambda: 3.0
    )
    with pytest.raises(PermissionError, match="process-owned"):
        active.decide(
            decision(value, "approved"), actor_binding=binding("different-reviewer")
        )


def test_correction_requires_superseded_predecessor_before_approval(tmp_path):
    store = ScientificClaimReviewStore(tmp_path / "claims.sqlite3")
    original = proposal()
    corrected = proposal(
        proposer="model-2", version="2", supersedes=original.proposal_id
    )
    store.submit_many((original, corrected))
    service = GovernedScientificClaimReviewService(
        store=store, policy=policy(), clock=lambda: 3.0
    )

    with pytest.raises(RuntimeError, match="before predecessor supersession"):
        service.decide(decision(corrected, "approved"), actor_binding=binding())

    service.decide(
        decision(
            original,
            "superseded",
            replacement=corrected.proposal_id,
            decided_at=3.0,
        ),
        actor_binding=binding(),
    )
    stored, authorization = service.decide(
        decision(corrected, "approved", decided_at=4.0),
        actor_binding=binding(),
    )
    assert stored.decision == "approved"
    assert authorization.separation_of_duties_enforced is True

    annotations = approved_claim_annotations(
        owner_id="alice",
        doc_id="doc1",
        generation=1,
        content_sha256=corrected.content_sha256,
        profile_fingerprint=corrected.profile_fingerprint,
        proposal_ids=(corrected.proposal_id,),
        store=store,
    )
    assert len(annotations) == 1
    assert annotations[0].node_type == "claim"
    assert annotations[0].metadata["explicit_reviewed_claim"] is True
    assert annotations[0].metadata["semantic_relation_inference_performed"] is False
    assert annotations[0].metadata["supersedes_proposal_id"] == original.proposal_id


def test_atomic_decision_authorization_replay_preserves_original_times(tmp_path):
    value = proposal()
    store = ScientificClaimReviewStore(tmp_path / "claims.sqlite3")
    store.submit(value)
    service = GovernedScientificClaimReviewService(
        store=store, policy=policy(), clock=lambda: 3.0
    )

    first, first_auth = service.decide(
        decision(value, "approved", decided_at=3.0), actor_binding=binding()
    )
    second, second_auth = service.decide(
        decision(value, "approved", decided_at=99.0), actor_binding=binding()
    )
    assert second.decision_id == first.decision_id
    assert second.decided_at == first.decided_at == 3.0
    assert second_auth.authorization_digest == first_auth.authorization_digest
    assert second_auth.authorized_at == first_auth.authorized_at == 3.0


def test_rejected_pending_and_obsolete_claims_do_not_convert(tmp_path):
    store = ScientificClaimReviewStore(tmp_path / "claims.sqlite3")
    pending = proposal(proposer="model-pending")
    rejected = proposal(proposer="model-rejected", version="2")
    store.submit_many((pending, rejected))
    service = GovernedScientificClaimReviewService(
        store=store, policy=policy(), clock=lambda: 3.0
    )
    service.decide(decision(rejected, "rejected"), actor_binding=binding())

    for value, message in ((pending, "lacks"), (rejected, "only approved")):
        with pytest.raises((RuntimeError, PermissionError), match=message):
            approved_claim_annotations(
                owner_id="alice",
                doc_id="doc1",
                generation=1,
                content_sha256=value.content_sha256,
                profile_fingerprint=value.profile_fingerprint,
                proposal_ids=(value.proposal_id,),
                store=store,
            )


def test_database_identity_and_payload_tampering_fail_closed(tmp_path):
    path = tmp_path / "claims.sqlite3"
    store = ScientificClaimReviewStore(path)
    value = proposal()
    store.submit(value)

    with store._lock, store._connect() as connection:
        connection.execute(
            "UPDATE scientific_claim_proposals SET proposal_digest=? WHERE proposal_id=?",
            ("f" * 64, value.proposal_id),
        )
    with pytest.raises(RuntimeError, match="digest"):
        store.get_proposal(value.proposal_id)

    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    with pytest.raises(RuntimeError, match="identity changed"):
        store.list(owner_id="alice", doc_id="doc1")
