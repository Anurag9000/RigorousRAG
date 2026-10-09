"""Server-allowlisted post-generation citation set verification.

Synthetic claim/evidence scores exercise identity and support contracts;
these tests do not produce retrieved evidence or establish factual truth.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from tools.citation_refinement import (
    CitationRefinementPolicy, ClaimBinding, ClaimEvidenceAssessment,
    ClaimRefinementStatus, UnresolvedClaimAction, refine_citations,
)


def _sha(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _claim(name="claim-1", citations=()):
    return ClaimBinding(name, _sha("claim-content-" + name), tuple(citations))


def _assessment(claim="claim-1", evidence="e1", *,
                support=0.9, contradiction=0.01, quality=1.0, group=None):
    return ClaimEvidenceAssessment(
        claim_id=claim,
        evidence_id=evidence,
        evidence_sha256=_sha("evidence-" + evidence),
        source_group_sha256=_sha(group or "group-" + evidence),
        support_probability=support,
        contradiction_probability=contradiction,
        evidence_quality=quality,
    )


def _refine(claims, assessments, *, allowed=None, policy=None):
    return refine_citations(
        answer_sha256=_sha("generated-answer"),
        allowed_evidence_set_sha256=_sha("server-owned-allowlist"),
        claims=claims,
        assessments=assessments,
        allowed_evidence_ids=allowed or tuple(dict.fromkeys(x.evidence_id for x in assessments)),
        policy=policy or CitationRefinementPolicy(),
    )


def test_server_allowlisted_evidence_can_make_deterministic_supported_receipt():
    claim = _claim(citations=("e1",))
    entries = (_assessment(), _assessment(evidence="e2", support=0.95))
    first = _refine((claim,), entries)
    second = _refine((claim,), tuple(reversed(entries)))
    assert first == second
    assert first.policy_sha256 == CitationRefinementPolicy().policy_sha256
    assert not first.requires_abstention
    assert not first.requires_review
    result = first.claim_results[0]
    assert result.status == ClaimRefinementStatus.SUPPORTED
    assert result.refined_citation_ids == ("e1",)
    assert result.added_citation_ids == ()
    assert result.removed_citation_ids == ()


def test_refiner_adds_diverse_evidence_when_one_citation_is_insufficient():
    policy = CitationRefinementPolicy(
        minimum_claim_support=0.9, minimum_independent_sources=2,
    )
    claim = _claim(citations=("e1",))
    receipt = _refine(
        (claim,),
        (_assessment(evidence="e1", support=0.8), _assessment(evidence="e2", support=0.8)),
        policy=policy,
    )
    result = receipt.claim_results[0]
    assert result.status == ClaimRefinementStatus.SUPPORTED
    assert result.refined_citation_ids == ("e1", "e2")
    assert result.added_citation_ids == ("e2",)
    assert result.independent_sources == 2
    assert result.combined_support == pytest.approx(0.96)


def test_weak_original_citation_is_removed_with_explicit_identity_delta():
    claim = _claim(citations=("weak",))
    receipt = _refine(
        (claim,),
        (_assessment(evidence="weak", support=0.01),
         _assessment(evidence="strong", support=0.98)),
    )
    result = receipt.claim_results[0]
    assert result.status == ClaimRefinementStatus.SUPPORTED
    assert result.refined_citation_ids == ("strong",)
    assert result.added_citation_ids == ("strong",)
    assert result.removed_citation_ids == ("weak",)


def test_contradicted_claim_stays_contradicted_even_with_credible_other_evidence():
    claim = _claim(citations=("e1",))
    policy = CitationRefinementPolicy(unresolved_action=UnresolvedClaimAction.ABSTAIN)
    receipt = _refine(
        (claim,), (_assessment(), _assessment(evidence="e2", support=0.1, contradiction=0.8)),
        policy=policy,
    )
    result = receipt.claim_results[0]
    assert result.status == ClaimRefinementStatus.CONTRADICTED
    assert result.unresolved_action == UnresolvedClaimAction.ABSTAIN
    assert receipt.requires_abstention and receipt.requires_review
    assert result.maximum_contradiction == pytest.approx(0.8)


def test_no_eligible_evidence_is_unresolved_and_requires_action():
    policy = CitationRefinementPolicy(unresolved_action=UnresolvedClaimAction.REVIEW)
    receipt = _refine(
        (_claim(),), (_assessment(support=0.1, quality=0.3),), policy=policy,
    )
    result = receipt.claim_results[0]
    assert result.status == ClaimRefinementStatus.UNRESOLVED
    assert result.refined_citation_ids == ()
    assert result.unresolved_action == UnresolvedClaimAction.REVIEW
    assert receipt.requires_review
    assert not receipt.requires_abstention


def test_partial_support_is_not_silently_reported_as_grounded():
    policy = CitationRefinementPolicy(minimum_claim_support=0.99)
    receipt = _refine((_claim(),), (_assessment(support=0.55),), policy=policy)
    result = receipt.claim_results[0]
    assert result.status == ClaimRefinementStatus.PARTIALLY_SUPPORTED
    assert receipt.requires_review
    assert result.combined_support == pytest.approx(0.55)


def test_evidence_not_server_allowlisted_is_rejected_before_scoring():
    with pytest.raises(ValueError, match="server-owned allowlist"):
        _refine((_claim(),), (_assessment(),), allowed=("other",))
    with pytest.raises(ValueError, match="server-owned allowlist"):
        _refine((_claim(citations=("other",)),), (_assessment(),))


def test_assessments_must_refer_to_known_claims_and_unique_evidence_pairs():
    with pytest.raises(ValueError, match="unknown claim"):
        _refine((_claim(),), (_assessment(claim="unbound"),))
    with pytest.raises(ValueError, match="duplicate claim/evidence"):
        _refine((_claim(),), (_assessment(), _assessment()))


def test_claim_and_allowed_evidence_ids_must_be_unique():
    with pytest.raises(ValueError, match="claim ids must be unique"):
        _refine((_claim(), _claim()), (_assessment(),))
    with pytest.raises(ValueError, match="allowed_evidence_ids must be unique"):
        _refine((_claim(),), (_assessment(),), allowed=("e1", "e1"))


def test_refined_receipt_digest_rejects_metric_and_identity_tampering():
    receipt = _refine((_claim(),), (_assessment(),))
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(receipt, requires_review=True)
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(receipt, allowed_evidence_set_sha256=_sha("different-universe"))
    with pytest.raises(ValueError, match="added citation identity mismatch"):
        replace(receipt.claim_results[0], added_citation_ids=("e1",))


def test_multi_claim_receipt_keeps_claims_separate():
    claims = (_claim("alpha", ("e1",)), _claim("beta", ("e2",)))
    assessments = (
        _assessment("alpha", "e1", support=0.9),
        _assessment("beta", "e2", support=0.9),
    )
    receipt = _refine(claims, assessments)
    assert tuple(x.claim_id for x in receipt.claim_results) == ("alpha", "beta")
    assert all(x.status == ClaimRefinementStatus.SUPPORTED for x in receipt.claim_results)


def test_citation_cap_and_independent_source_gate_are_enforced():
    policy = CitationRefinementPolicy(minimum_independent_sources=3, maximum_citations_per_claim=2)
    receipt = _refine(
        (_claim(),),
        tuple(_assessment(evidence=f"e{i}", support=0.6) for i in range(5)),
        policy=policy,
    )
    result = receipt.claim_results[0]
    assert len(result.refined_citation_ids) == 2
    assert result.independent_sources == 2
    assert result.status != ClaimRefinementStatus.SUPPORTED
    assert receipt.requires_review


@pytest.mark.parametrize("changes", (
    {"minimum_evidence_support": -0.1},
    {"minimum_claim_support": 1.1},
    {"maximum_citations_per_claim": 0},
    {"minimum_independent_sources": True},
))
def test_policy_rejects_unbounded_thresholds(changes):
    with pytest.raises(ValueError):
        replace(CitationRefinementPolicy(), **changes)


@pytest.mark.parametrize("changes", (
    {"support_probability": 1.1},
    {"contradiction_probability": 0.4, "support_probability": 0.9},
    {"evidence_quality": float("nan")},
))
def test_scored_evidence_rejects_invalid_probabilities(changes):
    with pytest.raises(ValueError):
        replace(_assessment(), **changes)


def test_nonempty_claims_assessments_and_policy_are_required():
    with pytest.raises(ValueError, match="claims"):
        _refine((), (_assessment(),))
    with pytest.raises(ValueError, match="assessments"):
        _refine((_claim(),), ())
    with pytest.raises(ValueError, match="policy"):
        _refine((_claim(),), (_assessment(),), policy="not-a-policy")
