"""Executable contradiction-first answer publication and materialized-context tests.

Constructs genuinely digest-bound local evidence but no model output,
network search, production answer or authority endorsement.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib

import pytest

from evaluation.answer_authority import (
    AnswerAuthorityPolicy, AnswerDraftManifest, ClaimEvidenceAuthority,
    DraftClaim, evaluate_answer_authority,
)
from evaluation.semantic_support import SemanticProbabilities
from tools import evidence_context_materialization as material


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _context():
    text = "Verified context passage."
    item = material.MaterializedEvidence(
        order=1,
        evidence_sha256=_sha("verified-evidence"),
        text=text,
        text_sha256=_sha(text),
        token_count=4,
    )
    payload = {
        "schema": "rigorousrag-materialized-context/v1",
        "packing_receipt_sha256": _sha("receipt"),
        "tokenizer_sha256": _sha("tokenizer"),
        "binding_set_sha256": _sha("binding-set"),
        "evidence": [{
            "order": item.order,
            "evidence_sha256": item.evidence_sha256,
            "text_sha256": item.text_sha256,
            "token_count": item.token_count,
        }],
        "total_tokens": 4,
    }
    return material.MaterializedContext(
        packing_receipt_sha256=payload["packing_receipt_sha256"],
        tokenizer_sha256=payload["tokenizer_sha256"],
        binding_set_sha256=payload["binding_set_sha256"],
        evidence=(item,),
        total_tokens=4,
        context_sha256=material._digest(payload),
    )


def _draft(context, *, factual=True, claims=None):
    if claims is None:
        claims = (DraftClaim("claim-1", _sha("claim-1-body"), factual),)
    return AnswerDraftManifest(
        request_sha256=_sha("request"),
        answer_sha256=_sha("draft-answer"),
        materialized_context_sha256=context.context_sha256,
        generator_model_sha256=_sha("model"),
        generation_config_sha256=_sha("generation-config"),
        prompt_template_sha256=_sha("prompt"),
        claims=tuple(claims),
    )


def _observation(*, claim="claim-1", evidence=None, entailment=0.9,
                 neutral=0.05, contradiction=0.05, authoritative=True):
    return ClaimEvidenceAuthority(
        claim_id=claim,
        evidence_sha256=evidence or _sha("verified-evidence"),
        probabilities=SemanticProbabilities(entailment, neutral, contradiction),
        authoritative=authoritative,
    )


def test_verified_context_and_entailing_authority_publish_with_digest_receipt():
    ctx = _context()
    draft = _draft(ctx)
    first = evaluate_answer_authority(draft, ctx, (_observation(),))
    second = evaluate_answer_authority(draft, ctx, (_observation(),))
    assert first == second
    assert first.action == "publish"
    assert not first.reason_codes
    assert first.supported_claim_fraction == pytest.approx(1)
    assert first.claim_results[0].status == "supported"
    assert first.claim_results[0].cited_evidence_sha256s == (_sha("verified-evidence"),)


def test_evidence_outside_exact_materialized_context_blocks_even_high_entailment():
    ctx = _context()
    report = evaluate_answer_authority(
        _draft(ctx), ctx, (_observation(evidence=_sha("not-materialized")),)
    )
    assert report.action == "blocked"
    assert report.context_mismatch_count == 1
    assert report.claim_results[0].status == "context_mismatch"
    assert "cited_evidence_outside_verified_context" in report.reason_codes


def test_contradictory_verified_evidence_requires_abstention():
    ctx = _context()
    report = evaluate_answer_authority(
        _draft(ctx), ctx,
        (_observation(entailment=0.05, neutral=0.05, contradiction=0.9),),
    )
    assert report.action == "abstain"
    assert report.contradicted_claim_count == 1
    assert report.claim_results[0].status == "contradicted"


def test_unknown_or_unverified_support_never_publishes_as_authoritative():
    ctx = _context()
    result = evaluate_answer_authority(
        _draft(ctx), ctx, (_observation(authoritative=False),)
    )
    assert result.action == "abstain"
    assert result.unverified_authority_fraction == 1
    assert result.claim_results[0].status == "authority_unverified"


def test_weak_entailment_requires_review_or_abstention_by_policy():
    ctx = _context()
    observation = _observation(entailment=0.2, neutral=0.75, contradiction=0.05)
    default = evaluate_answer_authority(_draft(ctx), ctx, (observation,))
    review = evaluate_answer_authority(
        _draft(ctx), ctx, (observation,),
        policy=AnswerAuthorityPolicy(review_on_unsupported=True),
    )
    assert default.action == "abstain"
    assert review.action == "review_required"
    assert review.claim_results[0].status == "unsupported"
    assert "supported_claim_fraction_below_threshold" in default.reason_codes


def test_factual_uncited_claim_is_explicit_failure_not_hallucinated_citation():
    ctx = _context()
    result = evaluate_answer_authority(_draft(ctx), ctx, ())
    assert result.action == "abstain"
    assert result.uncited_claim_count == 1
    assert result.claim_results[0].status == "uncited"
    assert "uncited_factual_claim_present" in result.reason_codes


def test_non_factual_claim_needs_no_citation_and_is_not_factual_denominator():
    ctx = _context()
    result = evaluate_answer_authority(_draft(ctx, factual=False), ctx, ())
    assert result.action == "publish"
    assert result.supported_claim_fraction == 1
    assert result.claim_results[0].status == "non_factual"


def test_mixed_factual_claims_require_all_mandatory_evidence_under_default_policy():
    ctx = _context()
    claims = (
        DraftClaim("claim-1", _sha("claim-1")),
        DraftClaim("claim-2", _sha("claim-2")),
    )
    result = evaluate_answer_authority(
        _draft(ctx, claims=claims), ctx, (_observation(),),
    )
    assert result.action == "abstain"
    assert result.supported_claim_fraction == pytest.approx(0.5)
    assert result.uncited_claim_count == 1


def test_manifest_cannot_bind_a_different_context_or_unregistered_claim():
    ctx = _context()
    bad_draft = replace(_draft(ctx), materialized_context_sha256=_sha("wrong-context"))
    with pytest.raises(ValueError, match="does not bind"):
        evaluate_answer_authority(bad_draft, ctx, ())
    with pytest.raises(ValueError, match="outside the draft manifest"):
        evaluate_answer_authority(
            _draft(ctx), ctx, (_observation(claim="unknown"),),
        )


def test_materialized_context_digest_and_text_tampering_detected():
    ctx = _context()
    with pytest.raises(ValueError, match="digest"):
        replace(ctx, total_tokens=5)
    with pytest.raises(ValueError, match="text_sha256"):
        replace(ctx.evidence[0], text="different content")


def test_decision_and_claim_result_hashes_reject_tampering():
    ctx = _context()
    result = evaluate_answer_authority(_draft(ctx), ctx, (_observation(),))
    with pytest.raises(ValueError, match="decision_sha256"):
        replace(result, supported_claim_fraction=0.5)
    with pytest.raises(ValueError, match="result_sha256"):
        replace(result.claim_results[0], status="uncited")


def test_duplicate_or_invalid_draft_claims_rejected():
    ctx = _context()
    a = DraftClaim("claim-1", _sha("first"))
    b = DraftClaim("claim-1", _sha("second"))
    with pytest.raises(ValueError, match="unique"):
        _draft(ctx, claims=(a, b))
    with pytest.raises(ValueError, match="non-empty"):
        _draft(ctx, claims=())
    with pytest.raises(ValueError, match="requires_evidence"):
        DraftClaim("claim", _sha("claim"), requires_evidence=1)


@pytest.mark.parametrize("value", (-0.1, 1.1, float("nan"), True))
def test_authority_thresholds_are_bounded(value):
    with pytest.raises(ValueError):
        AnswerAuthorityPolicy(min_entailment_probability=value)


@pytest.mark.parametrize("probabilities", (
    (0.8, 0.4, 0.0),
    (0.4, 0.4, 0.4),
    (float("nan"), 0.2, 0.8),
))
def test_semantic_score_vector_requires_real_probabilities(probabilities):
    with pytest.raises(ValueError):
        SemanticProbabilities(*probabilities)


def test_observation_bad_digest_or_non_boolean_provenance_is_rejected():
    with pytest.raises(ValueError, match="SHA-256"):
        _observation(evidence="malformed")
    with pytest.raises(ValueError, match="authoritative"):
        _observation(authoritative=1)
