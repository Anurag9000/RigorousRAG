from __future__ import annotations

import hashlib

import pytest

from evaluation.multimodal_support import (
    MultimodalEvidence,
    MultimodalSupportScore,
    VisualEvidenceAnchor,
    evaluate_multimodal_support,
    image_digest,
    score_multimodal_evidence,
)
from evaluation.semantic_support import ModelIdentity, SemanticProbabilities
from models.local_hf_multimodal_entailment import MultimodalLabelMapping


MODEL = ModelIdentity(
    provider="local-hf",
    model_name="reviewed-visual-nli",
    model_version="revision-1",
    artifact_sha256=hashlib.sha256(b"model").hexdigest(),
)


def evidence(text: str | None = "table value: 42") -> MultimodalEvidence:
    image = b"synthetic-image-fixture"
    anchor = VisualEvidenceAnchor(
        document_id="doc-1",
        generation_id="gen-7",
        page=3,
        region_id="figure-2",
        image_sha256=image_digest(image),
    )
    return MultimodalEvidence(anchor, image, text)


class Scorer:
    @property
    def model_identity(self):
        return MODEL

    def score(self, claim_id, claim_text, selected):
        text_digest = None if selected.evidence_text is None else hashlib.sha256(selected.evidence_text.encode()).hexdigest()
        return MultimodalSupportScore(
            claim_id=claim_id,
            claim_sha256=hashlib.sha256(claim_text.encode()).hexdigest(),
            anchor=selected.anchor,
            probabilities=SemanticProbabilities(0.8, 0.1, 0.1),
            model=MODEL,
            evidence_text_sha256=text_digest,
        )


def test_visual_evidence_is_bound_to_image_digest() -> None:
    selected = evidence()
    with pytest.raises(ValueError, match="digest"):
        MultimodalEvidence(
            VisualEvidenceAnchor(
                document_id=selected.anchor.document_id,
                generation_id=selected.anchor.generation_id,
                page=selected.anchor.page,
                region_id=selected.anchor.region_id,
                image_sha256=hashlib.sha256(b"different").hexdigest(),
            ),
            selected.image_bytes,
        )


def test_score_wrapper_revalidates_claim_anchor_and_model_identity() -> None:
    result = score_multimodal_evidence(
        Scorer(),
        claim_id="claim-1",
        claim_text="The displayed value is 42.",
        evidence=evidence(),
    )
    assert result.anchor.region_id == "figure-2"
    assert result.probabilities.entailment == 0.8


def test_multimodal_support_is_contradiction_first_per_claim() -> None:
    selected = evidence()
    claim_hash = hashlib.sha256(b"claim").hexdigest()
    scores = (
        MultimodalSupportScore(
            "claim-1",
            claim_hash,
            selected.anchor,
            SemanticProbabilities(0.82, 0.10, 0.08),
            MODEL,
            hashlib.sha256(selected.evidence_text.encode()).hexdigest(),
        ),
        MultimodalSupportScore(
            "claim-1",
            claim_hash,
            selected.anchor,
            SemanticProbabilities(0.05, 0.05, 0.90),
            MODEL,
            hashlib.sha256(selected.evidence_text.encode()).hexdigest(),
        ),
    )
    metrics = evaluate_multimodal_support(scores)
    assert metrics.claim_count == 1
    assert metrics.contradicted_claim_rate == 1.0
    assert metrics.supported_claim_rate == 0.0


def test_abstained_evidence_reduces_coverage_without_fabricating_support() -> None:
    selected = evidence(None)
    score = MultimodalSupportScore(
        "claim-1",
        hashlib.sha256(b"claim").hexdigest(),
        selected.anchor,
        SemanticProbabilities(1 / 3, 1 / 3, 1 / 3),
        MODEL,
        abstained=True,
        abstention_reason="unsupported_modality",
    )
    metrics = evaluate_multimodal_support((score,))
    assert metrics.claim_coverage == 0.0
    assert metrics.supported_claim_rate is None
    assert metrics.abstained_evidence_rate == 1.0


def test_semantic_label_mapping_requires_three_unique_explicit_indices() -> None:
    mapping = MultimodalLabelMapping(entailment_index=2, neutral_index=0, contradiction_index=1)
    assert mapping.maximum_index == 2
    with pytest.raises(ValueError, match="unique"):
        MultimodalLabelMapping(entailment_index=0, neutral_index=0, contradiction_index=1)
