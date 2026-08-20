"""Production artifact admission guarded by strict authoritative promotion evidence.

The generic artifact module exposes reusable primitive admission helpers for research and
internal composition. Production admission uses this module: exact artifact bytes are re-hashed,
evaluator-bound result evidence is re-verified, the embedded promotion policy is independently
re-run with exact direction coverage, and only then is the artifact handed to the admission sink.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from training.advanced_rag_artifact_directory import (
    assert_artifact_directory_matches_manifest,
)
from training.advanced_rag_artifacts import AdvancedArtifactManifest, ArtifactAdmissionSink
from training.authoritative_advanced_promotion import AuthoritativeAdvancedPromotionEvidence
from training.strict_authoritative_advanced_promotion import (
    assert_strict_authoritative_advanced_promotion,
)


def admit_authoritative_advanced_artifact(
    directory: str | Path,
    manifest: AdvancedArtifactManifest,
    promotion: AuthoritativeAdvancedPromotionEvidence,
    sink: ArtifactAdmissionSink,
) -> Any:
    """Admit one exact artifact only after full production evidence re-verification."""
    if not isinstance(manifest, AdvancedArtifactManifest):
        raise ValueError("manifest must be AdvancedArtifactManifest")
    if not isinstance(promotion, AuthoritativeAdvancedPromotionEvidence):
        raise ValueError(
            "production admission requires AuthoritativeAdvancedPromotionEvidence"
        )
    selected = assert_artifact_directory_matches_manifest(directory, manifest)
    assert_strict_authoritative_advanced_promotion(manifest, promotion)
    if not promotion.promoted:
        raise ValueError("only an authoritatively promoted artifact may enter admission")
    if promotion.artifact_sha256 != manifest.artifact_sha256:
        raise ValueError("authoritative promotion is bound to a different artifact")
    return sink.admit(
        str(selected),
        artifact_sha256=manifest.artifact_sha256,
        promotion_receipt_sha256=promotion.evidence_sha256,
    )


__all__ = ["admit_authoritative_advanced_artifact"]
