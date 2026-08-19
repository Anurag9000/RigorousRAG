"""Bind qualified advanced-RAG artifacts into the existing runtime-stack authority.

Production runtime binding requires the full evidence chain: exact exported artifact bytes,
authoritative promotion evidence, authoritative advanced evaluation evidence, and verified v2
benchmark-result artifacts.  A compact ``AdvancedPromotionEvidence`` alone is intentionally
insufficient for this production handoff.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from orchestration.runtime_stack_authority import (
    RuntimeComponent,
    RuntimePromotionEvidence,
    RuntimeStackArtifact,
)
from training.advanced_rag_artifact_directory import (
    assert_artifact_directory_matches_manifest,
)
from training.advanced_rag_artifacts import AdvancedArtifactManifest
from training.authoritative_advanced_promotion import (
    AuthoritativeAdvancedPromotionEvidence,
    assert_authoritative_advanced_promotion,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _component_kind(manifest: AdvancedArtifactManifest) -> str:
    if manifest.kind == "grounded_generator":
        return "generator"
    if manifest.kind == "dynamic_rag_policy":
        # Runtime stack authority predates generation-time dynamic retrieval learning. Its
        # query_router slot is the existing policy-routing authority surface and is kept for
        # schema compatibility; component_id/contract retain the exact dynamic-policy meaning.
        return "query_router"
    raise ValueError("unsupported advanced artifact kind")


def _component_contract(manifest: AdvancedArtifactManifest) -> str:
    return _digest(
        {
            "schema": "rigorousrag-advanced-runtime-component-contract/v2",
            "artifact_sha256": manifest.artifact_sha256,
            "artifact_kind": manifest.kind,
            "checkpoint_digest": manifest.checkpoint_digest,
            "plan_sha256": manifest.plan_sha256,
            "training_input_sha256": manifest.training_input_sha256,
            "training_config_sha256": manifest.training_config_sha256,
            "source_commit": manifest.source_commit,
            "dataset_manifest_sha256": manifest.dataset_manifest_sha256,
            "architecture_sha256": manifest.architecture_sha256,
            "base_model_sha256": manifest.base_model_sha256,
            "generator_family": manifest.generator_family,
            "tokenizer_sha256": manifest.tokenizer_sha256,
            "retrieval_stack_sha256": manifest.retrieval_stack_sha256,
            "budget_sha256": manifest.budget_sha256,
            "runtime_config": manifest.runtime_config,
            "evaluation_receipt_sha256": manifest.evaluation_receipt_sha256,
            "production_evidence_requirement": "authoritative_advanced_promotion/v1",
        }
    )


@dataclass(frozen=True)
class AdvancedRuntimeComponentBinding:
    component: RuntimeComponent
    artifact_sha256: str
    artifact_kind: str
    promotion_evidence_sha256: str
    primitive_promotion_evidence_sha256: str
    authoritative_evaluation_evidence_sha256: str
    evaluation_receipt_sha256: str
    binding_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.component, RuntimeComponent):
            raise ValueError("component must be RuntimeComponent")
        if self.artifact_kind not in {"grounded_generator", "dynamic_rag_policy"}:
            raise ValueError("unsupported advanced artifact kind")
        for name in (
            "artifact_sha256",
            "promotion_evidence_sha256",
            "primitive_promotion_evidence_sha256",
            "authoritative_evaluation_evidence_sha256",
            "evaluation_receipt_sha256",
            "binding_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.component.artifact_sha256 != self.artifact_sha256:
            raise ValueError("runtime component artifact differs from advanced artifact")
        if _digest(self._payload()) != self.binding_sha256:
            raise ValueError("advanced runtime component binding digest mismatch")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-advanced-runtime-component-binding/v2",
            "component": asdict(self.component),
            "artifact_sha256": self.artifact_sha256,
            "artifact_kind": self.artifact_kind,
            "promotion_evidence_sha256": self.promotion_evidence_sha256,
            "primitive_promotion_evidence_sha256": self.primitive_promotion_evidence_sha256,
            "authoritative_evaluation_evidence_sha256": self.authoritative_evaluation_evidence_sha256,
            "evaluation_receipt_sha256": self.evaluation_receipt_sha256,
        }


def bind_qualified_advanced_artifact(
    artifact_directory: str | Path,
    manifest: AdvancedArtifactManifest,
    promotion: AuthoritativeAdvancedPromotionEvidence,
    *,
    component_id: str,
) -> AdvancedRuntimeComponentBinding:
    """Create a runtime component only from exact artifact bytes plus authoritative evidence."""
    if not isinstance(manifest, AdvancedArtifactManifest):
        raise ValueError("manifest must be AdvancedArtifactManifest")
    if not isinstance(promotion, AuthoritativeAdvancedPromotionEvidence):
        raise ValueError(
            "production runtime binding requires AuthoritativeAdvancedPromotionEvidence"
        )
    assert_artifact_directory_matches_manifest(artifact_directory, manifest)
    assert_authoritative_advanced_promotion(manifest, promotion)
    if not promotion.promoted:
        raise ValueError("runtime binding requires promoted advanced artifact evidence")
    if promotion.artifact_sha256 != manifest.artifact_sha256:
        raise ValueError("promotion evidence is bound to a different advanced artifact")
    if (
        manifest.evaluation_receipt_sha256 is not None
        and manifest.evaluation_receipt_sha256 != promotion.evaluation_receipt_sha256
    ):
        raise ValueError("artifact/promotion evaluation receipt binding differs")

    component = RuntimeComponent(
        kind=_component_kind(manifest),
        component_id=component_id,
        artifact_sha256=manifest.artifact_sha256,
        contract_sha256=_component_contract(manifest),
    )
    unsigned = {
        "schema": "rigorousrag-advanced-runtime-component-binding/v2",
        "component": asdict(component),
        "artifact_sha256": manifest.artifact_sha256,
        "artifact_kind": manifest.kind,
        "promotion_evidence_sha256": promotion.evidence_sha256,
        "primitive_promotion_evidence_sha256": promotion.promotion.evidence_sha256,
        "authoritative_evaluation_evidence_sha256": promotion.authoritative_evaluation_evidence_sha256,
        "evaluation_receipt_sha256": promotion.evaluation_receipt_sha256,
    }
    return AdvancedRuntimeComponentBinding(
        component=component,
        artifact_sha256=manifest.artifact_sha256,
        artifact_kind=manifest.kind,
        promotion_evidence_sha256=promotion.evidence_sha256,
        primitive_promotion_evidence_sha256=promotion.promotion.evidence_sha256,
        authoritative_evaluation_evidence_sha256=promotion.authoritative_evaluation_evidence_sha256,
        evaluation_receipt_sha256=promotion.evaluation_receipt_sha256,
        binding_sha256=_digest(unsigned),
    )


def build_runtime_stack_with_advanced_bindings(
    *,
    stack_id: str,
    advanced_bindings: Sequence[AdvancedRuntimeComponentBinding],
    other_components: Sequence[RuntimeComponent] = (),
    retrieval_contract_sha256: str,
    generation_contract_sha256: str,
    compatibility_sha256: str,
    source_revision: str,
) -> RuntimeStackArtifact:
    bindings = tuple(advanced_bindings)
    if not bindings or any(
        not isinstance(value, AdvancedRuntimeComponentBinding) for value in bindings
    ):
        raise ValueError(
            "advanced_bindings must be a non-empty AdvancedRuntimeComponentBinding sequence"
        )
    if len({value.artifact_sha256 for value in bindings}) != len(bindings):
        raise ValueError("advanced runtime bindings may not repeat an artifact")
    components = tuple(other_components) + tuple(value.component for value in bindings)
    return RuntimeStackArtifact.build(
        stack_id=stack_id,
        components=components,
        retrieval_contract_sha256=retrieval_contract_sha256,
        generation_contract_sha256=generation_contract_sha256,
        compatibility_sha256=compatibility_sha256,
        source_revision=source_revision,
    )


def advanced_offline_quality_evidence(
    stack: RuntimeStackArtifact,
    bindings: Sequence[AdvancedRuntimeComponentBinding],
    *,
    valid_from: float,
    expires_at: float | None = None,
) -> RuntimePromotionEvidence:
    """Bind all authoritative advanced qualifications into one exact stack evidence row."""
    if not isinstance(stack, RuntimeStackArtifact):
        raise ValueError("stack must be RuntimeStackArtifact")
    selected = tuple(
        sorted(bindings, key=lambda value: (value.component.kind, value.component.component_id))
    )
    if not selected or any(
        not isinstance(value, AdvancedRuntimeComponentBinding) for value in selected
    ):
        raise ValueError(
            "bindings must be a non-empty AdvancedRuntimeComponentBinding sequence"
        )
    stack_components = {
        (value.kind, value.component_id, value.artifact_sha256, value.contract_sha256)
        for value in stack.components
    }
    for binding in selected:
        component = binding.component
        if (
            component.kind,
            component.component_id,
            component.artifact_sha256,
            component.contract_sha256,
        ) not in stack_components:
            raise ValueError(
                "advanced qualification binding is not an exact component of the runtime stack"
            )
    evidence_sha = _digest(
        {
            "schema": "rigorousrag-advanced-runtime-offline-quality/v2",
            "stack_sha256": stack.stack_sha256,
            "bindings": [
                {
                    "binding_sha256": value.binding_sha256,
                    "promotion_evidence_sha256": value.promotion_evidence_sha256,
                    "primitive_promotion_evidence_sha256": value.primitive_promotion_evidence_sha256,
                    "authoritative_evaluation_evidence_sha256": value.authoritative_evaluation_evidence_sha256,
                    "evaluation_receipt_sha256": value.evaluation_receipt_sha256,
                }
                for value in selected
            ],
        }
    )
    return RuntimePromotionEvidence(
        kind="offline_quality",
        evidence_sha256=evidence_sha,
        stack_sha256=stack.stack_sha256,
        valid_from=valid_from,
        expires_at=expires_at,
    )


__all__ = [
    "AdvancedRuntimeComponentBinding",
    "advanced_offline_quality_evidence",
    "bind_qualified_advanced_artifact",
    "build_runtime_stack_with_advanced_bindings",
]
