"""Governed multimodal entailment adapters bound to page-coordinate evidence.

Model implementations are injected and must be pinned by ``ModelArtifactSpec``. Raw
model rationale text is never exposed. Each decision is instead bound to the exact
region/content digest and server-owned coordinate citation plus a small rationale-code
taxonomy suitable for audit and calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from tools.model_artifacts import ModelArtifactSpec
from tools.multimodal_evidence import (
    EvidenceRegion,
    PageCoordinateCitation,
    content_digest,
    region_citation,
)

_LABELS = {"entailed", "contradicted", "insufficient"}
_RATIONALES = {
    "direct_match",
    "partial_match",
    "numeric_match",
    "numeric_conflict",
    "visual_conflict",
    "missing_context",
    "unreadable",
    "model_uncertain",
}
_MAX_CLAIM = 20_000
_MAX_PAYLOAD_BYTES = 100_000_000


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a bounded string.")
    selected = value.strip()
    if (
        not selected
        or len(selected) > maximum
        or any(ord(ch) < 32 and ch not in "\t\r\n" for ch in selected)
    ):
        raise ValueError(f"{label} is invalid.")
    return selected


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be between 0 and 1.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be between 0 and 1.") from exc
    if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return selected


def _payload(value: Any) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError("region payload must be bytes.")
    selected = bytes(value)
    if not selected or len(selected) > _MAX_PAYLOAD_BYTES:
        raise ValueError("region payload is empty or exceeds the byte limit.")
    return selected


@dataclass(frozen=True)
class RegionPayload:
    region: EvidenceRegion
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.region, EvidenceRegion):
            raise ValueError("region must be EvidenceRegion.")
        selected = _payload(self.payload)
        if content_digest(selected) != self.region.content_sha256:
            raise ValueError("region payload does not match the authoritative content digest.")
        object.__setattr__(self, "payload", selected)


@dataclass(frozen=True)
class MultimodalEntailmentDecision:
    region_id: str
    citation: PageCoordinateCitation
    artifact_fingerprint: str
    label: str
    score: float
    rationale_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.citation, PageCoordinateCitation):
            raise ValueError("citation must be PageCoordinateCitation.")
        if self.region_id != self.citation.region_id:
            raise ValueError("decision region_id must match the coordinate citation.")
        if (
            not isinstance(self.artifact_fingerprint, str)
            or len(self.artifact_fingerprint) != 64
            or any(ch not in "0123456789abcdef" for ch in self.artifact_fingerprint)
        ):
            raise ValueError("artifact_fingerprint must be a SHA-256 digest.")
        if self.label not in _LABELS:
            raise ValueError("entailment label is unsupported.")
        object.__setattr__(self, "score", _unit(self.score, "score"))
        if self.rationale_code not in _RATIONALES:
            raise ValueError("rationale_code is unsupported.")


def _decision(
    spec: ModelArtifactSpec,
    region: EvidenceRegion,
    value: Any,
) -> MultimodalEntailmentDecision:
    if not isinstance(value, Mapping):
        raise RuntimeError("multimodal model returned an invalid decision mapping.")
    label = value.get("label")
    rationale = value.get("rationale_code")
    if label not in _LABELS or rationale not in _RATIONALES:
        raise RuntimeError("multimodal model returned an unsupported bounded decision.")
    try:
        score = _unit(value.get("score"), "model score")
    except ValueError as exc:
        raise RuntimeError("multimodal model returned an invalid score.") from exc
    return MultimodalEntailmentDecision(
        region_id=region.region_id,
        citation=region_citation(region),
        artifact_fingerprint=spec.artifact_fingerprint,
        label=label,
        score=score,
        rationale_code=rationale,
    )


class GovernedImageTextEntailmentAdapter:
    def __init__(
        self,
        spec: ModelArtifactSpec,
        infer: Callable[[str, bytes], Mapping[str, Any]],
    ) -> None:
        if not isinstance(spec, ModelArtifactSpec) or spec.kind != "image_text":
            raise ValueError("image-text adapter requires an image_text artifact.")
        if not callable(infer):
            raise ValueError("infer must be callable.")
        self.spec = spec
        self.artifact_fingerprint = spec.artifact_fingerprint
        self._infer = infer

    def evaluate(self, claim: str, evidence: RegionPayload) -> MultimodalEntailmentDecision:
        bounded_claim = _text(claim, "claim", _MAX_CLAIM)
        if not isinstance(evidence, RegionPayload):
            raise ValueError("evidence must be RegionPayload.")
        if evidence.region.kind not in {"figure", "chart"}:
            raise ValueError("image-text entailment requires a figure or chart region.")
        try:
            value = self._infer(bounded_claim, evidence.payload)
        except Exception as exc:
            raise RuntimeError("image-text entailment execution failed.") from exc
        return _decision(self.spec, evidence.region, value)


class GovernedTableChartEntailmentAdapter:
    def __init__(
        self,
        spec: ModelArtifactSpec,
        infer: Callable[[str, bytes, str], Mapping[str, Any]],
    ) -> None:
        if not isinstance(spec, ModelArtifactSpec) or spec.kind != "table_chart":
            raise ValueError("table/chart adapter requires a table_chart artifact.")
        if not callable(infer):
            raise ValueError("infer must be callable.")
        self.spec = spec
        self.artifact_fingerprint = spec.artifact_fingerprint
        self._infer = infer

    def evaluate(self, claim: str, evidence: RegionPayload) -> MultimodalEntailmentDecision:
        bounded_claim = _text(claim, "claim", _MAX_CLAIM)
        if not isinstance(evidence, RegionPayload):
            raise ValueError("evidence must be RegionPayload.")
        if evidence.region.kind not in {"table", "chart"}:
            raise ValueError("table/chart entailment requires a table or chart region.")
        try:
            value = self._infer(bounded_claim, evidence.payload, evidence.region.kind)
        except Exception as exc:
            raise RuntimeError("table/chart entailment execution failed.") from exc
        return _decision(self.spec, evidence.region, value)


__all__ = [
    "GovernedImageTextEntailmentAdapter",
    "GovernedTableChartEntailmentAdapter",
    "MultimodalEntailmentDecision",
    "RegionPayload",
]
