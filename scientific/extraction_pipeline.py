"""Closed-schema scientific extraction pipeline into RigorousRAG evidence semantics.

The model call itself is injected because the repository supports multiple local/remote
providers, but everything around it is executable source: prompt/evidence packaging,
closed output validation, source-anchor resolution, PICO/PECO study extraction, effect
normalization records, risk-of-bias fields, review routing, and immutable extraction
receipts.  Provider output never becomes evidence unless every cited region resolves to
the authoritative structured document generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from scientific.document_structure import DocumentRegion, StructuredDocument
from scientific.evidence_semantics import (
    BiasDomainAssessment,
    BiasJudgement,
    CertaintyLevel,
    EffectEstimate,
    EffectMeasure,
    EvidenceQualityAssessment,
    EvidenceSpan,
    ExtractedValue,
    QuestionFramework,
    StructuredResearchQuestion,
    StudyDescriptor,
    StudyEvidenceBundle,
)


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _text(value: Any, label: str, maximum: int = 100_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or "\x00" in selected:
        raise ValueError(f"{label} is invalid")
    return selected


def _probability(value: Any, label: str) -> float:
    selected = float(value)
    if not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be in [0,1]")
    return selected


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


SCIENTIFIC_EXTRACTION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["study_id", "title", "methods", "populations", "interventions_or_exposures", "comparators", "outcomes", "limitations", "effects", "risk_of_bias"],
    "additionalProperties": False,
    "properties": {
        "study_id": {"type": "string"},
        "title": {"$ref": "#/definitions/extracted"},
        "methods": {"type": "array", "items": {"$ref": "#/definitions/extracted"}},
        "populations": {"type": "array", "items": {"$ref": "#/definitions/extracted"}},
        "interventions_or_exposures": {"type": "array", "items": {"$ref": "#/definitions/extracted"}},
        "comparators": {"type": "array", "items": {"$ref": "#/definitions/extracted"}},
        "outcomes": {"type": "array", "items": {"$ref": "#/definitions/extracted"}},
        "limitations": {"type": "array", "items": {"$ref": "#/definitions/extracted"}},
        "effects": {"type": "array", "items": {"type": "object"}},
        "risk_of_bias": {"type": "array", "items": {"type": "object"}},
        "certainty": {"type": "string"},
    },
    "definitions": {
        "extracted": {
            "type": "object",
            "required": ["value", "confidence", "region_ids"],
            "additionalProperties": False,
            "properties": {
                "value": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "region_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            },
        }
    },
}


class ScientificJsonProvider(Protocol):
    @property
    def provider_identity(self) -> str: ...

    def extract_json(
        self,
        *,
        instruction: str,
        evidence_blocks: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class CallableScientificJsonProvider:
    """Concrete adapter for any already-configured local/remote structured-output callable."""

    def __init__(
        self,
        identity: str,
        function: Callable[[str, Sequence[Mapping[str, Any]], Mapping[str, Any]], Mapping[str, Any]],
    ) -> None:
        self._identity = _identifier(identity, "provider identity")
        if not callable(function):
            raise ValueError("function must be callable")
        self._function = function

    @property
    def provider_identity(self) -> str:
        return self._identity

    def extract_json(
        self,
        *,
        instruction: str,
        evidence_blocks: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        result = self._function(instruction, evidence_blocks, schema)
        if not isinstance(result, Mapping):
            raise ValueError("scientific extraction provider must return an object")
        return result


@dataclass(frozen=True)
class ExtractionPolicy:
    minimum_auto_accept_confidence: float = 0.85
    max_evidence_regions_per_value: int = 20
    max_regions_in_prompt: int = 5_000
    max_chars_per_region: int = 20_000
    require_human_review_for_risk_of_bias: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_auto_accept_confidence",
            _probability(self.minimum_auto_accept_confidence, "minimum_auto_accept_confidence"),
        )
        for name in ("max_evidence_regions_per_value", "max_regions_in_prompt", "max_chars_per_region"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.require_human_review_for_risk_of_bias, bool):
            raise ValueError("require_human_review_for_risk_of_bias must be boolean")


@dataclass(frozen=True)
class ReviewIssue:
    code: str
    record_path: str
    reason: str
    severity: str = "review"

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _identifier(self.code, "review code", 500))
        object.__setattr__(self, "record_path", _identifier(self.record_path, "record path", 2_000))
        object.__setattr__(self, "reason", _text(self.reason, "review reason", 20_000))
        if self.severity not in {"review", "block"}:
            raise ValueError("severity must be review or block")


@dataclass(frozen=True)
class ExtractionReceipt:
    provider_identity: str
    structured_document_digest: str
    question_digest: str
    raw_response_digest: str
    bundle_digest: str
    review_issue_count: int

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True)
class ExtractionResult:
    bundle: StudyEvidenceBundle
    review_issues: tuple[ReviewIssue, ...]
    receipt: ExtractionReceipt

    @property
    def blocked(self) -> bool:
        return any(issue.severity == "block" for issue in self.review_issues)


class JsonlReviewQueue:
    """Append-only local review queue with no raw-document duplication."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def enqueue(self, result: ExtractionResult) -> str:
        payload = {
            "receipt_digest": result.receipt.digest,
            "bundle_digest": result.bundle.digest,
            "study_id": result.bundle.study.study_id,
            "question_id": result.bundle.question.question_id,
            "issues": [asdict(issue) for issue in result.review_issues],
        }
        line = _canonical(payload) + b"\n"
        # O_APPEND makes each bounded record append atomic on normal local filesystems.
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, line)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return hashlib.sha256(line).hexdigest()


def _region_map(document: StructuredDocument) -> dict[str, DocumentRegion]:
    return {region.region_id: region for region in document.regions}


def _evidence_span(document: StructuredDocument, region: DocumentRegion) -> EvidenceSpan:
    quote_digest = None
    if region.text:
        quote_digest = hashlib.sha256(region.text.encode("utf-8")).hexdigest()
    return EvidenceSpan(
        document_id=document.document_id,
        generation_id=document.generation_id,
        page=region.anchor.page,
        block_id=region.region_id,
        quote_digest=quote_digest,
    )


def _resolve_regions(
    document: StructuredDocument,
    region_ids: Any,
    *,
    policy: ExtractionPolicy,
    path: str,
) -> tuple[tuple[EvidenceSpan, ...], list[ReviewIssue]]:
    issues: list[ReviewIssue] = []
    if not isinstance(region_ids, list) or not region_ids:
        raise ValueError(f"{path}.region_ids must be a non-empty array")
    if len(region_ids) > policy.max_evidence_regions_per_value:
        raise ValueError(f"{path}.region_ids exceeds policy")
    by_id = _region_map(document)
    spans: list[EvidenceSpan] = []
    seen: set[str] = set()
    for raw_id in region_ids:
        region_id = _identifier(raw_id, f"{path}.region_id")
        if region_id in seen:
            continue
        seen.add(region_id)
        region = by_id.get(region_id)
        if region is None:
            issues.append(ReviewIssue("unknown_evidence_region", path, f"provider cited unknown region {region_id}", "block"))
            continue
        spans.append(_evidence_span(document, region))
    if not spans:
        issues.append(ReviewIssue("no_resolved_evidence", path, "no cited evidence region resolves", "block"))
    return tuple(spans), issues


def _extracted(
    value: Any,
    *,
    document: StructuredDocument,
    provider_identity: str,
    policy: ExtractionPolicy,
    path: str,
) -> tuple[ExtractedValue, list[ReviewIssue]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    if set(value) != {"value", "confidence", "region_ids"}:
        raise ValueError(f"{path} contains missing or unknown keys")
    confidence = _probability(value["confidence"], f"{path}.confidence")
    spans, issues = _resolve_regions(document, value["region_ids"], policy=policy, path=path)
    if confidence < policy.minimum_auto_accept_confidence:
        issues.append(ReviewIssue("low_extraction_confidence", path, f"confidence {confidence:.4f} is below auto-accept threshold"))
    return (
        ExtractedValue(
            value=_text(value["value"], f"{path}.value"),
            confidence=confidence,
            evidence=spans,
            extractor=provider_identity,
        ),
        issues,
    )


def _extracted_array(
    values: Any,
    *,
    document: StructuredDocument,
    provider_identity: str,
    policy: ExtractionPolicy,
    path: str,
) -> tuple[tuple[ExtractedValue, ...], list[ReviewIssue]]:
    if not isinstance(values, list) or len(values) > 10_000:
        raise ValueError(f"{path} must be a bounded array")
    records: list[ExtractedValue] = []
    issues: list[ReviewIssue] = []
    for index, value in enumerate(values):
        record, record_issues = _extracted(
            value,
            document=document,
            provider_identity=provider_identity,
            policy=policy,
            path=f"{path}[{index}]",
        )
        records.append(record)
        issues.extend(record_issues)
    return tuple(records), issues


def _effect(
    value: Any,
    *,
    study_id: str,
    document: StructuredDocument,
    policy: ExtractionPolicy,
    path: str,
) -> tuple[EffectEstimate, list[ReviewIssue]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    allowed = {
        "effect_id", "outcome", "measure", "estimate", "confidence_level", "ci_lower", "ci_upper",
        "standard_error", "sample_size", "events", "region_ids", "notes"
    }
    unknown = set(value) - allowed
    required = {"effect_id", "outcome", "measure", "estimate", "region_ids"}
    if unknown or not required <= set(value):
        raise ValueError(f"{path} has missing/unknown effect fields")
    evidence, issues = _resolve_regions(document, value["region_ids"], policy=policy, path=path)
    effect = EffectEstimate(
        effect_id=value["effect_id"],
        study_id=study_id,
        outcome=value["outcome"],
        measure=EffectMeasure(value["measure"]),
        estimate=value["estimate"],
        confidence_level=value.get("confidence_level", 0.95),
        ci_lower=value.get("ci_lower"),
        ci_upper=value.get("ci_upper"),
        standard_error=value.get("standard_error"),
        sample_size=value.get("sample_size"),
        events=value.get("events"),
        evidence=evidence,
        notes=value.get("notes"),
    )
    if effect.ci_lower is None and effect.standard_error is None:
        issues.append(ReviewIssue("effect_uncertainty_missing", path, "effect has neither confidence interval nor standard error"))
    return effect, issues


def _bias(
    value: Any,
    *,
    document: StructuredDocument,
    policy: ExtractionPolicy,
    provider_identity: str,
    path: str,
) -> tuple[BiasDomainAssessment, list[ReviewIssue]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    allowed = {"domain", "judgement", "rationale", "region_ids", "tool_name", "tool_version"}
    required = {"domain", "judgement", "rationale", "region_ids"}
    if set(value) - allowed or not required <= set(value):
        raise ValueError(f"{path} has missing/unknown bias fields")
    evidence, issues = _resolve_regions(document, value["region_ids"], policy=policy, path=path)
    record = BiasDomainAssessment(
        domain=value["domain"],
        judgement=BiasJudgement(value["judgement"]),
        rationale=value["rationale"],
        evidence=evidence,
        assessor=provider_identity,
        tool_name=value.get("tool_name", "model-assisted-structured-extraction"),
        tool_version=value.get("tool_version", "1"),
        human_reviewed=False,
    )
    if policy.require_human_review_for_risk_of_bias:
        issues.append(ReviewIssue("risk_of_bias_requires_human_review", path, "automated risk-of-bias judgement requires human review"))
    return record, issues


def evidence_blocks(document: StructuredDocument, policy: ExtractionPolicy) -> tuple[Mapping[str, Any], ...]:
    blocks: list[Mapping[str, Any]] = []
    for region_id in document.reading_order[: policy.max_regions_in_prompt]:
        region = next(value for value in document.regions if value.region_id == region_id)
        if not region.text:
            continue
        blocks.append(
            {
                "region_id": region.region_id,
                "page": region.anchor.page,
                "kind": region.kind.value,
                "text": region.text[: policy.max_chars_per_region],
            }
        )
    return tuple(blocks)


def extract_study_evidence(
    document: StructuredDocument,
    question: StructuredResearchQuestion,
    provider: ScientificJsonProvider,
    *,
    policy: ExtractionPolicy = ExtractionPolicy(),
) -> ExtractionResult:
    if not isinstance(document, StructuredDocument):
        raise ValueError("document must be StructuredDocument")
    if not isinstance(question, StructuredResearchQuestion):
        raise ValueError("question must be StructuredResearchQuestion")
    provider_identity = _identifier(provider.provider_identity, "provider identity")
    instruction = (
        "Extract only claims explicitly supported by the supplied evidence blocks. "
        "For every field cite authoritative region_ids. Do not infer missing values. "
        f"Research question framework={question.framework.value}; population={question.population}; "
        f"intervention_or_exposure={question.intervention_or_exposure}; comparator={question.comparator}; "
        f"outcome={question.outcome}."
    )
    blocks = evidence_blocks(document, policy)
    raw = provider.extract_json(instruction=instruction, evidence_blocks=blocks, schema=SCIENTIFIC_EXTRACTION_SCHEMA)
    if not isinstance(raw, Mapping):
        raise ValueError("provider result must be an object")
    required = {
        "study_id", "title", "methods", "populations", "interventions_or_exposures", "comparators",
        "outcomes", "limitations", "effects", "risk_of_bias"
    }
    allowed = required | {"certainty"}
    if not required <= set(raw) or set(raw) - allowed:
        raise ValueError("provider response violates closed scientific extraction schema")

    issues: list[ReviewIssue] = []
    title, title_issues = _extracted(raw["title"], document=document, provider_identity=provider_identity, policy=policy, path="title")
    issues.extend(title_issues)
    extracted_groups: dict[str, tuple[ExtractedValue, ...]] = {}
    for name in ("methods", "populations", "interventions_or_exposures", "comparators", "outcomes", "limitations"):
        records, record_issues = _extracted_array(
            raw[name],
            document=document,
            provider_identity=provider_identity,
            policy=policy,
            path=name,
        )
        extracted_groups[name] = records
        issues.extend(record_issues)

    study_id = _identifier(raw["study_id"], "study_id")
    study = StudyDescriptor(
        study_id=study_id,
        title=title,
        methods=extracted_groups["methods"],
        populations=extracted_groups["populations"],
        interventions_or_exposures=extracted_groups["interventions_or_exposures"],
        comparators=extracted_groups["comparators"],
        outcomes=extracted_groups["outcomes"],
        limitations=extracted_groups["limitations"],
    )
    effects: list[EffectEstimate] = []
    if not isinstance(raw["effects"], list) or len(raw["effects"]) > 10_000:
        raise ValueError("effects must be a bounded array")
    for index, value in enumerate(raw["effects"]):
        effect, effect_issues = _effect(value, study_id=study_id, document=document, policy=policy, path=f"effects[{index}]")
        effects.append(effect)
        issues.extend(effect_issues)

    bias: list[BiasDomainAssessment] = []
    if not isinstance(raw["risk_of_bias"], list) or len(raw["risk_of_bias"]) > 100:
        raise ValueError("risk_of_bias must be a bounded array")
    for index, value in enumerate(raw["risk_of_bias"]):
        assessment, assessment_issues = _bias(
            value,
            document=document,
            policy=policy,
            provider_identity=provider_identity,
            path=f"risk_of_bias[{index}]",
        )
        bias.append(assessment)
        issues.extend(assessment_issues)
    quality = EvidenceQualityAssessment(
        study_id=study_id,
        certainty=CertaintyLevel(raw.get("certainty", CertaintyLevel.UNASSESSED.value)),
        risk_of_bias=tuple(bias),
        human_review_required=True,
    )
    bundle = StudyEvidenceBundle(
        question=question,
        study=study,
        effects=tuple(effects),
        quality=quality,
        metadata={
            "structured_document_digest": document.digest,
            "provider_identity": provider_identity,
        },
    )
    receipt = ExtractionReceipt(
        provider_identity=provider_identity,
        structured_document_digest=document.digest,
        question_digest=question.digest,
        raw_response_digest=canonical_digest(raw),
        bundle_digest=bundle.digest,
        review_issue_count=len(issues),
    )
    return ExtractionResult(bundle, tuple(issues), receipt)


__all__ = [
    "CallableScientificJsonProvider",
    "ExtractionPolicy",
    "ExtractionReceipt",
    "ExtractionResult",
    "JsonlReviewQueue",
    "ReviewIssue",
    "SCIENTIFIC_EXTRACTION_SCHEMA",
    "ScientificJsonProvider",
    "canonical_digest",
    "evidence_blocks",
    "extract_study_evidence",
]
