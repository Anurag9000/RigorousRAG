"""Closed-schema scientific evidence semantics with provenance and review lineage.

The extraction layer may propose values, but this module keeps the downstream evidence
model explicit and reviewable.  It supports PICO/PECO-style questions, methods/results
fields, normalized effect estimates, risk-of-bias assessments, evidence relationships,
and immutable human corrections.  Nothing here claims that an automated extraction or
risk assessment is scientifically valid without the cited evidence and required review.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence

_MAX_TEXT = 100_000
_MAX_ITEMS = 10_000
_EPS = 1e-12


def _text(value: Any, label: str, maximum: int = _MAX_TEXT, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if (not result and not allow_empty) or len(result) > maximum:
        raise ValueError(f"{label} is empty or too long")
    if any(ord(character) == 0 for character in result):
        raise ValueError(f"{label} contains NUL")
    return result


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    result = _text(value, label, maximum)
    if any(ord(character) < 32 or ord(character) == 127 for character in result):
        raise ValueError(f"{label} contains control characters")
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _probability(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return result


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class QuestionFramework(str, Enum):
    PICO = "pico"
    PECO = "peco"
    PICOS = "picos"
    FREEFORM = "freeform"


class EffectMeasure(str, Enum):
    RISK_RATIO = "risk_ratio"
    ODDS_RATIO = "odds_ratio"
    HAZARD_RATIO = "hazard_ratio"
    RATE_RATIO = "rate_ratio"
    RISK_DIFFERENCE = "risk_difference"
    MEAN_DIFFERENCE = "mean_difference"
    STANDARDIZED_MEAN_DIFFERENCE = "standardized_mean_difference"
    CORRELATION = "correlation"
    PROPORTION = "proportion"
    GENERIC = "generic"


class BiasJudgement(str, Enum):
    LOW = "low"
    SOME_CONCERNS = "some_concerns"
    HIGH = "high"
    CRITICAL = "critical"
    UNASSESSED = "unassessed"


class CertaintyLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"
    UNASSESSED = "unassessed"


class EvidenceRelation(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    QUALIFIES = "qualifies"
    LIMITS = "limits"
    CONFLICTS_WITH = "conflicts_with"
    DERIVED_FROM = "derived_from"


@dataclass(frozen=True)
class EvidenceSpan:
    """Precise source anchor used for every extracted or reviewed scientific field."""

    document_id: str
    generation_id: str
    page: int | None = None
    section: str | None = None
    block_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    quote_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _identifier(self.document_id, "document_id"))
        object.__setattr__(self, "generation_id", _identifier(self.generation_id, "generation_id"))
        if self.page is not None and (isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 1):
            raise ValueError("page must be a positive integer")
        for name in ("section", "block_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name, 2_000))
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must be supplied together")
        if self.char_start is not None:
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.char_start, self.char_end)):
                raise ValueError("character offsets must be integers")
            if self.char_start < 0 or self.char_end <= self.char_start:
                raise ValueError("character offsets are invalid")
        if self.quote_digest is not None:
            digest = _identifier(self.quote_digest, "quote_digest", 64).lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("quote_digest must be a SHA-256 hex digest")
            object.__setattr__(self, "quote_digest", digest)


@dataclass(frozen=True)
class ExtractedValue:
    """A proposed structured value with confidence and mandatory evidence anchors."""

    value: str
    confidence: float
    evidence: tuple[EvidenceSpan, ...]
    extractor: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _text(self.value, "value"))
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))
        if not self.evidence or len(self.evidence) > 100:
            raise ValueError("extracted values require bounded evidence")
        if any(not isinstance(span, EvidenceSpan) for span in self.evidence):
            raise ValueError("evidence must contain EvidenceSpan values")
        object.__setattr__(self, "extractor", _identifier(self.extractor, "extractor", 1_000))
        object.__setattr__(self, "schema_version", _identifier(self.schema_version, "schema_version", 100))


@dataclass(frozen=True)
class StructuredResearchQuestion:
    question_id: str
    framework: QuestionFramework
    population: str
    outcome: str
    intervention_or_exposure: str | None = None
    comparator: str | None = None
    study_design: str | None = None
    setting: str | None = None
    time_horizon: str | None = None
    freeform_question: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_id", _identifier(self.question_id, "question_id"))
        if not isinstance(self.framework, QuestionFramework):
            object.__setattr__(self, "framework", QuestionFramework(self.framework))
        object.__setattr__(self, "population", _text(self.population, "population"))
        object.__setattr__(self, "outcome", _text(self.outcome, "outcome"))
        for name in (
            "intervention_or_exposure",
            "comparator",
            "study_design",
            "setting",
            "time_horizon",
            "freeform_question",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        if self.framework != QuestionFramework.FREEFORM and not self.intervention_or_exposure:
            raise ValueError("PICO/PECO questions require an intervention or exposure")
        if self.framework == QuestionFramework.FREEFORM and not self.freeform_question:
            raise ValueError("freeform questions require freeform_question")
        if self.framework == QuestionFramework.PICOS and not self.study_design:
            raise ValueError("PICOS questions require study_design")

    @property
    def digest(self) -> str:
        return _canonical_digest(asdict(self))


@dataclass(frozen=True)
class StudyDescriptor:
    study_id: str
    title: ExtractedValue
    methods: tuple[ExtractedValue, ...] = ()
    populations: tuple[ExtractedValue, ...] = ()
    interventions_or_exposures: tuple[ExtractedValue, ...] = ()
    comparators: tuple[ExtractedValue, ...] = ()
    outcomes: tuple[ExtractedValue, ...] = ()
    limitations: tuple[ExtractedValue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _identifier(self.study_id, "study_id"))
        if not isinstance(self.title, ExtractedValue):
            raise ValueError("title must be ExtractedValue")
        for name in (
            "methods",
            "populations",
            "interventions_or_exposures",
            "comparators",
            "outcomes",
            "limitations",
        ):
            values = getattr(self, name)
            if len(values) > _MAX_ITEMS or any(not isinstance(value, ExtractedValue) for value in values):
                raise ValueError(f"{name} must contain bounded ExtractedValue records")


_RATIO_MEASURES = {
    EffectMeasure.RISK_RATIO,
    EffectMeasure.ODDS_RATIO,
    EffectMeasure.HAZARD_RATIO,
    EffectMeasure.RATE_RATIO,
}


@dataclass(frozen=True)
class EffectEstimate:
    """Study effect plus enough uncertainty information to support reproducible synthesis."""

    effect_id: str
    study_id: str
    outcome: str
    measure: EffectMeasure
    estimate: float
    confidence_level: float = 0.95
    ci_lower: float | None = None
    ci_upper: float | None = None
    standard_error: float | None = None
    sample_size: int | None = None
    events: int | None = None
    evidence: tuple[EvidenceSpan, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect_id", _identifier(self.effect_id, "effect_id"))
        object.__setattr__(self, "study_id", _identifier(self.study_id, "study_id"))
        object.__setattr__(self, "outcome", _text(self.outcome, "outcome"))
        if not isinstance(self.measure, EffectMeasure):
            object.__setattr__(self, "measure", EffectMeasure(self.measure))
        estimate = _finite(self.estimate, "estimate")
        if self.measure in _RATIO_MEASURES and estimate <= 0.0:
            raise ValueError("ratio estimates must be positive")
        if self.measure == EffectMeasure.PROPORTION and not 0.0 <= estimate <= 1.0:
            raise ValueError("proportions must be between 0 and 1")
        if self.measure == EffectMeasure.CORRELATION and not -1.0 <= estimate <= 1.0:
            raise ValueError("correlations must be between -1 and 1")
        object.__setattr__(self, "estimate", estimate)
        level = _probability(self.confidence_level, "confidence_level")
        if not 0.5 < level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")
        object.__setattr__(self, "confidence_level", level)
        if (self.ci_lower is None) != (self.ci_upper is None):
            raise ValueError("confidence interval bounds must be supplied together")
        if self.ci_lower is not None:
            lower = _finite(self.ci_lower, "ci_lower")
            upper = _finite(self.ci_upper, "ci_upper")
            if not lower < upper:
                raise ValueError("confidence interval must be increasing")
            if self.measure in _RATIO_MEASURES and lower <= 0.0:
                raise ValueError("ratio confidence intervals must be positive")
            object.__setattr__(self, "ci_lower", lower)
            object.__setattr__(self, "ci_upper", upper)
        if self.standard_error is not None:
            standard_error = _finite(self.standard_error, "standard_error")
            if standard_error <= 0.0:
                raise ValueError("standard_error must be positive")
            object.__setattr__(self, "standard_error", standard_error)
        for name in ("sample_size", "events"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.sample_size is not None and self.events is not None and self.events > self.sample_size:
            raise ValueError("events cannot exceed sample_size")
        if len(self.evidence) > 100 or any(not isinstance(span, EvidenceSpan) for span in self.evidence):
            raise ValueError("evidence must contain bounded EvidenceSpan values")
        if self.notes is not None:
            object.__setattr__(self, "notes", _text(self.notes, "notes"))

    @property
    def synthesis_scale(self) -> float:
        """Normalize common effects to an additive scale used by meta-analysis engines."""

        if self.measure in _RATIO_MEASURES:
            return math.log(self.estimate)
        if self.measure == EffectMeasure.CORRELATION:
            clipped = min(max(self.estimate, -1.0 + _EPS), 1.0 - _EPS)
            return 0.5 * math.log((1.0 + clipped) / (1.0 - clipped))
        return self.estimate

    def synthesis_interval(self) -> tuple[float, float] | None:
        if self.ci_lower is None or self.ci_upper is None:
            return None
        if self.measure in _RATIO_MEASURES:
            return math.log(self.ci_lower), math.log(self.ci_upper)
        if self.measure == EffectMeasure.CORRELATION:
            def fisher(value: float) -> float:
                clipped = min(max(value, -1.0 + _EPS), 1.0 - _EPS)
                return 0.5 * math.log((1.0 + clipped) / (1.0 - clipped))
            return fisher(self.ci_lower), fisher(self.ci_upper)
        return self.ci_lower, self.ci_upper


@dataclass(frozen=True)
class BiasDomainAssessment:
    domain: str
    judgement: BiasJudgement
    rationale: str
    evidence: tuple[EvidenceSpan, ...]
    assessor: str
    tool_name: str
    tool_version: str
    human_reviewed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _text(self.domain, "domain", 1_000))
        if not isinstance(self.judgement, BiasJudgement):
            object.__setattr__(self, "judgement", BiasJudgement(self.judgement))
        object.__setattr__(self, "rationale", _text(self.rationale, "rationale"))
        if not self.evidence or len(self.evidence) > 100 or any(
            not isinstance(span, EvidenceSpan) for span in self.evidence
        ):
            raise ValueError("risk-of-bias assessments require bounded evidence")
        for name in ("assessor", "tool_name", "tool_version"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name, 1_000))
        if not isinstance(self.human_reviewed, bool):
            raise ValueError("human_reviewed must be boolean")


@dataclass(frozen=True)
class EvidenceQualityAssessment:
    study_id: str
    certainty: CertaintyLevel
    risk_of_bias: tuple[BiasDomainAssessment, ...]
    inconsistency: str | None = None
    indirectness: str | None = None
    imprecision: str | None = None
    publication_bias: str | None = None
    human_review_required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _identifier(self.study_id, "study_id"))
        if not isinstance(self.certainty, CertaintyLevel):
            object.__setattr__(self, "certainty", CertaintyLevel(self.certainty))
        if len(self.risk_of_bias) > 100 or any(
            not isinstance(value, BiasDomainAssessment) for value in self.risk_of_bias
        ):
            raise ValueError("risk_of_bias must contain bounded domain assessments")
        for name in ("inconsistency", "indirectness", "imprecision", "publication_bias"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        if not isinstance(self.human_review_required, bool):
            raise ValueError("human_review_required must be boolean")
        if not self.human_review_required and any(not item.human_reviewed for item in self.risk_of_bias):
            raise ValueError("human_review_required may be cleared only after every bias domain is human reviewed")


@dataclass(frozen=True)
class EvidenceLink:
    link_id: str
    source_id: str
    target_id: str
    relation: EvidenceRelation
    rationale: str
    evidence: tuple[EvidenceSpan, ...] = ()

    def __post_init__(self) -> None:
        for name in ("link_id", "source_id", "target_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.source_id == self.target_id:
            raise ValueError("evidence links may not self-reference")
        if not isinstance(self.relation, EvidenceRelation):
            object.__setattr__(self, "relation", EvidenceRelation(self.relation))
        object.__setattr__(self, "rationale", _text(self.rationale, "rationale"))
        if len(self.evidence) > 100 or any(not isinstance(span, EvidenceSpan) for span in self.evidence):
            raise ValueError("evidence must contain bounded EvidenceSpan values")


@dataclass(frozen=True)
class CorrectionRecord:
    correction_id: str
    record_id: str
    field_path: str
    previous_value_digest: str
    replacement_value: str
    reviewer_id: str
    reason: str
    evidence: tuple[EvidenceSpan, ...]
    parent_correction_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("correction_id", "record_id", "reviewer_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "field_path", _text(self.field_path, "field_path", 2_000))
        digest = _identifier(self.previous_value_digest, "previous_value_digest", 64).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("previous_value_digest must be a SHA-256 hex digest")
        object.__setattr__(self, "previous_value_digest", digest)
        object.__setattr__(self, "replacement_value", _text(self.replacement_value, "replacement_value"))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        if not self.evidence or len(self.evidence) > 100 or any(
            not isinstance(span, EvidenceSpan) for span in self.evidence
        ):
            raise ValueError("corrections require bounded evidence")
        if self.parent_correction_id is not None:
            object.__setattr__(
                self,
                "parent_correction_id",
                _identifier(self.parent_correction_id, "parent_correction_id"),
            )

    @property
    def digest(self) -> str:
        return _canonical_digest(asdict(self))


def apply_extracted_value_correction(
    value: ExtractedValue,
    correction: CorrectionRecord,
    *,
    reviewer_extractor_prefix: str = "human-review",
) -> ExtractedValue:
    """Create a new reviewed value; never mutate the original extraction in place."""

    if not isinstance(value, ExtractedValue) or not isinstance(correction, CorrectionRecord):
        raise ValueError("value and correction have invalid types")
    expected = _canonical_digest(asdict(value))
    if correction.previous_value_digest != expected:
        raise ValueError("correction does not target the supplied extraction version")
    merged_evidence = tuple(dict.fromkeys((*value.evidence, *correction.evidence)))
    return replace(
        value,
        value=correction.replacement_value,
        confidence=1.0,
        evidence=merged_evidence,
        extractor=f"{_identifier(reviewer_extractor_prefix, 'reviewer_extractor_prefix')}:{correction.reviewer_id}",
    )


@dataclass(frozen=True)
class StudyEvidenceBundle:
    question: StructuredResearchQuestion
    study: StudyDescriptor
    effects: tuple[EffectEstimate, ...] = ()
    quality: EvidenceQualityAssessment | None = None
    links: tuple[EvidenceLink, ...] = ()
    corrections: tuple[CorrectionRecord, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.question, StructuredResearchQuestion):
            raise ValueError("question must be StructuredResearchQuestion")
        if not isinstance(self.study, StudyDescriptor):
            raise ValueError("study must be StudyDescriptor")
        if len(self.effects) > _MAX_ITEMS or any(not isinstance(value, EffectEstimate) for value in self.effects):
            raise ValueError("effects must contain bounded EffectEstimate records")
        if any(value.study_id != self.study.study_id for value in self.effects):
            raise ValueError("all effects must belong to the bundled study")
        if self.quality is not None:
            if not isinstance(self.quality, EvidenceQualityAssessment):
                raise ValueError("quality has invalid type")
            if self.quality.study_id != self.study.study_id:
                raise ValueError("quality assessment must belong to the bundled study")
        if len(self.links) > _MAX_ITEMS or any(not isinstance(value, EvidenceLink) for value in self.links):
            raise ValueError("links must contain bounded EvidenceLink records")
        if len(self.corrections) > _MAX_ITEMS or any(
            not isinstance(value, CorrectionRecord) for value in self.corrections
        ):
            raise ValueError("corrections must contain bounded CorrectionRecord values")
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 1_000:
            raise ValueError("metadata must be a bounded mapping")
        cleaned: dict[str, str] = {}
        for key, value in self.metadata.items():
            cleaned[_identifier(key, "metadata key", 200)] = _text(value, "metadata value", 10_000)
        object.__setattr__(self, "metadata", cleaned)

    @property
    def digest(self) -> str:
        return _canonical_digest(asdict(self))


__all__ = [
    "BiasDomainAssessment",
    "BiasJudgement",
    "CertaintyLevel",
    "CorrectionRecord",
    "EffectEstimate",
    "EffectMeasure",
    "EvidenceLink",
    "EvidenceQualityAssessment",
    "EvidenceRelation",
    "EvidenceSpan",
    "ExtractedValue",
    "QuestionFramework",
    "StructuredResearchQuestion",
    "StudyDescriptor",
    "StudyEvidenceBundle",
    "apply_extracted_value_correction",
]
