"""Closed scientific-evidence schemas for PICO/PECO and effect-size review."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Tuple


class QuestionFramework(str, Enum):
    PICO = "PICO"
    PECO = "PECO"


@dataclass(frozen=True)
class ResearchQuestion:
    question_id: str
    framework: QuestionFramework
    population: str
    intervention_or_exposure: str
    comparator: str = ""
    outcomes: Tuple[str, ...] = ()
    context: str = ""

    def __post_init__(self) -> None:
        if not self.question_id.strip() or not self.population.strip() or not self.intervention_or_exposure.strip():
            raise ValueError("question_id, population, and intervention/exposure are required.")
        if not self.outcomes:
            raise ValueError("at least one outcome is required.")
        if any(not outcome.strip() for outcome in self.outcomes):
            raise ValueError("outcomes must be non-empty strings.")


class EffectMeasure(str, Enum):
    RISK_RATIO = "risk_ratio"
    ODDS_RATIO = "odds_ratio"
    HAZARD_RATIO = "hazard_ratio"
    MEAN_DIFFERENCE = "mean_difference"
    STANDARDIZED_MEAN_DIFFERENCE = "standardized_mean_difference"
    CORRELATION = "correlation"
    PROPORTION = "proportion"
    OTHER = "other"


@dataclass(frozen=True)
class EffectEstimate:
    measure: EffectMeasure
    value: float
    confidence_low: Optional[float] = None
    confidence_high: Optional[float] = None
    confidence_level: float = 0.95
    unit: str = ""

    def __post_init__(self) -> None:
        values = [float(self.value), float(self.confidence_level)]
        if self.confidence_low is not None:
            values.append(float(self.confidence_low))
        if self.confidence_high is not None:
            values.append(float(self.confidence_high))
        if any(not math.isfinite(value) for value in values):
            raise ValueError("effect estimates must be finite.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1).")
        if (self.confidence_low is None) != (self.confidence_high is None):
            raise ValueError("confidence interval bounds must be supplied together.")
        if self.confidence_low is not None and self.confidence_low > self.confidence_high:
            raise ValueError("confidence_low may not exceed confidence_high.")
        if self.measure in {EffectMeasure.RISK_RATIO, EffectMeasure.ODDS_RATIO, EffectMeasure.HAZARD_RATIO} and self.value <= 0:
            raise ValueError("ratio effect measures must be positive.")
        if self.measure == EffectMeasure.CORRELATION and not -1.0 <= self.value <= 1.0:
            raise ValueError("correlation must be in [-1, 1].")
        if self.measure == EffectMeasure.PROPORTION and not 0.0 <= self.value <= 1.0:
            raise ValueError("proportion must be in [0, 1].")


class RiskLevel(str, Enum):
    LOW = "low"
    SOME_CONCERNS = "some_concerns"
    HIGH = "high"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class RiskOfBias:
    overall: RiskLevel
    randomization: RiskLevel = RiskLevel.UNCLEAR
    deviations: RiskLevel = RiskLevel.UNCLEAR
    missing_data: RiskLevel = RiskLevel.UNCLEAR
    outcome_measurement: RiskLevel = RiskLevel.UNCLEAR
    selective_reporting: RiskLevel = RiskLevel.UNCLEAR
    notes: str = ""


@dataclass(frozen=True)
class ScientificEvidenceRecord:
    evidence_id: str
    source_id: str
    question_id: str
    population: str
    intervention_or_exposure: str
    comparator: str
    outcome: str
    result_text: str
    effect: Optional[EffectEstimate] = None
    risk_of_bias: Optional[RiskOfBias] = None
    limitations: Tuple[str, ...] = ()
    provenance: Mapping[str, str] = field(default_factory=dict)
    reviewed: bool = False

    def __post_init__(self) -> None:
        required = (
            self.evidence_id,
            self.source_id,
            self.question_id,
            self.population,
            self.intervention_or_exposure,
            self.outcome,
            self.result_text,
        )
        if any(not value.strip() for value in required):
            raise ValueError("scientific evidence identity and core result fields are required.")


@dataclass(frozen=True)
class EvidenceConflict:
    outcome: str
    supporting_ids: Tuple[str, ...]
    contradicting_ids: Tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not self.outcome.strip() or not self.rationale.strip():
            raise ValueError("conflicts require an outcome and rationale.")
        if not self.supporting_ids or not self.contradicting_ids:
            raise ValueError("conflicts require both supporting and contradicting evidence.")
        if set(self.supporting_ids) & set(self.contradicting_ids):
            raise ValueError("the same evidence cannot appear on both sides of a conflict.")


def normalize_ratio_to_log(effect: EffectEstimate) -> float:
    if effect.measure not in {
        EffectMeasure.RISK_RATIO,
        EffectMeasure.ODDS_RATIO,
        EffectMeasure.HAZARD_RATIO,
    }:
        raise ValueError("only positive ratio measures have a canonical log transform.")
    return math.log(effect.value)
