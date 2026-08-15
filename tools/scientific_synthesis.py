"""Deterministic scientific evidence schemas and quantitative synthesis primitives.

This module deliberately separates *representation and arithmetic* from extraction or
clinical/scientific judgment.  It does not download data, train models, infer missing
statistics, or declare evidence trustworthy.  Callers must supply governed, reviewed
study records and compatible effect estimates.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

_MAX_TEXT = 4000
_MAX_ITEMS = 256
_EFFECT_TYPES = frozenset(
    {"risk_ratio", "odds_ratio", "hazard_ratio", "mean_difference", "standardized_mean_difference", "correlation", "proportion"}
)
_RISK_LEVELS = frozenset({"low", "some_concerns", "high", "unclear", "not_assessed"})


def _text(value: Any, label: str, maximum: int = _MAX_TEXT, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _finite(value: Any, label: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    if positive and parsed <= 0:
        raise ValueError(f"{label} must be positive")
    if nonnegative and parsed < 0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class ResearchQuestion:
    """PICO/PECO/PICOS-compatible normalized research question."""

    population: str
    intervention_or_exposure: str
    comparator: str
    outcomes: tuple[str, ...]
    study_designs: tuple[str, ...] = ()
    question_type: str = "PICO"
    context: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "population", _text(self.population, "population"))
        object.__setattr__(self, "intervention_or_exposure", _text(self.intervention_or_exposure, "intervention_or_exposure"))
        object.__setattr__(self, "comparator", _text(self.comparator, "comparator", allow_empty=True))
        if not 1 <= len(self.outcomes) <= 64:
            raise ValueError("outcomes must contain between 1 and 64 items")
        object.__setattr__(self, "outcomes", tuple(dict.fromkeys(_text(item, "outcome", 500) for item in self.outcomes)))
        if len(self.study_designs) > 32:
            raise ValueError("study_designs exceeds the item limit")
        object.__setattr__(self, "study_designs", tuple(dict.fromkeys(_text(item, "study_design", 300) for item in self.study_designs)))
        qt = _text(self.question_type, "question_type", 16).upper()
        if qt not in {"PICO", "PECO", "PICOS"}:
            raise ValueError("question_type must be PICO, PECO, or PICOS")
        object.__setattr__(self, "question_type", qt)
        object.__setattr__(self, "context", _text(self.context, "context", 2000, allow_empty=True))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class RiskOfBias:
    randomization: str = "not_assessed"
    deviations: str = "not_assessed"
    missing_data: str = "not_assessed"
    outcome_measurement: str = "not_assessed"
    selective_reporting: str = "not_assessed"
    confounding: str = "not_assessed"
    overall: str = "not_assessed"
    rationale_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "randomization", "deviations", "missing_data", "outcome_measurement",
            "selective_reporting", "confounding", "overall",
        ):
            value = _text(getattr(self, name), name, 32).lower()
            if value not in _RISK_LEVELS:
                raise ValueError(f"unsupported risk-of-bias level for {name}")
            object.__setattr__(self, name, value)
        digest = self.rationale_sha256.lower().strip() if isinstance(self.rationale_sha256, str) else ""
        if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError("rationale_sha256 is invalid")
        object.__setattr__(self, "rationale_sha256", digest)


@dataclass(frozen=True)
class StudyEvidence:
    study_id: str
    population: str
    intervention_or_exposure: str
    comparator: str
    outcome: str
    study_design: str
    sample_size: int
    follow_up: str = ""
    methods: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    funding: str = ""
    conflicts: str = ""
    risk_of_bias: RiskOfBias = field(default_factory=RiskOfBias)
    source_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _text(self.study_id, "study_id", 256))
        for name in ("population", "intervention_or_exposure", "comparator", "outcome", "study_design"):
            allow_empty = name == "comparator"
            object.__setattr__(self, name, _text(getattr(self, name), name, 1000, allow_empty=allow_empty))
        if isinstance(self.sample_size, bool) or not isinstance(self.sample_size, int) or self.sample_size < 0 or self.sample_size > 10**9:
            raise ValueError("sample_size is invalid")
        object.__setattr__(self, "follow_up", _text(self.follow_up, "follow_up", 500, allow_empty=True))
        for name in ("methods", "limitations"):
            values = getattr(self, name)
            if len(values) > _MAX_ITEMS:
                raise ValueError(f"{name} exceeds the item limit")
            object.__setattr__(self, name, tuple(dict.fromkeys(_text(item, name, 1000) for item in values)))
        object.__setattr__(self, "funding", _text(self.funding, "funding", 1000, allow_empty=True))
        object.__setattr__(self, "conflicts", _text(self.conflicts, "conflicts", 1000, allow_empty=True))
        if not isinstance(self.risk_of_bias, RiskOfBias):
            raise ValueError("risk_of_bias must be RiskOfBias")
        digest = self.source_sha256.lower().strip() if isinstance(self.source_sha256, str) else ""
        if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError("source_sha256 is invalid")
        object.__setattr__(self, "source_sha256", digest)


@dataclass(frozen=True)
class EffectEstimate:
    study_id: str
    outcome: str
    effect_type: str
    estimate: float
    standard_error: float
    lower_ci: float | None = None
    upper_ci: float | None = None
    confidence_level: float = 0.95
    unit: str = ""
    direction: str = "as_reported"

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _text(self.study_id, "study_id", 256))
        object.__setattr__(self, "outcome", _text(self.outcome, "outcome", 1000))
        et = _text(self.effect_type, "effect_type", 64).lower()
        if et not in _EFFECT_TYPES:
            raise ValueError("unsupported effect_type")
        object.__setattr__(self, "effect_type", et)
        estimate = _finite(self.estimate, "estimate")
        if et in {"risk_ratio", "odds_ratio", "hazard_ratio"} and estimate <= 0:
            raise ValueError("ratio effect estimates must be positive")
        if et == "proportion" and not 0.0 <= estimate <= 1.0:
            raise ValueError("proportion must be between 0 and 1")
        if et == "correlation" and not -1.0 < estimate < 1.0:
            raise ValueError("correlation must lie strictly between -1 and 1")
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "standard_error", _finite(self.standard_error, "standard_error", positive=True))
        if (self.lower_ci is None) != (self.upper_ci is None):
            raise ValueError("both confidence interval bounds must be supplied together")
        if self.lower_ci is not None:
            lower = _finite(self.lower_ci, "lower_ci")
            upper = _finite(self.upper_ci, "upper_ci")
            if lower > upper or not lower <= estimate <= upper:
                raise ValueError("confidence interval is inconsistent with the estimate")
            object.__setattr__(self, "lower_ci", lower)
            object.__setattr__(self, "upper_ci", upper)
        confidence = _finite(self.confidence_level, "confidence_level")
        if not 0.5 < confidence < 1.0:
            raise ValueError("confidence_level must lie between 0.5 and 1.0")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "unit", _text(self.unit, "unit", 100, allow_empty=True))
        direction = _text(self.direction, "direction", 64).lower()
        if direction not in {"as_reported", "higher_is_better", "higher_is_worse"}:
            raise ValueError("direction is invalid")
        object.__setattr__(self, "direction", direction)

    @property
    def analysis_estimate(self) -> float:
        if self.effect_type in {"risk_ratio", "odds_ratio", "hazard_ratio"}:
            return math.log(self.estimate)
        if self.effect_type == "correlation":
            return math.atanh(self.estimate)
        return self.estimate

    def from_analysis_scale(self, value: float) -> float:
        parsed = _finite(value, "analysis value")
        if self.effect_type in {"risk_ratio", "odds_ratio", "hazard_ratio"}:
            return math.exp(parsed)
        if self.effect_type == "correlation":
            return math.tanh(parsed)
        return parsed


@dataclass(frozen=True)
class SynthesisCompatibility:
    compatible: bool
    reasons: tuple[str, ...]


def assess_compatibility(estimates: Sequence[EffectEstimate]) -> SynthesisCompatibility:
    if not estimates:
        return SynthesisCompatibility(False, ("no_estimates",))
    if len(estimates) > 10_000:
        raise ValueError("too many estimates")
    reasons: set[str] = set()
    effect_types = {item.effect_type for item in estimates}
    outcomes = {item.outcome.casefold() for item in estimates}
    units = {item.unit.casefold() for item in estimates if item.unit}
    directions = {item.direction for item in estimates}
    if len(effect_types) != 1:
        reasons.add("effect_type_mismatch")
    if len(outcomes) != 1:
        reasons.add("outcome_mismatch")
    if len(units) > 1 and effect_types & {"mean_difference"}:
        reasons.add("unit_mismatch")
    if len(directions - {"as_reported"}) > 1:
        reasons.add("direction_mismatch")
    if len({item.study_id for item in estimates}) != len(estimates):
        reasons.add("duplicate_study_id")
    return SynthesisCompatibility(not reasons, tuple(sorted(reasons)))


@dataclass(frozen=True)
class MetaAnalysisResult:
    model: str
    effect_type: str
    outcome: str
    studies: int
    pooled_estimate: float
    pooled_standard_error: float
    lower_ci: float
    upper_ci: float
    q: float
    i_squared: float
    tau_squared: float
    weights: Mapping[str, float]
    compatibility: SynthesisCompatibility
    fingerprint: str


def _normal_quantile(confidence: float) -> float:
    """Acklam approximation of the standard-normal quantile for two-sided CIs."""
    p = (1.0 + confidence) / 2.0
    # Peter J. Acklam's rational approximation coefficients.
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def meta_analyze(estimates: Sequence[EffectEstimate], *, model: str = "random", confidence_level: float = 0.95) -> MetaAnalysisResult:
    rows = tuple(estimates)
    compatibility = assess_compatibility(rows)
    if not compatibility.compatible:
        raise ValueError(f"incompatible estimates: {','.join(compatibility.reasons)}")
    if len(rows) < 1:
        raise ValueError("at least one estimate is required")
    selected_model = _text(model, "model", 16).lower()
    if selected_model not in {"fixed", "random"}:
        raise ValueError("model must be fixed or random")
    confidence = _finite(confidence_level, "confidence_level")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence_level is invalid")

    y = [item.analysis_estimate for item in rows]
    variances = [item.standard_error ** 2 for item in rows]
    fixed_w = [1.0 / variance for variance in variances]
    sum_w = sum(fixed_w)
    fixed_mu = sum(weight * value for weight, value in zip(fixed_w, y)) / sum_w
    q = sum(weight * ((value - fixed_mu) ** 2) for weight, value in zip(fixed_w, y))
    df = max(0, len(rows) - 1)
    c = sum_w - (sum(weight * weight for weight in fixed_w) / sum_w)
    tau2 = max(0.0, (q - df) / c) if selected_model == "random" and df > 0 and c > 0 else 0.0
    weights = [1.0 / (variance + tau2) for variance in variances]
    weight_sum = sum(weights)
    mu = sum(weight * value for weight, value in zip(weights, y)) / weight_sum
    se = math.sqrt(1.0 / weight_sum)
    z = _normal_quantile(confidence)
    lower_analysis = mu - z * se
    upper_analysis = mu + z * se
    prototype = rows[0]
    pooled = prototype.from_analysis_scale(mu)
    lower = prototype.from_analysis_scale(lower_analysis)
    upper = prototype.from_analysis_scale(upper_analysis)
    i2 = max(0.0, ((q - df) / q) * 100.0) if q > 0 and df > 0 else 0.0
    normalized_weights = {row.study_id: weight / weight_sum for row, weight in zip(rows, weights)}
    payload = {
        "model": selected_model,
        "effect_type": prototype.effect_type,
        "outcome": prototype.outcome,
        "studies": len(rows),
        "pooled_estimate": pooled,
        "pooled_standard_error": se,
        "lower_ci": lower,
        "upper_ci": upper,
        "q": q,
        "i_squared": i2,
        "tau_squared": tau2,
        "weights": normalized_weights,
    }
    return MetaAnalysisResult(selected_model, prototype.effect_type, prototype.outcome, len(rows), pooled, se, lower, upper, q, i2, tau2, normalized_weights, compatibility, hashlib.sha256(_canonical(payload)).hexdigest())


def leave_one_out(estimates: Sequence[EffectEstimate], *, model: str = "random") -> Mapping[str, MetaAnalysisResult]:
    rows = tuple(estimates)
    if len(rows) < 2:
        raise ValueError("leave-one-out synthesis requires at least two estimates")
    if len(rows) > 1_000:
        raise ValueError("leave-one-out input exceeds the bounded size")
    return {row.study_id: meta_analyze(tuple(item for item in rows if item.study_id != row.study_id), model=model) for row in rows}


def evidence_quality_summary(studies: Iterable[StudyEvidence]) -> Mapping[str, Any]:
    rows = tuple(studies)
    if len(rows) > 10_000:
        raise ValueError("too many studies")
    counts = {level: 0 for level in sorted(_RISK_LEVELS)}
    sample_total = 0
    for study in rows:
        counts[study.risk_of_bias.overall] += 1
        sample_total += study.sample_size
    high = counts.get("high", 0)
    assessed = len(rows) - counts.get("not_assessed", 0)
    return {
        "study_count": len(rows),
        "sample_size_total": sample_total,
        "risk_of_bias_counts": counts,
        "assessed_fraction": (assessed / len(rows)) if rows else 0.0,
        "high_risk_fraction": (high / len(rows)) if rows else 0.0,
        "note": "Structured risk-of-bias metadata is not proof of study validity or GRADE certainty.",
    }


__all__ = [
    "EffectEstimate",
    "MetaAnalysisResult",
    "ResearchQuestion",
    "RiskOfBias",
    "StudyEvidence",
    "SynthesisCompatibility",
    "assess_compatibility",
    "evidence_quality_summary",
    "leave_one_out",
    "meta_analyze",
]
