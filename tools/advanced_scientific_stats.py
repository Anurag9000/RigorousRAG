"""Advanced deterministic quantitative synthesis for governed scientific evidence.

Implements weighted meta-regression and a fixed-effect consistency network meta-analysis
using small pure-Python linear algebra. Inputs must already be governed effect estimates;
this module does not infer moderators, transitivity, exchangeability or study validity.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.scientific_synthesis import EffectEstimate, assess_compatibility, meta_analyze

_MAX_MODERATORS = 32
_MAX_TREATMENTS = 128
_MAX_ROWS = 10_000
_Z95 = 1.959963984540054


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, ...]:
    n = len(vector)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("linear system dimensions are inconsistent")
    augmented = [
        [float(matrix[i][j]) for j in range(n)] + [float(vector[i])]
        for i in range(n)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) <= 1e-14:
            raise ValueError("design matrix is singular or numerically rank deficient")
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[col])
            ]
    return tuple(augmented[row][-1] for row in range(n))


def _inverse(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    n = len(matrix)
    columns = []
    for index in range(n):
        basis = [0.0] * n
        basis[index] = 1.0
        columns.append(_solve(matrix, basis))
    return tuple(tuple(columns[col][row] for col in range(n)) for row in range(n))


def _xtwx(
    design: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    p = len(design[0])
    return tuple(
        tuple(
            sum(weights[i] * design[i][a] * design[i][b] for i in range(len(design)))
            for b in range(p)
        )
        for a in range(p)
    )


def _xtwy(
    design: Sequence[Sequence[float]],
    weights: Sequence[float],
    y: Sequence[float],
) -> tuple[float, ...]:
    p = len(design[0])
    return tuple(
        sum(weights[i] * design[i][a] * y[i] for i in range(len(design)))
        for a in range(p)
    )


@dataclass(frozen=True)
class MetaRegressionCoefficient:
    term: str
    estimate_analysis_scale: float
    standard_error: float
    lower_ci: float
    upper_ci: float


@dataclass(frozen=True)
class MetaRegressionResult:
    outcome: str
    effect_type: str
    unit: str
    moderators: tuple[str, ...]
    studies: int
    tau_squared: float
    coefficients: tuple[MetaRegressionCoefficient, ...]
    residual_q: float
    residual_df: int
    fingerprint: str
    note: str = (
        "Meta-regression associations are observational across studies and do not establish "
        "individual-level effect modification or causality."
    )


def meta_regress(
    estimates: Sequence[EffectEstimate],
    moderators: Mapping[str, Mapping[str, float]],
    *,
    moderator_names: Sequence[str] | None = None,
    random_effects: bool = True,
) -> MetaRegressionResult:
    rows = tuple(estimates)
    if not rows or len(rows) > _MAX_ROWS:
        raise ValueError("meta-regression requires bounded non-empty estimates")
    compatibility = assess_compatibility(rows)
    if not compatibility.compatible:
        raise ValueError("meta-regression estimates are incompatible: " + ",".join(compatibility.reasons))
    if moderator_names is None:
        names = tuple(sorted({key for row in moderators.values() for key in row}))
    else:
        names = tuple(dict.fromkeys(_text(item, "moderator", 128) for item in moderator_names))
    if not names or len(names) > _MAX_MODERATORS:
        raise ValueError("meta-regression requires between 1 and 32 moderators")
    if len(rows) <= len(names) + 1:
        raise ValueError("meta-regression requires more studies than fitted coefficients")

    design: list[list[float]] = []
    y: list[float] = []
    variances: list[float] = []
    for effect in rows:
        covariates = moderators.get(effect.study_id)
        if covariates is None:
            raise ValueError(f"moderators missing for study {effect.study_id}")
        values = [1.0]
        for name in names:
            if name not in covariates:
                raise ValueError(f"moderator {name} missing for study {effect.study_id}")
            values.append(_finite(covariates[name], f"moderator {name}"))
        design.append(values)
        y.append(effect.analysis_estimate)
        variances.append(effect.standard_error ** 2)

    tau2 = meta_analyze(rows, model="random").tau_squared if random_effects else 0.0
    weights = [1.0 / (variance + tau2) for variance in variances]
    information = _xtwx(design, weights)
    beta = _solve(information, _xtwy(design, weights, y))
    covariance = _inverse(information)
    coefficients: list[MetaRegressionCoefficient] = []
    terms = ("intercept", *names)
    for index, term in enumerate(terms):
        se = math.sqrt(max(0.0, covariance[index][index]))
        estimate = beta[index]
        coefficients.append(
            MetaRegressionCoefficient(
                term=term,
                estimate_analysis_scale=estimate,
                standard_error=se,
                lower_ci=estimate - _Z95 * se,
                upper_ci=estimate + _Z95 * se,
            )
        )
    fitted = [sum(beta[j] * design[i][j] for j in range(len(beta))) for i in range(len(rows))]
    residual_q = sum(weights[i] * (y[i] - fitted[i]) ** 2 for i in range(len(rows)))
    residual_df = len(rows) - len(beta)
    payload = {
        "outcome": rows[0].outcome,
        "effect_type": rows[0].effect_type,
        "unit": rows[0].unit,
        "moderators": names,
        "studies": len(rows),
        "tau_squared": tau2,
        "coefficients": [asdict(item) for item in coefficients],
        "residual_q": residual_q,
        "residual_df": residual_df,
    }
    return MetaRegressionResult(
        outcome=rows[0].outcome,
        effect_type=rows[0].effect_type,
        unit=rows[0].unit,
        moderators=names,
        studies=len(rows),
        tau_squared=tau2,
        coefficients=tuple(coefficients),
        residual_q=residual_q,
        residual_df=residual_df,
        fingerprint=hashlib.sha256(_canonical(payload)).hexdigest(),
    )


@dataclass(frozen=True)
class NetworkContrast:
    study_id: str
    treatment_a: str
    treatment_b: str
    effect_type: str
    estimate: float
    standard_error: float
    outcome: str
    unit: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _text(self.study_id, "study_id", 256))
        a = _text(self.treatment_a, "treatment_a", 500)
        b = _text(self.treatment_b, "treatment_b", 500)
        if a == b:
            raise ValueError("network contrast treatments must differ")
        object.__setattr__(self, "treatment_a", a)
        object.__setattr__(self, "treatment_b", b)
        effect_type = _text(self.effect_type, "effect_type", 64).lower()
        if effect_type not in {
            "risk_ratio",
            "odds_ratio",
            "hazard_ratio",
            "mean_difference",
            "standardized_mean_difference",
            "correlation",
        }:
            raise ValueError("unsupported network effect_type")
        object.__setattr__(self, "effect_type", effect_type)
        estimate = _finite(self.estimate, "estimate")
        if effect_type in {"risk_ratio", "odds_ratio", "hazard_ratio"} and estimate <= 0:
            raise ValueError("ratio network effects must be positive")
        if effect_type == "correlation" and not -1.0 < estimate < 1.0:
            raise ValueError("network correlation must lie strictly between -1 and 1")
        object.__setattr__(self, "estimate", estimate)
        se = _finite(self.standard_error, "standard_error")
        if se <= 0:
            raise ValueError("standard_error must be positive")
        object.__setattr__(self, "standard_error", se)
        object.__setattr__(self, "outcome", _text(self.outcome, "outcome", 1000))
        unit = "" if self.unit is None else str(self.unit).strip()
        if len(unit) > 100:
            raise ValueError("unit is invalid")
        object.__setattr__(self, "unit", unit)

    @property
    def analysis_estimate(self) -> float:
        if self.effect_type in {"risk_ratio", "odds_ratio", "hazard_ratio"}:
            return math.log(self.estimate)
        if self.effect_type == "correlation":
            return math.atanh(self.estimate)
        return self.estimate

    def from_analysis_scale(self, value: float) -> float:
        if self.effect_type in {"risk_ratio", "odds_ratio", "hazard_ratio"}:
            return math.exp(value)
        if self.effect_type == "correlation":
            return math.tanh(value)
        return value


@dataclass(frozen=True)
class NetworkTreatmentEffect:
    treatment: str
    versus_reference_estimate: float
    standard_error_analysis_scale: float
    lower_ci: float
    upper_ci: float


@dataclass(frozen=True)
class NetworkMetaAnalysisResult:
    outcome: str
    effect_type: str
    unit: str
    reference_treatment: str
    treatments: tuple[str, ...]
    contrasts: int
    treatment_effects: tuple[NetworkTreatmentEffect, ...]
    pairwise_estimates: Mapping[str, float]
    residual_q: float
    residual_df: int
    connected: bool
    fingerprint: str
    note: str = (
        "Consistency-model network estimates require substantive transitivity and consistency "
        "assumptions that are not proven by network connectivity or model fit."
    )


def _network_connected(treatments: Sequence[str], rows: Sequence[NetworkContrast]) -> bool:
    adjacency = {item: set() for item in treatments}
    for row in rows:
        adjacency[row.treatment_a].add(row.treatment_b)
        adjacency[row.treatment_b].add(row.treatment_a)
    seen: set[str] = set()
    pending = [treatments[0]] if treatments else []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(adjacency[current] - seen)
    return len(seen) == len(treatments)


def network_meta_analyze(
    contrasts: Sequence[NetworkContrast],
    *,
    reference_treatment: str | None = None,
) -> NetworkMetaAnalysisResult:
    rows = tuple(contrasts)
    if not rows or len(rows) > _MAX_ROWS:
        raise ValueError("network meta-analysis requires bounded non-empty contrasts")
    effect_types = {item.effect_type for item in rows}
    outcomes = {item.outcome.casefold() for item in rows}
    units = {item.unit.casefold() for item in rows if item.unit}
    if len(effect_types) != 1 or len(outcomes) != 1 or len(units) > 1:
        raise ValueError("network contrasts must share outcome/effect type and compatible units")
    treatments = tuple(sorted({item.treatment_a for item in rows} | {item.treatment_b for item in rows}))
    if len(treatments) < 2 or len(treatments) > _MAX_TREATMENTS:
        raise ValueError("network treatment count is invalid")
    if not _network_connected(treatments, rows):
        raise ValueError("network of treatments is disconnected")
    reference = _text(reference_treatment or treatments[0], "reference_treatment", 500)
    if reference not in treatments:
        raise ValueError("reference treatment is absent from network")
    non_reference = tuple(item for item in treatments if item != reference)
    columns = {treatment: index for index, treatment in enumerate(non_reference)}
    design: list[list[float]] = []
    y: list[float] = []
    weights: list[float] = []
    for contrast in rows:
        vector = [0.0] * len(non_reference)
        # Observation is theta_b - theta_a.
        if contrast.treatment_b != reference:
            vector[columns[contrast.treatment_b]] += 1.0
        if contrast.treatment_a != reference:
            vector[columns[contrast.treatment_a]] -= 1.0
        design.append(vector)
        y.append(contrast.analysis_estimate)
        weights.append(1.0 / (contrast.standard_error ** 2))
    information = _xtwx(design, weights)
    beta = _solve(information, _xtwy(design, weights, y))
    covariance = _inverse(information)
    prototype = rows[0]
    effects: list[NetworkTreatmentEffect] = [
        NetworkTreatmentEffect(reference, prototype.from_analysis_scale(0.0), 0.0, prototype.from_analysis_scale(0.0), prototype.from_analysis_scale(0.0))
    ]
    theta = {reference: 0.0}
    for treatment in non_reference:
        index = columns[treatment]
        estimate = beta[index]
        se = math.sqrt(max(0.0, covariance[index][index]))
        theta[treatment] = estimate
        effects.append(
            NetworkTreatmentEffect(
                treatment=treatment,
                versus_reference_estimate=prototype.from_analysis_scale(estimate),
                standard_error_analysis_scale=se,
                lower_ci=prototype.from_analysis_scale(estimate - _Z95 * se),
                upper_ci=prototype.from_analysis_scale(estimate + _Z95 * se),
            )
        )
    pairwise: dict[str, float] = {}
    for a in treatments:
        for b in treatments:
            if a >= b:
                continue
            pairwise[f"{b} vs {a}"] = prototype.from_analysis_scale(theta[b] - theta[a])
    fitted = [sum(beta[j] * design[i][j] for j in range(len(beta))) for i in range(len(rows))]
    residual_q = sum(weights[i] * (y[i] - fitted[i]) ** 2 for i in range(len(rows)))
    residual_df = len(rows) - len(non_reference)
    payload = {
        "outcome": prototype.outcome,
        "effect_type": prototype.effect_type,
        "unit": prototype.unit,
        "reference_treatment": reference,
        "treatments": treatments,
        "contrasts": len(rows),
        "treatment_effects": [asdict(item) for item in effects],
        "pairwise_estimates": pairwise,
        "residual_q": residual_q,
        "residual_df": residual_df,
    }
    return NetworkMetaAnalysisResult(
        outcome=prototype.outcome,
        effect_type=prototype.effect_type,
        unit=prototype.unit,
        reference_treatment=reference,
        treatments=treatments,
        contrasts=len(rows),
        treatment_effects=tuple(sorted(effects, key=lambda item: item.treatment)),
        pairwise_estimates=dict(sorted(pairwise.items())),
        residual_q=residual_q,
        residual_df=residual_df,
        connected=True,
        fingerprint=hashlib.sha256(_canonical(payload)).hexdigest(),
    )


__all__ = [
    "MetaRegressionCoefficient",
    "MetaRegressionResult",
    "NetworkContrast",
    "NetworkMetaAnalysisResult",
    "NetworkTreatmentEffect",
    "meta_regress",
    "network_meta_analyze",
]
