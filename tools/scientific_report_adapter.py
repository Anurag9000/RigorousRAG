"""Render review-gated scientific evidence bundles into the authoritative report model."""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping, Sequence

from tools.research_report import EvidenceMatrixRow, ReportSection, ResearchReport
from tools.scientific_evidence_pipeline import (
    ScientificEvidenceBundle,
    effect_fingerprint,
    study_fingerprint,
)
from tools.scientific_synthesis import EffectEstimate, StudyEvidence


def _citation_ids(values: Sequence[str]) -> tuple[str, ...]:
    cleaned = tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if len(cleaned) > 100:
        raise ValueError("study citation IDs exceed the report row limit")
    return cleaned


def _effect_text(effect: EffectEstimate) -> str:
    result = f"{effect.effect_type}={effect.estimate:g}"
    if effect.unit:
        result += f" {effect.unit}"
    if effect.lower_ci is not None and effect.upper_ci is not None:
        result += f"; CI {effect.lower_ci:g} to {effect.upper_ci:g}"
    return result


def _uncertainty_text(effect: EffectEstimate) -> str:
    values = [f"SE={effect.standard_error:g}"]
    if effect.lower_ci is not None and effect.upper_ci is not None:
        values.append(f"confidence={effect.confidence_level:.3f}")
    return "; ".join(values)


def build_scientific_evidence_matrix(
    bundle: ScientificEvidenceBundle,
    *,
    studies: Sequence[StudyEvidence],
    effects: Sequence[EffectEstimate],
    study_citation_ids: Mapping[str, Sequence[str]],
) -> tuple[EvidenceMatrixRow, ...]:
    if not isinstance(bundle, ScientificEvidenceBundle):
        raise TypeError("bundle must be ScientificEvidenceBundle")
    accepted_studies = set(bundle.accepted_study_fingerprints)
    accepted_effects = set(bundle.accepted_effect_fingerprints)
    study_by_id = {item.study_id: item for item in studies}
    if len(study_by_id) != len(studies):
        raise ValueError("study IDs must be unique")
    effects_by_study: dict[str, list[EffectEstimate]] = {}
    effect_by_fingerprint: dict[str, EffectEstimate] = {}
    for effect in effects:
        fingerprint = effect_fingerprint(effect)
        effect_by_fingerprint[fingerprint] = effect
        if fingerprint in accepted_effects:
            effects_by_study.setdefault(effect.study_id, []).append(effect)

    rows: list[EvidenceMatrixRow] = []
    for study in studies:
        if study_fingerprint(study) not in accepted_studies:
            continue
        study_effects = effects_by_study.get(study.study_id, [])
        result = "; ".join(_effect_text(item) for item in study_effects)
        uncertainty = "; ".join(_uncertainty_text(item) for item in study_effects)
        limitations = list(study.limitations)
        limitations.append(f"risk_of_bias={study.risk_of_bias.overall}")
        rows.append(
            EvidenceMatrixRow(
                claim_id=f"study:{study.study_id}",
                claim_text=f"Reviewed structured evidence record for {study.outcome}.",
                support_status="supported",
                study_id=study.study_id,
                population=study.population,
                intervention_or_exposure=study.intervention_or_exposure,
                comparator=study.comparator,
                outcome=study.outcome,
                result=result,
                uncertainty=uncertainty,
                limitation="; ".join(item for item in limitations if item),
                citation_ids=_citation_ids(study_citation_ids.get(study.study_id, ())),
            )
        )

    for group in bundle.synthesis_groups:
        pooled = group.random_effect or group.fixed_effect
        if pooled is None:
            continue
        included_effects = [
            effect_by_fingerprint[fingerprint]
            for fingerprint in group.included_effect_fingerprints
            if fingerprint in effect_by_fingerprint
        ]
        citations: list[str] = []
        for effect in included_effects:
            citations.extend(study_citation_ids.get(effect.study_id, ()))
        model = group.random_effect.model if group.random_effect is not None else group.fixed_effect.model
        result = f"{model} pooled {pooled.effect_type}={pooled.pooled_estimate:g}"
        if group.unit:
            result += f" {group.unit}"
        uncertainty = (
            f"95% CI {pooled.lower_ci:g} to {pooled.upper_ci:g}; "
            f"I²={pooled.i_squared:.2f}%; tau²={pooled.tau_squared:.6g}"
        )
        rows.append(
            EvidenceMatrixRow(
                claim_id=f"pool:{group.group_id}",
                claim_text=(
                    f"Review-gated quantitative synthesis for {group.outcome} "
                    f"across {pooled.studies} compatible studies."
                ),
                support_status="supported",
                study_id="pooled",
                outcome=group.outcome,
                result=result,
                uncertainty=uncertainty,
                limitation=(
                    "Quantitative pooling reflects reviewed structured estimates and arithmetic; "
                    "it does not by itself establish certainty, applicability, or causality."
                ),
                citation_ids=_citation_ids(citations),
            )
        )
    return tuple(rows)


def scientific_bundle_section(bundle: ScientificEvidenceBundle) -> ReportSection:
    certainty = ", ".join(
        f"{item.outcome}={item.final_level}"
        for item in bundle.certainty
    ) or "not fully assessed"
    body = (
        f"Scientific evidence bundle {bundle.fingerprint}. "
        f"Accepted studies: {len(bundle.accepted_study_fingerprints)}; "
        f"accepted effects: {len(bundle.accepted_effect_fingerprints)}; "
        f"quantitative synthesis groups: {len(bundle.synthesis_groups)}. "
        f"Outcome certainty: {certainty}. "
        f"Unresolved governance or analytical items: {len(bundle.unresolved)}."
    )
    return ReportSection(heading="Governed scientific synthesis", body=body)


def augment_report_with_scientific_bundle(
    report: ResearchReport,
    bundle: ScientificEvidenceBundle,
    *,
    studies: Sequence[StudyEvidence],
    effects: Sequence[EffectEstimate],
    study_citation_ids: Mapping[str, Sequence[str]],
) -> ResearchReport:
    if not isinstance(report, ResearchReport):
        raise TypeError("report must be ResearchReport")
    rows = build_scientific_evidence_matrix(
        bundle,
        studies=studies,
        effects=effects,
        study_citation_ids=study_citation_ids,
    )
    warnings = list(report.warnings)
    warnings.append(f"scientific_evidence_bundle_sha256={bundle.fingerprint}")
    if bundle.unresolved:
        warnings.append(
            "Scientific bundle has unresolved items: " + "; ".join(bundle.unresolved[:50])
        )
    return ResearchReport(
        title=report.title,
        question=report.question,
        search_strategy=report.search_strategy,
        sections=(*report.sections, scientific_bundle_section(bundle)),
        evidence_matrix=(*report.evidence_matrix, *rows),
        citations=report.citations,
        conflicts=report.conflicts,
        limitations=report.limitations,
        warnings=tuple(warnings),
    )


__all__ = [
    "augment_report_with_scientific_bundle",
    "build_scientific_evidence_matrix",
    "scientific_bundle_section",
]
