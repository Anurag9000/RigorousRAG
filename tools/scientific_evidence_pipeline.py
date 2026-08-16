"""Review-gated scientific evidence orchestration.

This module connects the repository's existing study/effect, meta-analysis, certainty,
causal and numerical-consistency primitives without turning extraction output into truth.
Only explicitly accepted, fingerprint-matching study/effect records enter quantitative
synthesis. Risk-of-bias and certainty review remain separate, visible gates.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.causal_evidence import CausalClaim, CausalDAG, causal_claim_readiness
from tools.evidence_certainty import (
    CertaintyDomainJudgment,
    EvidenceCertaintyAssessment,
    assess_certainty,
)
from tools.numerical_consistency import (
    NumericalCheck,
    QuantitativeAssertion,
    QuantitativeEvidence,
    check_assertions,
)
from tools.scientific_synthesis import (
    EffectEstimate,
    MetaAnalysisResult,
    ResearchQuestion,
    StudyEvidence,
    assess_compatibility,
    evidence_quality_summary,
    leave_one_out,
    meta_analyze,
)

_REVIEW_KINDS = frozenset({"study", "effect", "risk_of_bias"})
_REVIEW_STATES = frozenset({"accepted", "rejected", "needs_revision"})


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(asdict(value))).hexdigest()


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if allow_empty and value in {None, ""}:
        return ""
    digest = _text(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


@dataclass(frozen=True)
class EvidenceReviewStamp:
    """Human review decision bound to the exact structured record fingerprint."""

    subject_kind: str
    subject_id: str
    subject_fingerprint: str
    status: str
    reviewer_id: str
    evidence_ids: tuple[str, ...]
    rationale_sha256: str
    reviewed_at: float
    attestation_id: str = ""
    attestation_verified: bool = False

    def __post_init__(self) -> None:
        kind = _text(self.subject_kind, "subject_kind", 64).lower()
        if kind not in _REVIEW_KINDS:
            raise ValueError("unsupported review subject kind")
        object.__setattr__(self, "subject_kind", kind)
        object.__setattr__(self, "subject_id", _text(self.subject_id, "subject_id", 500))
        object.__setattr__(
            self,
            "subject_fingerprint",
            _sha(self.subject_fingerprint, "subject_fingerprint"),
        )
        status = _text(self.status, "status", 32).lower()
        if status not in _REVIEW_STATES:
            raise ValueError("unsupported review status")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reviewer_id", _text(self.reviewer_id, "reviewer_id", 256))
        if len(self.evidence_ids) > 1000:
            raise ValueError("evidence_ids exceed the item limit")
        object.__setattr__(
            self,
            "evidence_ids",
            tuple(dict.fromkeys(_text(item, "evidence_id", 500) for item in self.evidence_ids)),
        )
        object.__setattr__(
            self,
            "rationale_sha256",
            _sha(self.rationale_sha256, "rationale_sha256"),
        )
        reviewed_at = float(self.reviewed_at)
        if reviewed_at <= 0 or not reviewed_at < float("inf"):
            raise ValueError("reviewed_at is invalid")
        object.__setattr__(self, "reviewed_at", reviewed_at)
        object.__setattr__(
            self,
            "attestation_id",
            _text(self.attestation_id, "attestation_id", 500, allow_empty=True),
        )
        if not isinstance(self.attestation_verified, bool):
            raise ValueError("attestation_verified must be boolean")
        if self.attestation_verified and not self.attestation_id:
            raise ValueError("verified review attestation requires attestation_id")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class SynthesisGroup:
    group_id: str
    outcome: str
    effect_type: str
    unit: str
    direction: str
    included_effect_fingerprints: tuple[str, ...]
    excluded_effect_fingerprints: tuple[str, ...]
    fixed_effect: MetaAnalysisResult | None
    random_effect: MetaAnalysisResult | None
    leave_one_out_random: Mapping[str, MetaAnalysisResult]
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class ScientificEvidenceBundle:
    question: ResearchQuestion
    accepted_study_fingerprints: tuple[str, ...]
    rejected_or_unreviewed_study_fingerprints: tuple[str, ...]
    accepted_effect_fingerprints: tuple[str, ...]
    rejected_or_unreviewed_effect_fingerprints: tuple[str, ...]
    risk_of_bias_reviewed_study_fingerprints: tuple[str, ...]
    quality_summary: Mapping[str, Any]
    synthesis_groups: tuple[SynthesisGroup, ...]
    certainty: tuple[EvidenceCertaintyAssessment, ...]
    causal_readiness: tuple[Mapping[str, Any], ...]
    numerical_checks: tuple[NumericalCheck, ...]
    unresolved: tuple[str, ...]
    review_stamp_fingerprints: tuple[str, ...]
    created_at: float
    fingerprint: str
    note: str = (
        "Bundle readiness reflects explicit record/review gates and deterministic arithmetic; "
        "it does not replace domain-expert appraisal or prove causal validity."
    )


def study_fingerprint(study: StudyEvidence) -> str:
    if not isinstance(study, StudyEvidence):
        raise TypeError("study must be StudyEvidence")
    return _fingerprint(study)


def effect_fingerprint(effect: EffectEstimate) -> str:
    if not isinstance(effect, EffectEstimate):
        raise TypeError("effect must be EffectEstimate")
    return _fingerprint(effect)


def risk_of_bias_fingerprint(study: StudyEvidence) -> str:
    if not isinstance(study, StudyEvidence):
        raise TypeError("study must be StudyEvidence")
    return hashlib.sha256(_canonical(asdict(study.risk_of_bias))).hexdigest()


def _review_index(
    stamps: Sequence[EvidenceReviewStamp],
) -> Mapping[tuple[str, str, str], EvidenceReviewStamp]:
    if len(stamps) > 100_000:
        raise ValueError("review stamps exceed the item limit")
    output: dict[tuple[str, str, str], EvidenceReviewStamp] = {}
    for stamp in stamps:
        if not isinstance(stamp, EvidenceReviewStamp):
            raise TypeError("review_stamps must contain EvidenceReviewStamp values")
        key = (stamp.subject_kind, stamp.subject_id, stamp.subject_fingerprint)
        previous = output.get(key)
        if previous is None or (stamp.reviewed_at, stamp.fingerprint) > (
            previous.reviewed_at,
            previous.fingerprint,
        ):
            output[key] = stamp
    return output


def _accepted(
    index: Mapping[tuple[str, str, str], EvidenceReviewStamp],
    *,
    kind: str,
    subject_id: str,
    fingerprint: str,
    require_verified_attestation: bool,
) -> bool:
    stamp = index.get((kind, subject_id, fingerprint))
    if stamp is None or stamp.status != "accepted":
        return False
    return (not require_verified_attestation) or stamp.attestation_verified


def _design_family(studies: Sequence[StudyEvidence]) -> str:
    if not studies:
        return "unknown"
    labels = {study.study_design.casefold().replace("-", "_").replace(" ", "_") for study in studies}
    randomized_tokens = {"rct", "randomized", "randomized_controlled_trial", "randomised_controlled_trial"}
    if labels and labels.issubset(randomized_tokens):
        return "randomized"
    return "observational"


def _group_effects(effects: Sequence[EffectEstimate]) -> Mapping[tuple[str, str, str, str], tuple[EffectEstimate, ...]]:
    groups: dict[tuple[str, str, str, str], list[EffectEstimate]] = {}
    for effect in effects:
        key = (
            effect.outcome.casefold(),
            effect.effect_type,
            effect.unit.casefold(),
            effect.direction,
        )
        groups.setdefault(key, []).append(effect)
    return {key: tuple(rows) for key, rows in sorted(groups.items())}


def build_scientific_evidence_bundle(
    *,
    question: ResearchQuestion,
    studies: Sequence[StudyEvidence],
    effects: Sequence[EffectEstimate],
    review_stamps: Sequence[EvidenceReviewStamp],
    certainty_judgments: Mapping[str, Sequence[CertaintyDomainJudgment]] | None = None,
    causal_dag: CausalDAG | None = None,
    causal_claims: Sequence[CausalClaim] = (),
    quantitative_assertions: Sequence[QuantitativeAssertion] = (),
    quantitative_evidence: Mapping[str, QuantitativeEvidence] | None = None,
    require_verified_review_attestations: bool = False,
    minimum_pooling_studies: int = 2,
) -> ScientificEvidenceBundle:
    if not isinstance(question, ResearchQuestion):
        raise TypeError("question must be ResearchQuestion")
    if isinstance(minimum_pooling_studies, bool) or not isinstance(minimum_pooling_studies, int) or not 2 <= minimum_pooling_studies <= 100:
        raise ValueError("minimum_pooling_studies must be between 2 and 100")
    if len(studies) > 10_000 or len(effects) > 100_000:
        raise ValueError("scientific evidence bundle exceeds bounded input size")
    if any(not isinstance(item, StudyEvidence) for item in studies):
        raise TypeError("studies must contain StudyEvidence values")
    if any(not isinstance(item, EffectEstimate) for item in effects):
        raise TypeError("effects must contain EffectEstimate values")

    review_index = _review_index(review_stamps)
    study_by_id: dict[str, StudyEvidence] = {}
    for study in studies:
        if study.study_id in study_by_id:
            raise ValueError(f"duplicate study_id: {study.study_id}")
        study_by_id[study.study_id] = study

    accepted_studies: list[StudyEvidence] = []
    rejected_studies: list[str] = []
    reviewed_rob_studies: list[StudyEvidence] = []
    unresolved: list[str] = []
    for study in studies:
        sf = study_fingerprint(study)
        if _accepted(
            review_index,
            kind="study",
            subject_id=study.study_id,
            fingerprint=sf,
            require_verified_attestation=require_verified_review_attestations,
        ):
            accepted_studies.append(study)
        else:
            rejected_studies.append(sf)
            unresolved.append(f"study_not_accepted:{study.study_id}")
        rob_fp = risk_of_bias_fingerprint(study)
        if _accepted(
            review_index,
            kind="risk_of_bias",
            subject_id=study.study_id,
            fingerprint=rob_fp,
            require_verified_attestation=require_verified_review_attestations,
        ):
            reviewed_rob_studies.append(study)
        else:
            unresolved.append(f"risk_of_bias_not_accepted:{study.study_id}")

    accepted_study_ids = {item.study_id for item in accepted_studies}
    accepted_effects: list[EffectEstimate] = []
    rejected_effect_fps: list[str] = []
    for effect in effects:
        ef = effect_fingerprint(effect)
        if effect.study_id not in study_by_id:
            rejected_effect_fps.append(ef)
            unresolved.append(f"effect_study_missing:{effect.study_id}:{ef[:12]}")
            continue
        if effect.study_id not in accepted_study_ids:
            rejected_effect_fps.append(ef)
            unresolved.append(f"effect_study_not_accepted:{effect.study_id}:{ef[:12]}")
            continue
        if not _accepted(
            review_index,
            kind="effect",
            subject_id=ef,
            fingerprint=ef,
            require_verified_attestation=require_verified_review_attestations,
        ):
            rejected_effect_fps.append(ef)
            unresolved.append(f"effect_not_accepted:{effect.study_id}:{ef[:12]}")
            continue
        accepted_effects.append(effect)

    synthesis_groups: list[SynthesisGroup] = []
    for key, group in _group_effects(accepted_effects).items():
        outcome_key, effect_type, unit_key, direction = key
        included = tuple(effect_fingerprint(item) for item in group)
        blockers: list[str] = []
        compatibility = assess_compatibility(group)
        if not compatibility.compatible:
            blockers.extend(compatibility.reasons)
        if len(group) < minimum_pooling_studies:
            blockers.append("insufficient_studies_for_pooling")
        fixed: MetaAnalysisResult | None = None
        random: MetaAnalysisResult | None = None
        loo: Mapping[str, MetaAnalysisResult] = {}
        if not blockers:
            fixed = meta_analyze(group, model="fixed")
            random = meta_analyze(group, model="random")
            if len(group) >= 3:
                loo = leave_one_out(group, model="random")
        group_id = hashlib.sha256(
            _canonical(
                {
                    "outcome": outcome_key,
                    "effect_type": effect_type,
                    "unit": unit_key,
                    "direction": direction,
                    "effects": included,
                }
            )
        ).hexdigest()
        synthesis_groups.append(
            SynthesisGroup(
                group_id=group_id,
                outcome=group[0].outcome,
                effect_type=effect_type,
                unit=group[0].unit,
                direction=direction,
                included_effect_fingerprints=included,
                excluded_effect_fingerprints=(),
                fixed_effect=fixed,
                random_effect=random,
                leave_one_out_random=loo,
                blockers=tuple(sorted(set(blockers))),
            )
        )
        for blocker in blockers:
            unresolved.append(f"synthesis:{group[0].outcome}:{effect_type}:{blocker}")

    certainty_rows: list[EvidenceCertaintyAssessment] = []
    judgments_by_outcome = certainty_judgments or {}
    for outcome in question.outcomes:
        matching_studies = [
            item for item in accepted_studies if item.outcome.casefold() == outcome.casefold()
        ]
        judgments = tuple(judgments_by_outcome.get(outcome, ()))
        if not judgments:
            # Also accept a case-insensitive mapping key without mutating caller data.
            for key, value in judgments_by_outcome.items():
                if str(key).casefold() == outcome.casefold():
                    judgments = tuple(value)
                    break
        if not judgments:
            unresolved.append(f"certainty_judgments_missing:{outcome}")
            continue
        certainty = assess_certainty(
            question_fingerprint=question.fingerprint,
            outcome=outcome,
            study_design_family=_design_family(matching_studies),
            judgments=judgments,
        )
        certainty_rows.append(certainty)
        for domain in certainty.unresolved_domains:
            unresolved.append(f"certainty_unreviewed:{outcome}:{domain}")

    causal_rows: list[Mapping[str, Any]] = []
    if causal_claims:
        if causal_dag is None:
            unresolved.append("causal_dag_missing")
            for claim in causal_claims:
                causal_rows.append(
                    {
                        "claim_id": claim.claim_id,
                        "ready_for_causal_language": False,
                        "reason": "causal_dag_missing",
                    }
                )
        else:
            for claim in causal_claims:
                row = dict(causal_claim_readiness(claim, causal_dag))
                causal_rows.append(row)
                if not row.get("ready_for_causal_language", False) and not claim.association_only:
                    unresolved.append(f"causal_claim_not_ready:{claim.claim_id}")

    numerical_rows = check_assertions(
        quantitative_assertions,
        quantitative_evidence or {},
    ) if quantitative_assertions else ()
    for row in numerical_rows:
        if row.status != "consistent":
            unresolved.append(f"numerical_check:{row.assertion_id}:{row.status}")

    quality = evidence_quality_summary(reviewed_rob_studies)
    if not reviewed_rob_studies and studies:
        unresolved.append("no_reviewed_risk_of_bias_records")

    accepted_study_fps = tuple(sorted(study_fingerprint(item) for item in accepted_studies))
    accepted_effect_fps = tuple(sorted(effect_fingerprint(item) for item in accepted_effects))
    reviewed_rob_fps = tuple(sorted(study_fingerprint(item) for item in reviewed_rob_studies))
    stamp_fps = tuple(sorted(stamp.fingerprint for stamp in review_stamps))
    created_at = time.time()
    payload = {
        "question_fingerprint": question.fingerprint,
        "accepted_studies": accepted_study_fps,
        "rejected_studies": sorted(rejected_studies),
        "accepted_effects": accepted_effect_fps,
        "rejected_effects": sorted(rejected_effect_fps),
        "reviewed_rob_studies": reviewed_rob_fps,
        "quality_summary": quality,
        "synthesis_groups": [asdict(item) for item in synthesis_groups],
        "certainty": [asdict(item) for item in certainty_rows],
        "causal_readiness": causal_rows,
        "numerical_checks": [asdict(item) for item in numerical_rows],
        "unresolved": sorted(set(unresolved)),
        "review_stamps": stamp_fps,
    }
    fingerprint = hashlib.sha256(_canonical(payload)).hexdigest()
    return ScientificEvidenceBundle(
        question=question,
        accepted_study_fingerprints=accepted_study_fps,
        rejected_or_unreviewed_study_fingerprints=tuple(sorted(rejected_studies)),
        accepted_effect_fingerprints=accepted_effect_fps,
        rejected_or_unreviewed_effect_fingerprints=tuple(sorted(rejected_effect_fps)),
        risk_of_bias_reviewed_study_fingerprints=reviewed_rob_fps,
        quality_summary=quality,
        synthesis_groups=tuple(synthesis_groups),
        certainty=tuple(certainty_rows),
        causal_readiness=tuple(causal_rows),
        numerical_checks=tuple(numerical_rows),
        unresolved=tuple(sorted(set(unresolved))),
        review_stamp_fingerprints=stamp_fps,
        created_at=created_at,
        fingerprint=fingerprint,
    )


__all__ = [
    "EvidenceReviewStamp",
    "ScientificEvidenceBundle",
    "SynthesisGroup",
    "build_scientific_evidence_bundle",
    "effect_fingerprint",
    "risk_of_bias_fingerprint",
    "study_fingerprint",
]
