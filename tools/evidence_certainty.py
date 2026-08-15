"""Transparent evidence-certainty bookkeeping for scientific synthesis.

The implementation provides a GRADE-inspired *data structure and deterministic rule
engine*, not an automated clinical guideline system. Domain judgments must be supplied
by reviewed evidence records; no LLM/model judgment is silently treated as ground truth.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

_LEVELS = ("very_low", "low", "moderate", "high")
_DOMAINS = (
    "risk_of_bias",
    "inconsistency",
    "indirectness",
    "imprecision",
    "publication_bias",
    "large_effect",
    "dose_response",
    "residual_confounding",
)


def _text(value: Any, label: str, maximum: int = 2000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class CertaintyDomainJudgment:
    domain: str
    adjustment: int
    rationale: str
    evidence_ids: tuple[str, ...] = ()
    reviewer_id: str = ""
    reviewed: bool = False

    def __post_init__(self) -> None:
        domain = _text(self.domain, "domain", 64).lower()
        if domain not in _DOMAINS:
            raise ValueError("unsupported certainty domain")
        object.__setattr__(self, "domain", domain)
        if isinstance(self.adjustment, bool) or not isinstance(self.adjustment, int) or not -2 <= self.adjustment <= 2:
            raise ValueError("certainty adjustment must be an integer from -2 to 2")
        if domain in {"risk_of_bias", "inconsistency", "indirectness", "imprecision", "publication_bias"} and self.adjustment > 0:
            raise ValueError("downgrade domains may not increase certainty")
        if domain in {"large_effect", "dose_response", "residual_confounding"} and self.adjustment < 0:
            raise ValueError("upgrade domains may not decrease certainty")
        object.__setattr__(self, "rationale", _text(self.rationale, "rationale", 5000))
        if len(self.evidence_ids) > 1000:
            raise ValueError("evidence_ids exceed the item limit")
        object.__setattr__(self, "evidence_ids", tuple(dict.fromkeys(_text(item, "evidence_id", 500) for item in self.evidence_ids)))
        object.__setattr__(self, "reviewer_id", _text(self.reviewer_id, "reviewer_id", 256, allow_empty=True))
        if not isinstance(self.reviewed, bool):
            raise ValueError("reviewed must be boolean")


@dataclass(frozen=True)
class EvidenceCertaintyAssessment:
    question_fingerprint: str
    outcome: str
    initial_level: str
    final_level: str
    judgments: tuple[CertaintyDomainJudgment, ...]
    reviewed_fraction: float
    unresolved_domains: tuple[str, ...]
    fingerprint: str
    note: str = "Structured certainty bookkeeping is not a substitute for expert guideline review."


def assess_certainty(
    *,
    question_fingerprint: str,
    outcome: str,
    study_design_family: str,
    judgments: Sequence[CertaintyDomainJudgment],
) -> EvidenceCertaintyAssessment:
    qf = _text(question_fingerprint, "question_fingerprint", 64).lower()
    if len(qf) != 64 or any(ch not in "0123456789abcdef" for ch in qf):
        raise ValueError("question_fingerprint must be SHA-256")
    selected_outcome = _text(outcome, "outcome", 1000)
    family = _text(study_design_family, "study_design_family", 64).lower()
    if family in {"randomized", "randomized_controlled_trial", "rct"}:
        initial_index = 3
        initial = "high"
    elif family in {"observational", "cohort", "case_control", "cross_sectional", "nonrandomized"}:
        initial_index = 1
        initial = "low"
    else:
        initial_index = 1
        initial = "low"
    if len(judgments) > len(_DOMAINS) * 4:
        raise ValueError("too many certainty judgments")
    if any(not isinstance(item, CertaintyDomainJudgment) for item in judgments):
        raise ValueError("judgments must contain CertaintyDomainJudgment values")

    by_domain: dict[str, list[CertaintyDomainJudgment]] = {}
    for item in judgments:
        by_domain.setdefault(item.domain, []).append(item)
    # Multiple reviewed judgments in one domain combine conservatively: strongest
    # downgrade/upgrade only, never double-counting repeated reviewers.
    total_adjustment = 0
    for domain, rows in by_domain.items():
        adjustments = [row.adjustment for row in rows if row.reviewed]
        if not adjustments:
            continue
        if domain in {"risk_of_bias", "inconsistency", "indirectness", "imprecision", "publication_bias"}:
            total_adjustment += min(adjustments)
        else:
            total_adjustment += max(adjustments)
    final_index = max(0, min(3, initial_index + total_adjustment))
    final = _LEVELS[final_index]
    reviewed_count = sum(1 for item in judgments if item.reviewed)
    reviewed_fraction = reviewed_count / len(judgments) if judgments else 0.0
    unresolved = tuple(sorted(domain for domain in _DOMAINS if domain not in by_domain or not any(row.reviewed for row in by_domain[domain])))
    payload = {
        "question_fingerprint": qf,
        "outcome": selected_outcome,
        "study_design_family": family,
        "initial_level": initial,
        "final_level": final,
        "judgments": [asdict(item) for item in judgments],
        "reviewed_fraction": reviewed_fraction,
        "unresolved_domains": unresolved,
    }
    return EvidenceCertaintyAssessment(
        qf,
        selected_outcome,
        initial,
        final,
        tuple(judgments),
        reviewed_fraction,
        unresolved,
        hashlib.sha256(_canonical(payload)).hexdigest(),
    )


def certainty_summary(assessments: Sequence[EvidenceCertaintyAssessment]) -> Mapping[str, Any]:
    counts = {level: 0 for level in _LEVELS}
    for item in assessments:
        if not isinstance(item, EvidenceCertaintyAssessment):
            raise ValueError("assessments contain an invalid value")
        counts[item.final_level] += 1
    return {
        "outcomes": len(assessments),
        "final_level_counts": counts,
        "fully_reviewed_outcomes": sum(1 for item in assessments if not item.unresolved_domains),
        "note": "Outcome-level certainty must be interpreted with the underlying judgments and evidence.",
    }


__all__ = ["CertaintyDomainJudgment", "EvidenceCertaintyAssessment", "assess_certainty", "certainty_summary"]
