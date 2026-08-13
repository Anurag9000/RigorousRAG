"""Evidence independence, diversity, minimality, and contradiction-aware scoring."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    source_id: str
    root_source_id: str
    supports_claims: frozenset[str]
    score: float = 1.0
    contradicts_claims: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.evidence_id.strip() or not self.source_id.strip() or not self.root_source_id.strip():
            raise ValueError("evidence and source identifiers are required.")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be in [0, 1].")


@dataclass(frozen=True)
class EvidenceQualityReport:
    claim_coverage: float
    independent_source_ratio: float
    redundancy_rate: float
    contradiction_rate: float
    weighted_support: float
    selected_evidence_ids: tuple[str, ...]


def independent_source_ratio(evidence: Sequence[EvidenceItem]) -> float:
    if not evidence:
        return 0.0
    roots = {item.root_source_id for item in evidence}
    return len(roots) / len(evidence)


def contradiction_rate(evidence: Sequence[EvidenceItem], claims: Iterable[str]) -> float:
    claim_set = set(claims)
    if not claim_set:
        return 0.0
    contradicted = {
        claim for item in evidence for claim in item.contradicts_claims if claim in claim_set
    }
    return len(contradicted) / len(claim_set)


def minimal_evidence_cover(
    evidence: Sequence[EvidenceItem],
    required_claims: Iterable[str],
    *,
    prefer_independent_roots: bool = True,
) -> tuple[EvidenceItem, ...]:
    """Greedy set cover with source-independence and evidence-score tie breaking."""

    remaining = set(required_claims)
    if not remaining:
        return ()
    candidates = list(evidence)
    selected: list[EvidenceItem] = []
    roots = set()
    while remaining:
        ranked = []
        for item in candidates:
            newly_covered = item.supports_claims & remaining
            if not newly_covered:
                continue
            root_bonus = 1 if prefer_independent_roots and item.root_source_id not in roots else 0
            ranked.append(
                (
                    len(newly_covered),
                    root_bonus,
                    item.score,
                    -len(item.supports_claims),
                    item.evidence_id,
                    item,
                )
            )
        if not ranked:
            break
        ranked.sort(key=lambda row: (-row[0], -row[1], -row[2], -row[3], row[4]))
        chosen = ranked[0][-1]
        selected.append(chosen)
        roots.add(chosen.root_source_id)
        remaining -= chosen.supports_claims
        candidates = [item for item in candidates if item.evidence_id != chosen.evidence_id]
    return tuple(selected)


def evidence_quality_report(
    evidence: Sequence[EvidenceItem],
    required_claims: Iterable[str],
) -> EvidenceQualityReport:
    claims = set(required_claims)
    selected = minimal_evidence_cover(evidence, claims)
    covered = {claim for item in selected for claim in item.supports_claims if claim in claims}
    coverage = len(covered) / len(claims) if claims else 1.0
    root_ratio = independent_source_ratio(selected)
    redundancy = 1.0 - root_ratio if selected else 0.0
    contradictions = contradiction_rate(evidence, claims)
    if selected:
        weighted = sum(
            item.score * len(item.supports_claims & claims) for item in selected
        ) / sum(max(len(item.supports_claims & claims), 1) for item in selected)
    else:
        weighted = 0.0
    return EvidenceQualityReport(
        claim_coverage=coverage,
        independent_source_ratio=root_ratio,
        redundancy_rate=redundancy,
        contradiction_rate=contradictions,
        weighted_support=weighted,
        selected_evidence_ids=tuple(item.evidence_id for item in selected),
    )


def source_dependency_groups(evidence: Sequence[EvidenceItem]) -> Mapping[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for item in evidence:
        grouped.setdefault(item.root_source_id, []).append(item.evidence_id)
    return {
        root: tuple(sorted(identifiers)) for root, identifiers in sorted(grouped.items())
    }
