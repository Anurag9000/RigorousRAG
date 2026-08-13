"""Proof-carrying answer schemas and deterministic evidence-path validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClaimProof:
    claim_id: str
    claim_text: str
    evidence_ids: tuple[str, ...]
    support_paths: tuple[tuple[str, ...], ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.claim_text.strip():
            raise ValueError("claim_id and claim_text are required.")
        if not self.evidence_ids:
            raise ValueError("each proof must cite at least one evidence item.")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1].")
        for path in self.support_paths:
            if len(path) < 2 or any(not str(node).strip() for node in path):
                raise ValueError("support paths must contain at least two non-empty nodes.")


@dataclass(frozen=True)
class ProofCarryingAnswer:
    answer: str
    claims: tuple[ClaimProof, ...]
    evidence_catalog: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.answer.strip():
            raise ValueError("answer is required.")
        ids = [claim.claim_id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("claim identifiers must be unique.")


@dataclass(frozen=True)
class ProofValidation:
    valid: bool
    claim_coverage: float
    evidence_resolution: float
    path_resolution: float
    missing_evidence_ids: tuple[str, ...]
    invalid_paths: tuple[tuple[str, ...], ...]


def validate_proof(
    answer: ProofCarryingAnswer,
    *,
    known_evidence_ids: Iterable[str] | None = None,
    known_edges: Iterable[tuple[str, str]] | None = None,
) -> ProofValidation:
    """Validate claim coverage, evidence references and optional path adjacency."""

    evidence_ids = (
        set(answer.evidence_catalog)
        if known_evidence_ids is None
        else {str(value) for value in known_evidence_ids}
    )
    edges = None if known_edges is None else {(str(a), str(b)) for a, b in known_edges}
    missing = []
    invalid_paths = []
    resolved_claims = 0
    resolved_references = 0
    total_references = 0
    resolved_paths = 0
    total_paths = 0

    for claim in answer.claims:
        claim_resolved = True
        for evidence_id in claim.evidence_ids:
            total_references += 1
            if evidence_id in evidence_ids:
                resolved_references += 1
            else:
                missing.append(evidence_id)
                claim_resolved = False
        for path in claim.support_paths:
            total_paths += 1
            valid_path = True
            if edges is not None:
                for left, right in zip(path, path[1:]):
                    if (left, right) not in edges and (right, left) not in edges:
                        valid_path = False
                        break
            terminal_matches = any(node in claim.evidence_ids for node in path)
            if not terminal_matches:
                valid_path = False
            if valid_path:
                resolved_paths += 1
            else:
                invalid_paths.append(path)
                claim_resolved = False
        if claim_resolved:
            resolved_claims += 1

    claim_coverage = resolved_claims / len(answer.claims) if answer.claims else 1.0
    evidence_resolution = (
        resolved_references / total_references if total_references else 1.0
    )
    path_resolution = resolved_paths / total_paths if total_paths else 1.0
    return ProofValidation(
        valid=claim_coverage == 1.0 and evidence_resolution == 1.0 and path_resolution == 1.0,
        claim_coverage=claim_coverage,
        evidence_resolution=evidence_resolution,
        path_resolution=path_resolution,
        missing_evidence_ids=tuple(sorted(set(missing))),
        invalid_paths=tuple(invalid_paths),
    )
