"""Deterministic, privacy-safe evaluation for scientific claim extraction proposals."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from tools.evidence_graph_claim_contracts import (
    CLAIM_MODALITIES,
    CLAIM_TYPES,
    ClaimEvidenceLocator,
    ScientificClaimProposal,
    _digest,
    _identifier,
    _integer,
    _sha256,
)
from tools.security import normalize_owner_id

_TOKEN = re.compile(r"\w+", re.UNICODE)
_MAX_CASES = 100_000


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")
    selected = value.strip()
    if not selected or len(selected) > 10_000 or "\x00" in selected:
        raise ValueError(f"{label} is invalid or too long.")
    return selected


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN.findall(value))


def _f1(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return float(left == right)
    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    for value in left:
        left_counts[value] = left_counts.get(value, 0) + 1
    for value in right:
        right_counts[value] = right_counts.get(value, 0) + 1
    overlap = sum(
        min(count, right_counts.get(key, 0))
        for key, count in left_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2.0 * precision * recall / (precision + recall)


def _span_iou(
    left: ClaimEvidenceLocator,
    right: ClaimEvidenceLocator,
) -> float:
    if left.section_index != right.section_index:
        return 0.0
    intersection = max(
        0,
        min(left.char_end, right.char_end)
        - max(left.char_start, right.char_start),
    )
    union = max(left.char_end, right.char_end) - min(
        left.char_start,
        right.char_start,
    )
    return 0.0 if union <= 0 else intersection / union


def _ratio(numerator: float, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _harmonic(precision: float, recall: float) -> float:
    return (
        0.0
        if precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )


def _finite_metric(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and between 0 and 1.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{label} must be finite and between 0 and 1."
        ) from exc
    if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be finite and between 0 and 1.")
    return selected


@dataclass(frozen=True)
class ScientificClaimGold:
    gold_id: str
    owner_id: str
    doc_id: str
    generation: int
    content_sha256: str
    profile_fingerprint: str
    claim_text: str
    claim_type: str
    modality: str
    locator: ClaimEvidenceLocator
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gold_id",
            _identifier(self.gold_id, "gold_id", 500),
        )
        object.__setattr__(
            self,
            "owner_id",
            normalize_owner_id(self.owner_id),
        )
        object.__setattr__(
            self,
            "doc_id",
            _identifier(self.doc_id, "doc_id", 200),
        )
        object.__setattr__(
            self,
            "generation",
            _integer(self.generation, "generation", 1, 2**63 - 1),
        )
        object.__setattr__(
            self,
            "content_sha256",
            _digest(self.content_sha256, "content_sha256"),
        )
        object.__setattr__(
            self,
            "profile_fingerprint",
            _digest(self.profile_fingerprint, "profile_fingerprint"),
        )
        object.__setattr__(
            self,
            "claim_text",
            _text(self.claim_text, "claim_text"),
        )
        claim_type = _identifier(self.claim_type, "claim_type", 50)
        if claim_type not in CLAIM_TYPES:
            raise ValueError("claim_type is unsupported.")
        object.__setattr__(self, "claim_type", claim_type)
        modality = _identifier(self.modality, "modality", 50)
        if modality not in CLAIM_MODALITIES:
            raise ValueError("modality is unsupported.")
        object.__setattr__(self, "modality", modality)
        if not isinstance(self.locator, ClaimEvidenceLocator):
            raise ValueError("locator must be ClaimEvidenceLocator.")
        if self.schema_version != 1:
            raise ValueError("scientific claim gold schema is unsupported.")

    @property
    def claim_text_sha256(self) -> str:
        return hashlib.sha256(
            self.claim_text.encode("utf-8")
        ).hexdigest()

    @property
    def gold_digest(self) -> str:
        return _sha256(
            {
                "scope": "rigorousrag-scientific-claim-gold-v1",
                "gold_id": self.gold_id,
                "owner_id": self.owner_id,
                "doc_id": self.doc_id,
                "generation": self.generation,
                "content_sha256": self.content_sha256,
                "profile_fingerprint": self.profile_fingerprint,
                "claim_text_sha256": self.claim_text_sha256,
                "claim_type": self.claim_type,
                "modality": self.modality,
                "locator_digest": self.locator.locator_digest,
            }
        )


@dataclass(frozen=True)
class ScientificClaimEvaluationMatch:
    gold_id: str
    gold_digest: str
    proposal_id: str
    proposal_digest: str
    exact_evidence_digest: bool
    exact_locator: bool
    span_iou: float
    claim_token_f1: float
    claim_type_correct: bool
    modality_correct: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gold_id",
            _identifier(self.gold_id, "gold_id", 500),
        )
        for name in (
            "gold_digest",
            "proposal_id",
            "proposal_digest",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )
        for name in (
            "exact_evidence_digest",
            "exact_locator",
            "claim_type_correct",
            "modality_correct",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")
        object.__setattr__(
            self,
            "span_iou",
            _finite_metric(self.span_iou, "span_iou"),
        )
        object.__setattr__(
            self,
            "claim_token_f1",
            _finite_metric(self.claim_token_f1, "claim_token_f1"),
        )


@dataclass(frozen=True)
class ScientificClaimEvaluationReport:
    owner_id: str
    doc_id: str
    generation: int
    content_sha256: str
    profile_fingerprint: str
    gold_count: int
    proposal_count: int
    matched_count: int
    precision: float
    recall: float
    f1: float
    exact_evidence_accuracy: float
    exact_locator_accuracy: float
    mean_span_iou: float
    mean_claim_token_f1: float
    claim_type_accuracy: float
    modality_accuracy: float
    confidence_brier_score: float
    unmatched_gold_ids: tuple[str, ...]
    unmatched_proposal_ids: tuple[str, ...]
    matches: tuple[ScientificClaimEvaluationMatch, ...]
    report_digest: str
    contains_claim_text: bool = False
    contains_evidence_text: bool = False
    semantic_entailment_evaluated: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "owner_id",
            normalize_owner_id(self.owner_id),
        )
        object.__setattr__(
            self,
            "doc_id",
            _identifier(self.doc_id, "doc_id", 200),
        )
        object.__setattr__(
            self,
            "generation",
            _integer(self.generation, "generation", 1, 2**63 - 1),
        )
        object.__setattr__(
            self,
            "content_sha256",
            _digest(self.content_sha256, "content_sha256"),
        )
        object.__setattr__(
            self,
            "profile_fingerprint",
            _digest(self.profile_fingerprint, "profile_fingerprint"),
        )
        for name in (
            "gold_count",
            "proposal_count",
            "matched_count",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name,
                    0,
                    _MAX_CASES,
                ),
            )
        if self.matched_count > min(
            self.gold_count,
            self.proposal_count,
        ):
            raise ValueError(
                "matched_count exceeds available claims."
            )
        for name in (
            "precision",
            "recall",
            "f1",
            "exact_evidence_accuracy",
            "exact_locator_accuracy",
            "mean_span_iou",
            "mean_claim_token_f1",
            "claim_type_accuracy",
            "modality_accuracy",
            "confidence_brier_score",
        ):
            object.__setattr__(
                self,
                name,
                _finite_metric(getattr(self, name), name),
            )
        if (
            not isinstance(self.unmatched_gold_ids, tuple)
            or any(
                not isinstance(value, str)
                for value in self.unmatched_gold_ids
            )
        ):
            raise ValueError(
                "unmatched_gold_ids must be a tuple of IDs."
            )
        object.__setattr__(
            self,
            "unmatched_gold_ids",
            tuple(
                _identifier(value, "gold_id", 500)
                for value in self.unmatched_gold_ids
            ),
        )
        if not isinstance(
            self.unmatched_proposal_ids,
            tuple,
        ):
            raise ValueError(
                "unmatched_proposal_ids must be a tuple."
            )
        object.__setattr__(
            self,
            "unmatched_proposal_ids",
            tuple(
                _digest(value, "proposal_id")
                for value in self.unmatched_proposal_ids
            ),
        )
        if (
            not isinstance(self.matches, tuple)
            or any(
                not isinstance(
                    value,
                    ScientificClaimEvaluationMatch,
                )
                for value in self.matches
            )
        ):
            raise ValueError(
                "matches must be a tuple of evaluation matches."
            )
        if len(self.matches) != self.matched_count:
            raise ValueError(
                "match count differs from matches."
            )
        if len(self.unmatched_gold_ids) != (
            self.gold_count - self.matched_count
        ):
            raise ValueError(
                "unmatched gold count is inconsistent."
            )
        if len(self.unmatched_proposal_ids) != (
            self.proposal_count - self.matched_count
        ):
            raise ValueError(
                "unmatched proposal count is inconsistent."
            )
        for name in (
            "contains_claim_text",
            "contains_evidence_text",
            "semantic_entailment_evaluated",
        ):
            if getattr(self, name) is not False:
                raise ValueError(
                    "evaluation safety flags must remain false."
                )
        object.__setattr__(
            self,
            "report_digest",
            _digest(self.report_digest, "report_digest"),
        )
        if self.schema_version != 1:
            raise ValueError(
                "scientific claim evaluation schema is unsupported."
            )


def evaluate_scientific_claim_extraction(
    *,
    gold: Iterable[ScientificClaimGold],
    proposals: Iterable[ScientificClaimProposal],
    minimum_span_iou: float = 0.5,
    minimum_claim_token_f1: float = 0.5,
) -> ScientificClaimEvaluationReport:
    """Evaluate exact provenance and lexical claim quality without retaining text."""

    if isinstance(gold, (str, bytes, bytearray)) or isinstance(
        proposals,
        (str, bytes, bytearray),
    ):
        raise ValueError("gold and proposals must be iterables.")
    gold_values = tuple(gold)
    proposal_values = tuple(proposals)
    if not 1 <= len(gold_values) <= _MAX_CASES:
        raise ValueError(
            "gold must contain a bounded non-empty set."
        )
    if len(proposal_values) > _MAX_CASES:
        raise ValueError(
            "proposals exceeds the item limit."
        )
    if any(
        not isinstance(value, ScientificClaimGold)
        for value in gold_values
    ):
        raise ValueError(
            "every gold value must be ScientificClaimGold."
        )
    if any(
        not isinstance(value, ScientificClaimProposal)
        for value in proposal_values
    ):
        raise ValueError(
            "every proposal must be ScientificClaimProposal."
        )
    span_threshold = _finite_metric(
        minimum_span_iou,
        "minimum_span_iou",
    )
    text_threshold = _finite_metric(
        minimum_claim_token_f1,
        "minimum_claim_token_f1",
    )
    if len({value.gold_id for value in gold_values}) != len(
        gold_values
    ):
        raise ValueError("gold contains duplicate IDs.")
    if len(
        {value.proposal_id for value in proposal_values}
    ) != len(proposal_values):
        raise ValueError("proposals contains duplicate IDs.")

    scope = (
        gold_values[0].owner_id,
        gold_values[0].doc_id,
        gold_values[0].generation,
        gold_values[0].content_sha256,
        gold_values[0].profile_fingerprint,
    )
    for value in (*gold_values, *proposal_values):
        current = (
            value.owner_id,
            value.doc_id,
            value.generation,
            value.content_sha256,
            value.profile_fingerprint,
        )
        if current != scope:
            raise PermissionError(
                "claim evaluation values differ in generation scope."
            )

    candidates: list[
        tuple[
            tuple[Any, ...],
            ScientificClaimGold,
            ScientificClaimProposal,
            float,
            float,
        ]
    ] = []
    for gold_value in gold_values:
        gold_tokens = _tokens(gold_value.claim_text)
        for proposal in proposal_values:
            span_iou = _span_iou(
                gold_value.locator,
                proposal.locator,
            )
            text_f1 = _f1(
                gold_tokens,
                _tokens(proposal.claim_text),
            )
            exact_evidence = (
                gold_value.locator.evidence_sha256
                == proposal.locator.evidence_sha256
            )
            if not (
                exact_evidence
                or (
                    span_iou >= span_threshold
                    and text_f1 >= text_threshold
                )
            ):
                continue
            score = (
                int(exact_evidence),
                span_iou,
                text_f1,
                int(
                    gold_value.claim_type
                    == proposal.claim_type
                ),
                int(
                    gold_value.modality
                    == proposal.modality
                ),
                gold_value.gold_id,
                proposal.proposal_id,
            )
            candidates.append(
                (
                    score,
                    gold_value,
                    proposal,
                    span_iou,
                    text_f1,
                )
            )

    used_gold: set[str] = set()
    used_proposals: set[str] = set()
    matches: list[ScientificClaimEvaluationMatch] = []
    for (
        _score,
        gold_value,
        proposal,
        span_iou,
        text_f1,
    ) in sorted(
        candidates,
        key=lambda value: (
            -value[0][0],
            -value[0][1],
            -value[0][2],
            -value[0][3],
            -value[0][4],
            value[0][5],
            value[0][6],
        ),
    ):
        if (
            gold_value.gold_id in used_gold
            or proposal.proposal_id in used_proposals
        ):
            continue
        used_gold.add(gold_value.gold_id)
        used_proposals.add(proposal.proposal_id)
        exact_locator = bool(
            gold_value.locator.section_index
            == proposal.locator.section_index
            and gold_value.locator.page_number
            == proposal.locator.page_number
            and gold_value.locator.char_start
            == proposal.locator.char_start
            and gold_value.locator.char_end
            == proposal.locator.char_end
        )
        matches.append(
            ScientificClaimEvaluationMatch(
                gold_id=gold_value.gold_id,
                gold_digest=gold_value.gold_digest,
                proposal_id=proposal.proposal_id,
                proposal_digest=proposal.proposal_digest,
                exact_evidence_digest=(
                    gold_value.locator.evidence_sha256
                    == proposal.locator.evidence_sha256
                ),
                exact_locator=exact_locator,
                span_iou=span_iou,
                claim_token_f1=text_f1,
                claim_type_correct=(
                    gold_value.claim_type
                    == proposal.claim_type
                ),
                modality_correct=(
                    gold_value.modality
                    == proposal.modality
                ),
            )
        )

    rendered_matches = tuple(
        sorted(
            matches,
            key=lambda value: (
                value.gold_id,
                value.proposal_id,
            ),
        )
    )
    matched = len(rendered_matches)
    precision = _ratio(
        matched,
        len(proposal_values),
    )
    recall = _ratio(
        matched,
        len(gold_values),
    )
    unmatched_gold = tuple(
        sorted(
            value.gold_id
            for value in gold_values
            if value.gold_id not in used_gold
        )
    )
    unmatched_proposals = tuple(
        sorted(
            value.proposal_id
            for value in proposal_values
            if value.proposal_id not in used_proposals
        )
    )
    exact_evidence_accuracy = _ratio(
        sum(
            value.exact_evidence_digest
            for value in rendered_matches
        ),
        matched,
    )
    exact_locator_accuracy = _ratio(
        sum(
            value.exact_locator
            for value in rendered_matches
        ),
        matched,
    )
    mean_span_iou = _ratio(
        sum(
            value.span_iou
            for value in rendered_matches
        ),
        matched,
    )
    mean_claim_token_f1 = _ratio(
        sum(
            value.claim_token_f1
            for value in rendered_matches
        ),
        matched,
    )
    claim_type_accuracy = _ratio(
        sum(
            value.claim_type_correct
            for value in rendered_matches
        ),
        matched,
    )
    modality_accuracy = _ratio(
        sum(
            value.modality_correct
            for value in rendered_matches
        ),
        matched,
    )
    confidence_brier = _ratio(
        sum(
            (
                value.confidence
                - float(
                    value.proposal_id
                    in used_proposals
                )
            )
            ** 2
            for value in proposal_values
        ),
        len(proposal_values),
    )
    report_values = {
        "owner_id": scope[0],
        "doc_id": scope[1],
        "generation": scope[2],
        "content_sha256": scope[3],
        "profile_fingerprint": scope[4],
        "gold_count": len(gold_values),
        "proposal_count": len(proposal_values),
        "matched_count": matched,
        "precision": precision,
        "recall": recall,
        "f1": _harmonic(precision, recall),
        "exact_evidence_accuracy": exact_evidence_accuracy,
        "exact_locator_accuracy": exact_locator_accuracy,
        "mean_span_iou": mean_span_iou,
        "mean_claim_token_f1": mean_claim_token_f1,
        "claim_type_accuracy": claim_type_accuracy,
        "modality_accuracy": modality_accuracy,
        "confidence_brier_score": confidence_brier,
        "unmatched_gold_ids": unmatched_gold,
        "unmatched_proposal_ids": unmatched_proposals,
        "matches": rendered_matches,
    }
    stable = {
        "scope": "rigorousrag-scientific-claim-evaluation-report-v1",
        **{
            **report_values,
            "matches": [
                asdict(value)
                for value in rendered_matches
            ],
        },
        "minimum_span_iou": span_threshold,
        "minimum_claim_token_f1": text_threshold,
    }
    return ScientificClaimEvaluationReport(
        **report_values,
        report_digest=_sha256(stable),
    )


__all__ = [
    "ScientificClaimEvaluationMatch",
    "ScientificClaimEvaluationReport",
    "ScientificClaimGold",
    "evaluate_scientific_claim_extraction",
]
