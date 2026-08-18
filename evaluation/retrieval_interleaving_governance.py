"""Sealed governance receipt for randomized retrieval interleaving promotion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from evaluation.retrieval_interleaving import InterleavingImpression, InterleavingOutcome, InterleavingSpec
from evaluation.retrieval_interleaving_promotion import (
    InterleavingPromotionPolicy,
    InterleavingPromotionReceipt,
    build_interleaving_evidence,
    qualify_interleaving_experiment,
)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return selected


@dataclass(frozen=True)
class GovernedInterleavingPromotionReceipt:
    spec_sha256: str
    evidence_pairs: tuple[tuple[str, str], ...]
    evidence_sha256: str
    promotion_policy_sha256: str
    promotion_receipt_sha256: str
    candidate_policy_sha256: str
    eligible: bool
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "spec_sha256",
            "evidence_sha256",
            "promotion_policy_sha256",
            "promotion_receipt_sha256",
            "candidate_policy_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        pairs = tuple(sorted((_sha(left, "impression sha256"), _sha(right, "outcome sha256")) for left, right in self.evidence_pairs))
        if not pairs or len({left for left, _ in pairs}) != len(pairs) or len({right for _, right in pairs}) != len(pairs):
            raise ValueError("evidence_pairs must be non-empty and one-to-one")
        object.__setattr__(self, "evidence_pairs", pairs)
        if not isinstance(self.eligible, bool):
            raise ValueError("eligible must be boolean")
        expected = _digest(self._payload())
        provided = _sha(self.receipt_sha256, "receipt_sha256")
        if expected != provided:
            raise ValueError("receipt_sha256 does not match governed interleaving promotion content")
        object.__setattr__(self, "receipt_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-governed-interleaving-promotion/v1",
            "spec_sha256": self.spec_sha256,
            "evidence_pairs": self.evidence_pairs,
            "evidence_sha256": self.evidence_sha256,
            "promotion_policy_sha256": self.promotion_policy_sha256,
            "promotion_receipt_sha256": self.promotion_receipt_sha256,
            "candidate_policy_sha256": self.candidate_policy_sha256,
            "eligible": self.eligible,
        }


@dataclass(frozen=True)
class GovernedInterleavingPromotion:
    promotion: InterleavingPromotionReceipt
    receipt: GovernedInterleavingPromotionReceipt


def run_governed_interleaving_promotion(
    spec: InterleavingSpec,
    impressions: Sequence[InterleavingImpression],
    outcomes: Sequence[InterleavingOutcome],
    *,
    policy: InterleavingPromotionPolicy = InterleavingPromotionPolicy(),
) -> GovernedInterleavingPromotion:
    impression_rows = tuple(impressions)
    outcome_rows = tuple(outcomes)
    evidence = build_interleaving_evidence(spec, impression_rows, outcome_rows)
    promotion = qualify_interleaving_experiment(spec, evidence, policy=policy)
    outcomes_by_impression = {outcome.impression_sha256: outcome for outcome in outcome_rows}
    pairs = tuple(sorted((impression.impression_sha256, outcomes_by_impression[impression.impression_sha256].outcome_sha256) for impression in impression_rows))
    payload = {
        "schema": "rigorousrag-governed-interleaving-promotion/v1",
        "spec_sha256": spec.spec_sha256,
        "evidence_pairs": pairs,
        "evidence_sha256": evidence.evidence_sha256,
        "promotion_policy_sha256": policy.policy_sha256,
        "promotion_receipt_sha256": promotion.receipt_sha256,
        "candidate_policy_sha256": promotion.candidate_policy_sha256,
        "eligible": promotion.eligible,
    }
    receipt = GovernedInterleavingPromotionReceipt(
        spec_sha256=payload["spec_sha256"],
        evidence_pairs=payload["evidence_pairs"],
        evidence_sha256=payload["evidence_sha256"],
        promotion_policy_sha256=payload["promotion_policy_sha256"],
        promotion_receipt_sha256=payload["promotion_receipt_sha256"],
        candidate_policy_sha256=payload["candidate_policy_sha256"],
        eligible=payload["eligible"],
        receipt_sha256=_digest(payload),
    )
    return GovernedInterleavingPromotion(promotion=promotion, receipt=receipt)


__all__ = ["GovernedInterleavingPromotion", "GovernedInterleavingPromotionReceipt", "run_governed_interleaving_promotion"]
