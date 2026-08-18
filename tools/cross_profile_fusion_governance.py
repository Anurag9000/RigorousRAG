"""Content-addressed governance receipts for heterogeneous retrieval fusion.

The mathematical fusion lives in :mod:`tools.cross_profile_fusion`.  This layer binds a
fusion decision to the exact ranked inputs, score-profile identities, calibration
artifacts and policy without persisting raw query/document text.  It also applies the
same per-list candidate ceiling as the underlying RRF engine before any fusion work.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.cross_profile_fusion import (
    CrossProfileFusionPolicy,
    CrossProfileFusionResult,
    IsotonicCalibrationArtifact,
    ProfileRankedList,
    fuse_cross_profile_rankings,
)

_MAX_CANDIDATES_PER_LIST = 2_000_000


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return selected


def fusion_policy_sha256(policy: CrossProfileFusionPolicy) -> str:
    if not isinstance(policy, CrossProfileFusionPolicy):
        raise ValueError("policy must be CrossProfileFusionPolicy.")
    return _canonical_digest(
        {
            "schema": "rigorousrag-cross-profile-fusion-policy/v1",
            "mode": policy.mode.value,
            "profile_weights": sorted(policy.profile_weights.items()),
            "max_fused_candidates": policy.max_fused_candidates,
            "max_per_document": policy.max_per_document,
            "max_per_source": policy.max_per_source,
            "rrf_k": policy.rrf_k,
        }
    )


def ranked_input_sha256(ranked_lists: Sequence[ProfileRankedList]) -> str:
    lists = tuple(ranked_lists)
    if not lists:
        raise ValueError("ranked_lists must be non-empty.")
    payload_lists: list[dict[str, Any]] = []
    seen_list_ids: set[str] = set()
    for item in sorted(lists, key=lambda value: value.list_id):
        if not isinstance(item, ProfileRankedList):
            raise ValueError("ranked_lists must contain ProfileRankedList values.")
        if item.list_id in seen_list_ids:
            raise ValueError("list_id values must be unique.")
        seen_list_ids.add(item.list_id)
        if len(item.candidates) > _MAX_CANDIDATES_PER_LIST:
            raise ValueError("profile ranked list exceeds the governed RRF candidate limit.")
        payload_lists.append(
            {
                "list_id": item.list_id,
                "profile_sha256": item.profile.profile_sha256,
                "candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "corpus_id": candidate.corpus_id,
                        "retriever_id": candidate.retriever_id,
                        "document_id": candidate.document_id,
                        "chunk_id": candidate.chunk_id,
                        "rank": candidate.rank,
                        "raw_score": candidate.raw_score,
                        "source_id": candidate.source_id,
                    }
                    for candidate in sorted(
                        item.candidates,
                        key=lambda candidate: (candidate.rank, candidate.candidate_id),
                    )
                ],
            }
        )
    return _canonical_digest(
        {
            "schema": "rigorousrag-cross-profile-ranked-input/v1",
            "lists": payload_lists,
        }
    )


@dataclass(frozen=True)
class GovernedCrossProfileFusionReceipt:
    input_sha256: str
    policy_sha256: str
    result_sha256: str
    mode: str
    calibration_contract_sha256: str | None
    profile_artifact_sha256s: tuple[tuple[str, str], ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_sha256", _sha256(self.input_sha256, "input_sha256"))
        object.__setattr__(self, "policy_sha256", _sha256(self.policy_sha256, "policy_sha256"))
        object.__setattr__(self, "result_sha256", _sha256(self.result_sha256, "result_sha256"))
        if self.calibration_contract_sha256 is not None:
            object.__setattr__(
                self,
                "calibration_contract_sha256",
                _sha256(self.calibration_contract_sha256, "calibration_contract_sha256"),
            )
        artifacts = tuple(sorted(self.profile_artifact_sha256s))
        for profile_id, digest in artifacts:
            if not isinstance(profile_id, str) or not profile_id.strip():
                raise ValueError("profile artifact ids must be non-empty strings.")
            _sha256(digest, "profile artifact sha256")
        object.__setattr__(self, "profile_artifact_sha256s", artifacts)
        expected = _canonical_digest(self._payload())
        provided = _sha256(self.receipt_sha256, "receipt_sha256")
        if expected != provided:
            raise ValueError("receipt_sha256 does not match governed fusion receipt content.")
        object.__setattr__(self, "receipt_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-governed-cross-profile-fusion-receipt/v1",
            "input_sha256": self.input_sha256,
            "policy_sha256": self.policy_sha256,
            "result_sha256": self.result_sha256,
            "mode": self.mode,
            "calibration_contract_sha256": self.calibration_contract_sha256,
            "profile_artifact_sha256s": self.profile_artifact_sha256s,
        }

    @classmethod
    def build(
        cls,
        *,
        input_sha256: str,
        policy_sha256: str,
        result: CrossProfileFusionResult,
    ) -> "GovernedCrossProfileFusionReceipt":
        if not isinstance(result, CrossProfileFusionResult):
            raise ValueError("result must be CrossProfileFusionResult.")
        payload = {
            "schema": "rigorousrag-governed-cross-profile-fusion-receipt/v1",
            "input_sha256": _sha256(input_sha256, "input_sha256"),
            "policy_sha256": _sha256(policy_sha256, "policy_sha256"),
            "result_sha256": _sha256(result.result_sha256, "result_sha256"),
            "mode": result.mode.value,
            "calibration_contract_sha256": result.calibration_contract_sha256,
            "profile_artifact_sha256s": tuple(sorted(result.profile_artifact_sha256s)),
        }
        return cls(
            input_sha256=payload["input_sha256"],
            policy_sha256=payload["policy_sha256"],
            result_sha256=payload["result_sha256"],
            mode=payload["mode"],
            calibration_contract_sha256=payload["calibration_contract_sha256"],
            profile_artifact_sha256s=payload["profile_artifact_sha256s"],
            receipt_sha256=_canonical_digest(payload),
        )


@dataclass(frozen=True)
class GovernedCrossProfileFusionRun:
    result: CrossProfileFusionResult
    receipt: GovernedCrossProfileFusionReceipt


def run_governed_cross_profile_fusion(
    ranked_lists: Sequence[ProfileRankedList],
    *,
    calibrators: Mapping[str, IsotonicCalibrationArtifact] | None = None,
    policy: CrossProfileFusionPolicy = CrossProfileFusionPolicy(),
) -> GovernedCrossProfileFusionRun:
    lists = tuple(ranked_lists)
    input_digest = ranked_input_sha256(lists)
    policy_digest = fusion_policy_sha256(policy)
    result = fuse_cross_profile_rankings(
        lists,
        calibrators=calibrators,
        policy=policy,
    )
    receipt = GovernedCrossProfileFusionReceipt.build(
        input_sha256=input_digest,
        policy_sha256=policy_digest,
        result=result,
    )
    return GovernedCrossProfileFusionRun(result=result, receipt=receipt)


__all__ = [
    "GovernedCrossProfileFusionReceipt",
    "GovernedCrossProfileFusionRun",
    "fusion_policy_sha256",
    "ranked_input_sha256",
    "run_governed_cross_profile_fusion",
]
