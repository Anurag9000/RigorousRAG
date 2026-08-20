"""Cohort-safe request context provider for recorded dynamic-RAG training episodes.

A provider *contract* must identify the context-construction policy, not one request instance.
The request itself is separately bound by ``request_sha256`` in every runtime snapshot/episode
receipt.  This provider therefore keeps one stable contract across a multi-query cohort while
exposing its exact request SHA for the production recorder to cross-check against the runtime.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from orchestration.dynamic_rag_runtime import DynamicRuntimeSnapshot


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: object, label: str, maximum: int = 10_000_000) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{label} is invalid")
    return value


@dataclass(frozen=True)
class CohortSafeRequestTrainingContextProvider:
    request_text: str
    include_retrieval_scores: bool = True
    include_verification: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_text", _text(self.request_text, "request_text"))
        if not isinstance(self.include_retrieval_scores, bool) or not isinstance(self.include_verification, bool):
            raise ValueError("context inclusion flags must be boolean")

    @property
    def request_sha256(self) -> str:
        return hashlib.sha256(self.request_text.encode("utf-8")).hexdigest()

    @property
    def contract_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-cohort-safe-dynamic-training-context-provider/v1",
            "include_retrieval_scores": self.include_retrieval_scores,
            "include_verification": self.include_verification,
            "template": "request+generated+evidence_fingerprints+optional_verification",
            "request_identity": "bound_separately_by_request_sha256",
        })

    def model_text(self, snapshot: DynamicRuntimeSnapshot) -> str:
        if not isinstance(snapshot, DynamicRuntimeSnapshot):
            raise ValueError("snapshot must be DynamicRuntimeSnapshot")
        if snapshot.request_sha256 != self.request_sha256:
            raise ValueError("context-provider request differs from runtime snapshot request")
        lines = ["Request:", self.request_text, "", "Generated:", snapshot.generated_text, "", "Evidence:"]
        for item in snapshot.evidence:
            record = f"{item.evidence_id}\tsha256={item.evidence_sha256}\tsource_group={item.source_group_sha256}"
            if self.include_retrieval_scores:
                record += f"\tscore={format(item.retrieval_score, '.17g')}"
            lines.append(record)
        if self.include_verification and snapshot.verification is not None:
            lines.extend([
                "",
                "Verification:",
                f"support={format(snapshot.verification.support_score, '.17g')}",
                f"contradiction={format(snapshot.verification.contradiction_score, '.17g')}",
                f"verifier_sha256={snapshot.verification.verifier_sha256}",
            ])
        return _text("\n".join(lines), "dynamic training context")


__all__ = ["CohortSafeRequestTrainingContextProvider"]
