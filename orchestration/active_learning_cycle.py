"""One-shot resumable coordinator for active-learning acquisition cycles.

A scheduler may call this function periodically, but this module owns no background
thread. Candidate discovery is injected through a paged provider. The cycle deduplicates
against previously materialized cases, selects a deterministic budgeted batch, and
materializes selected cases through the existing adjudication store.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol

from evaluation.active_learning import ActiveLearningBatch, ActiveLearningCandidate, ActiveLearningPolicy, select_active_learning_batch
from evaluation.expert_adjudication import ExpertAdjudicationStore
from orchestration.active_learning_adjudication import ActiveLearningMaterializationReceipt, ActiveLearningRoute, SQLiteActiveLearningJournal, materialize_active_learning_batch


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _text(value: str, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _cursor_digest(value: str | None) -> str:
    return _digest({"schema": "rigorousrag-active-learning-cursor/v1", "cursor": value})


@dataclass(frozen=True)
class ActiveLearningCandidatePage:
    source_snapshot_sha256: str
    candidates: tuple[ActiveLearningCandidate, ...]
    next_cursor: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_snapshot_sha256", _sha(self.source_snapshot_sha256, "source_snapshot_sha256"))
        rows = tuple(self.candidates)
        if len(rows) > 1_000_000 or any(not isinstance(row, ActiveLearningCandidate) for row in rows):
            raise ValueError("candidate page is invalid or exceeds the page limit")
        if len({row.candidate_sha256 for row in rows}) != len(rows):
            raise ValueError("candidate page contains duplicate candidate identities")
        object.__setattr__(self, "candidates", rows)
        if self.next_cursor is not None:
            object.__setattr__(self, "next_cursor", _text(self.next_cursor, "next_cursor", 4000))

    @property
    def page_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-active-learning-candidate-page/v1",
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "candidate_sha256s": tuple(row.candidate_sha256 for row in self.candidates),
            "next_cursor_sha256": _cursor_digest(self.next_cursor),
        })


class ActiveLearningCandidateProvider(Protocol):
    def fetch_candidates(self, *, owner_id: str, cursor: str | None, limit: int) -> ActiveLearningCandidatePage: ...


@dataclass(frozen=True)
class ActiveLearningCycleSpec:
    owner_id: str
    candidate_source_sha256: str
    policy: ActiveLearningPolicy
    page_limit: int = 10_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        object.__setattr__(self, "candidate_source_sha256", _sha(self.candidate_source_sha256, "candidate_source_sha256"))
        if not isinstance(self.policy, ActiveLearningPolicy):
            raise ValueError("policy must be ActiveLearningPolicy")
        if isinstance(self.page_limit, bool) or not isinstance(self.page_limit, int) or not 1 <= self.page_limit <= 1_000_000:
            raise ValueError("page_limit must be in [1, 1000000]")

    @property
    def spec_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-active-learning-cycle-spec/v1",
            "owner_id": self.owner_id,
            "candidate_source_sha256": self.candidate_source_sha256,
            "policy_sha256": self.policy.policy_sha256,
            "page_limit": self.page_limit,
        })


@dataclass(frozen=True)
class ActiveLearningCycleReceipt:
    spec_sha256: str
    source_snapshot_sha256: str
    page_sha256: str
    input_cursor_sha256: str
    next_cursor_sha256: str
    batch_sha256: str | None
    materialization_receipt_sha256: str | None
    candidate_count: int
    selected_count: int
    materialized_count: int
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("spec_sha256", "source_snapshot_sha256", "page_sha256", "input_cursor_sha256", "next_cursor_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        for name in ("batch_sha256", "materialization_receipt_sha256"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sha(value, name))
        for name in ("candidate_count", "selected_count", "materialized_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.selected_count > self.candidate_count or self.materialized_count != self.selected_count:
            raise ValueError("active-learning cycle counts are inconsistent")
        if self.candidate_count == 0 and (self.batch_sha256 is not None or self.materialization_receipt_sha256 is not None):
            raise ValueError("empty candidate page may not claim batch/materialization receipts")
        if self.candidate_count > 0 and (self.batch_sha256 is None or self.materialization_receipt_sha256 is None):
            raise ValueError("non-empty candidate page requires batch/materialization receipts")
        expected = _digest(self._payload())
        provided = _sha(self.receipt_sha256, "receipt_sha256")
        if expected != provided:
            raise ValueError("receipt_sha256 does not match active-learning cycle content")
        object.__setattr__(self, "receipt_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-active-learning-cycle/v1",
            "spec_sha256": self.spec_sha256,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "page_sha256": self.page_sha256,
            "input_cursor_sha256": self.input_cursor_sha256,
            "next_cursor_sha256": self.next_cursor_sha256,
            "batch_sha256": self.batch_sha256,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "materialized_count": self.materialized_count,
        }


@dataclass(frozen=True)
class ActiveLearningCycleResult:
    next_cursor: str | None
    batch: ActiveLearningBatch | None
    materialization: ActiveLearningMaterializationReceipt | None
    receipt: ActiveLearningCycleReceipt


def run_active_learning_cycle(
    spec: ActiveLearningCycleSpec,
    *,
    provider: ActiveLearningCandidateProvider,
    routes: Mapping[str, ActiveLearningRoute],
    adjudication_store: ExpertAdjudicationStore,
    journal: SQLiteActiveLearningJournal,
    cursor: str | None = None,
    now: float,
) -> ActiveLearningCycleResult:
    page = provider.fetch_candidates(owner_id=spec.owner_id, cursor=cursor, limit=spec.page_limit)
    if not isinstance(page, ActiveLearningCandidatePage):
        raise ValueError("candidate provider returned an invalid page")
    if any(row.owner_id != spec.owner_id for row in page.candidates):
        raise ValueError("candidate provider returned another owner's candidate")
    if not page.candidates:
        payload = {
            "schema": "rigorousrag-active-learning-cycle/v1",
            "spec_sha256": spec.spec_sha256,
            "source_snapshot_sha256": page.source_snapshot_sha256,
            "page_sha256": page.page_sha256,
            "input_cursor_sha256": _cursor_digest(cursor),
            "next_cursor_sha256": _cursor_digest(page.next_cursor),
            "batch_sha256": None,
            "materialization_receipt_sha256": None,
            "candidate_count": 0,
            "selected_count": 0,
            "materialized_count": 0,
        }
        receipt = ActiveLearningCycleReceipt(**payload, receipt_sha256=_digest(payload))
        return ActiveLearningCycleResult(page.next_cursor, None, None, receipt)

    blocked = journal.blocked_item_keys(owner_id=spec.owner_id)
    batch = select_active_learning_batch(page.candidates, policy=spec.policy, blocked_item_keys=blocked)
    materialization = materialize_active_learning_batch(
        batch,
        page.candidates,
        routes=routes,
        adjudication_store=adjudication_store,
        journal=journal,
        now=now,
    )
    payload = {
        "schema": "rigorousrag-active-learning-cycle/v1",
        "spec_sha256": spec.spec_sha256,
        "source_snapshot_sha256": page.source_snapshot_sha256,
        "page_sha256": page.page_sha256,
        "input_cursor_sha256": _cursor_digest(cursor),
        "next_cursor_sha256": _cursor_digest(page.next_cursor),
        "batch_sha256": batch.batch_sha256,
        "materialization_receipt_sha256": materialization.receipt_sha256,
        "candidate_count": len(page.candidates),
        "selected_count": len(batch.selected),
        "materialized_count": len(materialization.cases),
    }
    receipt = ActiveLearningCycleReceipt(**payload, receipt_sha256=_digest(payload))
    return ActiveLearningCycleResult(page.next_cursor, batch, materialization, receipt)


__all__ = [
    "ActiveLearningCandidatePage",
    "ActiveLearningCandidateProvider",
    "ActiveLearningCycleReceipt",
    "ActiveLearningCycleResult",
    "ActiveLearningCycleSpec",
    "run_active_learning_cycle",
]
