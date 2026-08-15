"""Backend-neutral blue/green migration and compensating cutover coordination.

The coordinator generalizes existing local same-dimension migration logic to arbitrary
participants (vector, sparse, manifest, graph, object/registry) and cross-dimension index
changes. It is a protocol/orchestration layer: participant implementations remain
responsible for durable storage, fencing and atomicity guarantees they advertise.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence


def _text(value: Any, label: str, maximum: int = 500, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class MigrationTarget:
    target_id: str
    source_generation_sha256: str
    target_profile_sha256: str
    vector_dimension: int
    index_schema_version: str
    content_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id", 256))
        for name in ("source_generation_sha256", "target_profile_sha256", "content_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if isinstance(self.vector_dimension, bool) or not isinstance(self.vector_dimension, int) or not 1 <= self.vector_dimension <= 1_000_000:
            raise ValueError("vector_dimension is invalid")
        object.__setattr__(self, "index_schema_version", _text(self.index_schema_version, "index_schema_version", 100))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class PreparedParticipant:
    participant_id: str
    preparation_sha256: str
    rollback_sha256: str
    target_sha256: str
    fencing_token: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "participant_id", _text(self.participant_id, "participant_id", 128))
        for name in ("preparation_sha256", "rollback_sha256", "target_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if isinstance(self.fencing_token, bool) or not isinstance(self.fencing_token, int) or self.fencing_token < 1:
            raise ValueError("fencing_token is invalid")


class MigrationParticipant(Protocol):
    @property
    def participant_id(self) -> str: ...
    def prepare(self, target: MigrationTarget, *, operation_id: str, fencing_token: int) -> PreparedParticipant: ...
    def publish(self, prepared: PreparedParticipant, *, operation_id: str, fencing_token: int) -> str: ...
    def validate(self, prepared: PreparedParticipant, *, operation_id: str) -> str: ...
    def rollback(self, prepared: PreparedParticipant, *, operation_id: str, fencing_token: int) -> str: ...
    def finalize(self, prepared: PreparedParticipant, *, operation_id: str) -> None: ...


@dataclass(frozen=True)
class CutoverPlan:
    operation_id: str
    target: MigrationTarget
    participant_ids: tuple[str, ...]
    fencing_token: int
    canary_fraction: float
    created_at: float
    plan_sha256: str


def make_cutover_plan(*, operation_id: str, target: MigrationTarget, participant_ids: Sequence[str], fencing_token: int, canary_fraction: float = 1.0) -> CutoverPlan:
    op = _text(operation_id, "operation_id", 256)
    ids = tuple(dict.fromkeys(_text(item, "participant_id", 128) for item in participant_ids))
    if not ids or len(ids) > 32:
        raise ValueError("participant_ids are invalid")
    if isinstance(fencing_token, bool) or not isinstance(fencing_token, int) or fencing_token < 1:
        raise ValueError("fencing_token is invalid")
    fraction = float(canary_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("canary_fraction must be in (0,1]")
    created = time.time()
    payload = {"operation_id": op, "target": asdict(target), "participant_ids": ids, "fencing_token": fencing_token, "canary_fraction": fraction, "created_at": created}
    return CutoverPlan(op, target, ids, fencing_token, fraction, created, hashlib.sha256(_canonical(payload)).hexdigest())


@dataclass(frozen=True)
class ParticipantReceipt:
    participant_id: str
    prepare_sha256: str
    publish_sha256: str
    validation_sha256: str
    rolled_back: bool = False
    rollback_sha256: str = ""


@dataclass(frozen=True)
class CutoverReceipt:
    plan_sha256: str
    status: str
    participants: tuple[ParticipantReceipt, ...]
    failed_participant_id: str = ""
    completed_at: float = 0.0


def execute_cutover(plan: CutoverPlan, participants: Mapping[str, MigrationParticipant]) -> CutoverReceipt:
    prepared: list[tuple[MigrationParticipant, PreparedParticipant]] = []
    published: list[tuple[MigrationParticipant, PreparedParticipant, str]] = []
    receipts: list[ParticipantReceipt] = []
    failed_id = ""
    try:
        for participant_id in plan.participant_ids:
            participant = participants.get(participant_id)
            if participant is None or _text(participant.participant_id, "participant_id", 128) != participant_id:
                raise RuntimeError(f"migration participant unavailable: {participant_id}")
            item = participant.prepare(plan.target, operation_id=plan.operation_id, fencing_token=plan.fencing_token)
            if item.participant_id != participant_id:
                raise RuntimeError("migration participant returned mismatched identity")
            prepared.append((participant, item))
        for participant, item in prepared:
            failed_id = item.participant_id
            published_sha = _sha(participant.publish(item, operation_id=plan.operation_id, fencing_token=plan.fencing_token), "publish digest")
            published.append((participant, item, published_sha))
        for participant, item, published_sha in published:
            failed_id = item.participant_id
            validated_sha = _sha(participant.validate(item, operation_id=plan.operation_id), "validation digest")
            if validated_sha != item.target_sha256:
                raise RuntimeError("published migration participant failed exact target validation")
            receipts.append(ParticipantReceipt(item.participant_id, item.preparation_sha256, published_sha, validated_sha))
        for participant, item in prepared:
            participant.finalize(item, operation_id=plan.operation_id)
        return CutoverReceipt(plan.plan_sha256, "committed", tuple(receipts), "", time.time())
    except Exception:
        rolled: dict[str, ParticipantReceipt] = {item.participant_id: receipt for item, receipt in []}
        del rolled
        rollback_receipts: list[ParticipantReceipt] = []
        published_map = {item.participant_id: publish_sha for _, item, publish_sha in published}
        for participant, item in reversed(prepared):
            publish_sha = published_map.get(item.participant_id, "")
            try:
                rollback_sha = _sha(participant.rollback(item, operation_id=plan.operation_id, fencing_token=plan.fencing_token), "rollback digest")
                if rollback_sha != item.rollback_sha256:
                    raise RuntimeError("rollback identity validation failed")
                rollback_receipts.append(ParticipantReceipt(item.participant_id, item.preparation_sha256, publish_sha, "", True, rollback_sha))
            except Exception:
                rollback_receipts.append(ParticipantReceipt(item.participant_id, item.preparation_sha256, publish_sha, "", True, ""))
        rollback_receipts.reverse()
        return CutoverReceipt(plan.plan_sha256, "rolled_back", tuple(rollback_receipts), failed_id, time.time())


__all__ = [
    "CutoverPlan", "CutoverReceipt", "MigrationParticipant", "MigrationTarget",
    "ParticipantReceipt", "PreparedParticipant", "execute_cutover", "make_cutover_plan",
]
