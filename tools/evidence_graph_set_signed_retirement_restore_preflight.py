"""Read-only restore preflight for signed retirement journal snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

from tools.evidence_graph_set_signed_retirement_contracts import _integer, _timestamp
from tools.evidence_graph_set_signed_retirement_snapshot import SignedRetirementSnapshot

_DISPOSITIONS = frozenset(
    {
        "empty_snapshot_no_restore",
        "empty_target_restore_candidate",
        "already_restored_exactly",
        "target_nonterminal_refusal",
        "immutable_collision_refusal",
        "state_collision_refusal",
        "partial_restore_refusal",
        "target_additional_history_refusal",
    }
)
_NONTERMINAL = frozenset({"planned", "running", "failed"})


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SignedRetirementRestoreComparison:
    retirement_id: str
    status: str
    snapshot_state: str | None
    target_state: str | None
    snapshot_phase: str | None
    target_phase: str | None


@dataclass(frozen=True)
class SignedRetirementRestorePreflight:
    owner_id: str
    snapshot_digest: str
    generated_at: float
    snapshot_record_count: int
    target_record_count: int
    exact_count: int
    missing_count: int
    additional_count: int
    immutable_collision_count: int
    state_collision_count: int
    nonterminal_target_count: int
    disposition: str
    eligible_for_future_restore: bool
    comparisons: tuple[SignedRetirementRestoreComparison, ...]
    report_digest: str
    target_mutation_performed: bool = False
    restore_performed: bool = False
    source_text_returned: bool = False

    def __post_init__(self) -> None:
        if self.disposition not in _DISPOSITIONS:
            raise ValueError("restore preflight disposition is unsupported.")
        if not isinstance(self.eligible_for_future_restore, bool):
            raise ValueError("eligible_for_future_restore must be boolean.")
        if self.eligible_for_future_restore != (
            self.disposition == "empty_target_restore_candidate"
        ):
            raise ValueError("restore eligibility differs from disposition.")
        if any(
            value is not False
            for value in (
                self.target_mutation_performed,
                self.restore_performed,
                self.source_text_returned,
            )
        ):
            raise ValueError("restore preflight safety flags must be false.")


def _immutable(value: Any) -> str:
    return value.immutable_digest


def _mutable(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    for key in (
        "retirement_id",
        "owner_id",
        "publication_operation_id",
        "graph_set_key",
        "signed_candidate_set_id",
        "signed_candidate_set_digest",
        "authorization_candidate_set_id",
        "signed_authority_digest",
        "schema_version",
    ):
        payload.pop(key, None)
    return payload


def preflight_signed_retirement_snapshot_restore(
    *,
    snapshot: SignedRetirementSnapshot,
    target_journal: Any,
    now: float | None = None,
    limit: int = 10_000,
) -> SignedRetirementRestorePreflight:
    if not isinstance(snapshot, SignedRetirementSnapshot):
        raise ValueError("snapshot must be SignedRetirementSnapshot.")
    if not callable(getattr(target_journal, "list", None)):
        raise ValueError("target journal lacks the required read boundary.")
    timestamp = _timestamp(time.time() if now is None else now, "now")
    count = _integer(limit, "limit", 1, 10_000)
    target_values = tuple(
        target_journal.list(owner_id=snapshot.owner_id, limit=count)
    )
    if len(target_values) >= count:
        raise RuntimeError("restore preflight reached the bounded result limit.")
    target_by_id = {value.retirement_id: value for value in target_values}
    if len(target_by_id) != len(target_values):
        raise RuntimeError("target journal returned duplicate retirement IDs.")
    snapshot_by_id = {value.retirement_id: value for value in snapshot.records}

    comparisons: list[SignedRetirementRestoreComparison] = []
    exact = missing = additional = immutable_collisions = state_collisions = 0
    nonterminal = sum(value.state in _NONTERMINAL for value in target_values)

    for retirement_id in sorted(set(snapshot_by_id) | set(target_by_id)):
        source = snapshot_by_id.get(retirement_id)
        target = target_by_id.get(retirement_id)
        if source is None:
            status = "additional_target_record"
            additional += 1
        elif target is None:
            status = "missing_target_record"
            missing += 1
        elif _immutable(source) != _immutable(target):
            status = "immutable_collision"
            immutable_collisions += 1
        elif _mutable(source) != _mutable(target):
            status = "state_collision"
            state_collisions += 1
        else:
            status = "exact_match"
            exact += 1
        comparisons.append(
            SignedRetirementRestoreComparison(
                retirement_id=retirement_id,
                status=status,
                snapshot_state=None if source is None else source.state,
                target_state=None if target is None else target.state,
                snapshot_phase=None if source is None else source.phase,
                target_phase=None if target is None else target.phase,
            )
        )

    if snapshot.record_count == 0:
        disposition = "empty_snapshot_no_restore"
    elif nonterminal:
        disposition = "target_nonterminal_refusal"
    elif immutable_collisions:
        disposition = "immutable_collision_refusal"
    elif state_collisions:
        disposition = "state_collision_refusal"
    elif not target_values:
        disposition = "empty_target_restore_candidate"
    elif missing:
        disposition = "partial_restore_refusal"
    elif additional:
        disposition = "target_additional_history_refusal"
    else:
        disposition = "already_restored_exactly"

    rendered = tuple(comparisons)
    stable = {
        "scope": "rigorousrag-signed-retirement-restore-preflight-v1",
        "owner_id": snapshot.owner_id,
        "snapshot_digest": snapshot.snapshot_digest,
        "generated_at": timestamp,
        "snapshot_record_count": snapshot.record_count,
        "target_record_count": len(target_values),
        "exact_count": exact,
        "missing_count": missing,
        "additional_count": additional,
        "immutable_collision_count": immutable_collisions,
        "state_collision_count": state_collisions,
        "nonterminal_target_count": nonterminal,
        "disposition": disposition,
        "eligible_for_future_restore": (
            disposition == "empty_target_restore_candidate"
        ),
        "comparisons": [asdict(value) for value in rendered],
    }
    return SignedRetirementRestorePreflight(
        owner_id=snapshot.owner_id,
        snapshot_digest=snapshot.snapshot_digest,
        generated_at=timestamp,
        snapshot_record_count=snapshot.record_count,
        target_record_count=len(target_values),
        exact_count=exact,
        missing_count=missing,
        additional_count=additional,
        immutable_collision_count=immutable_collisions,
        state_collision_count=state_collisions,
        nonterminal_target_count=nonterminal,
        disposition=disposition,
        eligible_for_future_restore=(
            disposition == "empty_target_restore_candidate"
        ),
        comparisons=rendered,
        report_digest=_canonical_digest(stable),
    )


__all__ = [
    "SignedRetirementRestoreComparison",
    "SignedRetirementRestorePreflight",
    "preflight_signed_retirement_snapshot_restore",
]
