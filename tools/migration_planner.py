"""Profile-drift inventory and deterministic migration task planning."""

from __future__ import annotations

import hashlib
from typing import Any

from tools.embedding_registry import resolve_embedding_profile
from tools.generation_store import GenerationStore
from tools.migration_types import MigrationCandidate, exact_integer
from tools.security import normalize_owner_id

_MAX_INVENTORY = 10_000


def migration_task_id(candidate: MigrationCandidate) -> str:
    if not isinstance(candidate, MigrationCandidate):
        raise ValueError("candidate must be a MigrationCandidate.")
    return hashlib.sha256(
        (
            f"{candidate.owner_id}\x00{candidate.doc_id}\x00"
            f"{candidate.source_sequence}\x00"
            f"{candidate.source_profile_fingerprint}\x00"
            f"{candidate.target_profile_fingerprint}"
        ).encode("utf-8")
    ).hexdigest()


def inventory_profile_migrations(
    *,
    owner_id: str,
    target_profile: str,
    generations: GenerationStore,
    document_store: Any,
    limit: int = _MAX_INVENTORY,
) -> tuple[MigrationCandidate, ...]:
    """Classify current generations without exposing retained-source paths."""

    owner = normalize_owner_id(owner_id)
    count = exact_integer(limit, "limit", 1, _MAX_INVENTORY)
    target = resolve_embedding_profile(target_profile)
    records = generations.list_current(owner_id=owner, limit=count)
    candidates: list[MigrationCandidate] = []
    for record in records:
        retained = False
        eligible = False
        if record.state == "deleted":
            reason = "deleted"
        elif record.profile_fingerprint == target.fingerprint:
            reason = "already_target_profile"
        else:
            try:
                registry_record = document_store.get(
                    owner_id=owner,
                    doc_id=record.doc_id,
                    verify_visual=False,
                )
            except Exception:
                reason = "registry_inspection_failed"
            else:
                if isinstance(registry_record, dict):
                    retained = bool(
                        registry_record.get("source_retained")
                        and registry_record.get("source_path")
                    )
                if retained:
                    eligible = True
                    reason = "ready"
                else:
                    reason = "retained_source_unavailable"
        candidates.append(
            MigrationCandidate(
                owner_id=owner,
                doc_id=record.doc_id,
                source_sequence=record.sequence,
                source_profile_fingerprint=record.profile_fingerprint,
                target_profile_name=target.name,
                target_profile_fingerprint=target.fingerprint,
                retained_source=retained,
                eligible=eligible,
                reason=reason,
            )
        )
    return tuple(candidates)


__all__ = ["inventory_profile_migrations", "migration_task_id"]
