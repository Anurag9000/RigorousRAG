"""Reconcile durable source-trust head activations into dependency invalidation.

The source-trust database is the transactional outbox: revision, active head and pending
activation are committed together. This module consumes pending activations and applies
idempotent dependency invalidation. It stores no evidence/query text and starts no
background worker; API or operator code invokes bounded reconciliation explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.dependency_invalidation import DependencyInvalidationStore, DependencyRef
from tools.security import normalize_owner_id
from tools.source_trust_store import SourceTrustActivation, SourceTrustStore

_REASON = "reviewed source-trust features changed; evidence admissibility requires re-evaluation"


@dataclass(frozen=True)
class SourceTrustActivationOutcome:
    activation_id: str
    source_id: str
    revision_id: str
    success: bool
    affected_artifacts: int = 0
    recompute_tasks: int = 0
    error_type: str = ""


@dataclass(frozen=True)
class SourceTrustReconciliationSummary:
    owner_id: str
    attempted: int
    completed: int
    failed: int
    affected_artifacts: int
    recompute_tasks: int
    outcomes: tuple[SourceTrustActivationOutcome, ...]


def _roots(activation: SourceTrustActivation) -> tuple[DependencyRef, ...]:
    values = [
        # Captures every source considered by the final admissibility gate, including
        # sources rejected from the published answer.
        DependencyRef("source_trust_subject", activation.source_id),
        # Covers final citations and artifacts created before source_trust_subject existed.
        DependencyRef("source", activation.source_id),
    ]
    if activation.previous_revision_id:
        values.append(DependencyRef("source_trust_revision", activation.previous_revision_id))
    return tuple(values)


def reconcile_activation(
    trust_store: SourceTrustStore,
    invalidations: DependencyInvalidationStore,
    activation: SourceTrustActivation,
) -> SourceTrustActivationOutcome:
    if not isinstance(trust_store, SourceTrustStore):
        raise TypeError("trust_store must be SourceTrustStore")
    if not isinstance(invalidations, DependencyInvalidationStore):
        raise TypeError("invalidations must be DependencyInvalidationStore")
    if not isinstance(activation, SourceTrustActivation):
        raise TypeError("activation must be SourceTrustActivation")
    if not activation.pending:
        return SourceTrustActivationOutcome(
            activation.activation_id,
            activation.source_id,
            activation.revision_id,
            True,
        )

    affected: dict[str, DependencyRef] = {}
    task_ids: set[str] = set()
    try:
        for root in _roots(activation):
            impact = invalidations.invalidate(
                activation.owner_id,
                root=root,
                reason=_REASON,
                event_type="source_trust_review_changed",
                replacement_id=activation.revision_id,
                event_sha256=activation.activation_id,
            )
            for artifact in impact.affected:
                affected[artifact.key] = artifact
            task_ids.update(task.task_id for task in impact.recompute_tasks)
        trust_store.mark_activation_completed(activation.owner_id, activation.activation_id)
        return SourceTrustActivationOutcome(
            activation.activation_id,
            activation.source_id,
            activation.revision_id,
            True,
            len(affected),
            len(task_ids),
        )
    except Exception as exc:
        error_type = type(exc).__name__[:200]
        try:
            trust_store.mark_activation_failed(
                activation.owner_id,
                activation.activation_id,
                error_type,
            )
        except Exception:
            # Preserve the original failure. A missing failure annotation must not hide
            # the fact that the activation itself remains pending and retryable.
            pass
        return SourceTrustActivationOutcome(
            activation.activation_id,
            activation.source_id,
            activation.revision_id,
            False,
            len(affected),
            len(task_ids),
            error_type,
        )


def reconcile_source_trust_activations(
    trust_store: SourceTrustStore,
    invalidations: DependencyInvalidationStore,
    owner_id: str,
    *,
    source_id: str | None = None,
    limit: int = 1000,
    stop_on_error: bool = False,
) -> SourceTrustReconciliationSummary:
    owner = normalize_owner_id(owner_id)
    if not 1 <= limit <= 10_000:
        raise ValueError("limit is invalid")
    activations = trust_store.pending_activations(
        owner,
        source_id=source_id,
        limit=limit,
    )
    outcomes: list[SourceTrustActivationOutcome] = []
    affected = 0
    tasks = 0
    for activation in activations:
        outcome = reconcile_activation(trust_store, invalidations, activation)
        outcomes.append(outcome)
        affected += outcome.affected_artifacts
        tasks += outcome.recompute_tasks
        if not outcome.success and stop_on_error:
            break
    completed = sum(1 for item in outcomes if item.success)
    return SourceTrustReconciliationSummary(
        owner_id=owner,
        attempted=len(outcomes),
        completed=completed,
        failed=len(outcomes) - completed,
        affected_artifacts=affected,
        recompute_tasks=tasks,
        outcomes=tuple(outcomes),
    )


__all__ = [
    "SourceTrustActivationOutcome",
    "SourceTrustReconciliationSummary",
    "reconcile_activation",
    "reconcile_source_trust_activations",
]
