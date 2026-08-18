"""Operator-authorized entrypoints for older high-risk governance mutations.

These wrappers add repository-owned RBAC/ABAC authority without weakening the existing
independent provenance and safety contracts.  Legal-hold actors remain cryptographically/
process bound ReviewActorBinding values; KMS/HSM operations remain delegated to the real
provider.  Operator authorization is an additional exact action/resource gate.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from security.key_management import KeyManagementProvider, KeyReference
from security.operator_authorization import (
    OperatorAuthorizationDecision,
    OperatorAuthorizationRequest,
    assert_operator_authorization,
)
from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_hold_boundary import (
    GovernedSignedRetirementRestoreHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_holds import (
    SignedRetirementRestoreHold,
    deterministic_restore_hold_id,
)


def _assert_exact_authorization(
    decision: OperatorAuthorizationDecision,
    request: OperatorAuthorizationRequest,
    *,
    policy_sha256: str,
    owner_id: str,
    domain_id: str,
    action: str,
    resource_type: str,
    resource_sha256: str,
    now: float,
) -> None:
    assert_operator_authorization(decision, request, policy_sha256=policy_sha256, now=now)
    if request.owner_id != owner_id or request.domain_id != domain_id:
        raise RuntimeError("operator authorization scope differs from sensitive mutation")
    if request.action != action or request.resource_type != resource_type:
        raise RuntimeError("operator authorization action/resource type differs from sensitive mutation")
    if request.resource_sha256 != resource_sha256:
        raise RuntimeError("operator authorization is bound to a different sensitive resource")


def place_restore_legal_hold_authorized(
    store: GovernedSignedRetirementRestoreHoldStore,
    *,
    owner_id: str,
    domain_id: str,
    restore_id: str,
    hold_key: str,
    reason_code: str,
    actor: ReviewActorBinding,
    restore_journal: Any,
    authorization: OperatorAuthorizationDecision,
    authorization_request: OperatorAuthorizationRequest,
    operator_policy_sha256: str,
    now: float,
) -> SignedRetirementRestoreHold:
    """Place a restore hold only under exact operator and review-actor authority."""

    if not isinstance(store, GovernedSignedRetirementRestoreHoldStore):
        raise ValueError("store must be GovernedSignedRetirementRestoreHoldStore")
    hold_id = deterministic_restore_hold_id(
        owner_id=owner_id,
        restore_id=restore_id,
        hold_key=hold_key,
    )
    _assert_exact_authorization(
        authorization,
        authorization_request,
        policy_sha256=operator_policy_sha256,
        owner_id=owner_id,
        domain_id=domain_id,
        action="legal_hold.place",
        resource_type="restore_legal_hold",
        resource_sha256=hold_id,
        now=now,
    )
    value = store.place(
        owner_id=owner_id,
        restore_id=restore_id,
        hold_key=hold_key,
        reason_code=reason_code,
        actor=actor,
        restore_journal=restore_journal,
        now=now,
    )
    if value.hold_id != hold_id or value.owner_id != owner_id:
        raise RuntimeError("legal-hold store returned a different authorized hold scope")
    return value


def release_restore_legal_hold_authorized(
    store: GovernedSignedRetirementRestoreHoldStore,
    hold_id: str,
    *,
    owner_id: str,
    domain_id: str,
    actor: ReviewActorBinding,
    authorization: OperatorAuthorizationDecision,
    authorization_request: OperatorAuthorizationRequest,
    operator_policy_sha256: str,
    now: float,
) -> SignedRetirementRestoreHold:
    """Release an existing restore hold only under authorization bound to its hold id."""

    if not isinstance(store, GovernedSignedRetirementRestoreHoldStore):
        raise ValueError("store must be GovernedSignedRetirementRestoreHoldStore")
    current = store.get(hold_id)
    if current.owner_id != owner_id:
        raise RuntimeError("restore hold belongs to a different owner")
    _assert_exact_authorization(
        authorization,
        authorization_request,
        policy_sha256=operator_policy_sha256,
        owner_id=owner_id,
        domain_id=domain_id,
        action="legal_hold.release",
        resource_type="restore_legal_hold",
        resource_sha256=current.hold_id,
        now=now,
    )
    value = store.release(
        current.hold_id,
        owner_id=owner_id,
        confirm_hold_id=current.hold_id,
        actor=actor,
        now=now,
    )
    if value.hold_id != current.hold_id or value.owner_id != owner_id or value.status != "released":
        raise RuntimeError("legal-hold release returned an inconsistent record")
    return value


def schedule_key_destruction_authorized(
    provider: KeyManagementProvider,
    key: KeyReference,
    *,
    owner_id: str,
    domain_id: str,
    not_before: datetime,
    authorization: OperatorAuthorizationDecision,
    authorization_request: OperatorAuthorizationRequest,
    operator_policy_sha256: str,
    now: float,
) -> KeyReference:
    """Schedule external KMS/HSM destruction only for the exactly authorized key version."""

    if not isinstance(key, KeyReference):
        raise ValueError("key must be KeyReference")
    _assert_exact_authorization(
        authorization,
        authorization_request,
        policy_sha256=operator_policy_sha256,
        owner_id=owner_id,
        domain_id=domain_id,
        action="key.schedule_destruction",
        resource_type="key_reference",
        resource_sha256=key.digest,
        now=now,
    )
    value = provider.schedule_key_destruction(key, not_before=not_before)
    if not isinstance(value, KeyReference):
        raise RuntimeError("key-management provider returned an invalid key reference")
    if value.provider != key.provider or value.key_id != key.key_id or value.key_version != key.key_version:
        raise RuntimeError("key-management provider changed key identity during destruction scheduling")
    return value


def cancel_key_destruction_authorized(
    provider: KeyManagementProvider,
    key: KeyReference,
    *,
    owner_id: str,
    domain_id: str,
    authorization: OperatorAuthorizationDecision,
    authorization_request: OperatorAuthorizationRequest,
    operator_policy_sha256: str,
    now: float,
) -> KeyReference:
    """Cancel external KMS/HSM destruction only for the exactly authorized key version."""

    if not isinstance(key, KeyReference):
        raise ValueError("key must be KeyReference")
    _assert_exact_authorization(
        authorization,
        authorization_request,
        policy_sha256=operator_policy_sha256,
        owner_id=owner_id,
        domain_id=domain_id,
        action="key.cancel_destruction",
        resource_type="key_reference",
        resource_sha256=key.digest,
        now=now,
    )
    value = provider.cancel_key_destruction(key)
    if not isinstance(value, KeyReference):
        raise RuntimeError("key-management provider returned an invalid key reference")
    if value.provider != key.provider or value.key_id != key.key_id or value.key_version != key.key_version:
        raise RuntimeError("key-management provider changed key identity during destruction cancellation")
    return value


__all__ = [
    "cancel_key_destruction_authorized",
    "place_restore_legal_hold_authorized",
    "release_restore_legal_hold_authorized",
    "schedule_key_destruction_authorized",
]
