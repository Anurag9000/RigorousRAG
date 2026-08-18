"""Authorized entrypoints for high-risk runtime, residency and region mutations.

These wrappers compose short-lived operator authorization with the underlying monotonic
CAS/fencing stores. Lower-level stores remain reusable orchestration primitives; production
control-plane code should enter through these functions so every mutation is bound to the
exact owner/domain/action/resource authorization request.
"""

from __future__ import annotations

from orchestration.multi_region_authority import RegionAuthorityDecision, RegionAuthorityRecord, SQLiteRegionAuthorityStore
from orchestration.runtime_stack_authority import (
    RuntimeAuthorityRecord,
    RuntimePromotionDecision,
    RuntimeRollbackRequest,
    RuntimeStackArtifact,
    SQLiteRuntimeStackAuthorityStore,
)
from security.data_residency import DataResidencyPolicy, ResidencyPolicyRecord, SQLiteResidencyPolicyStore
from security.operator_authorization import (
    OperatorAuthorizationDecision,
    OperatorAuthorizationRequest,
    assert_operator_authorization,
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
        raise RuntimeError("operator authorization scope differs from the control-plane mutation")
    if request.action != action or request.resource_type != resource_type or request.resource_sha256 != resource_sha256:
        raise RuntimeError("operator authorization is bound to a different control-plane action/resource")


def promote_runtime_stack_authorized(
    store: SQLiteRuntimeStackAuthorityStore,
    *,
    owner_id: str,
    service_id: str,
    domain_id: str,
    stack: RuntimeStackArtifact,
    promotion_decision: RuntimePromotionDecision,
    expected_authority_revision: int | None,
    authorization: OperatorAuthorizationDecision,
    authorization_request: OperatorAuthorizationRequest,
    operator_policy_sha256: str,
    now: float,
) -> RuntimeAuthorityRecord:
    _assert_exact_authorization(
        authorization,
        authorization_request,
        policy_sha256=operator_policy_sha256,
        owner_id=owner_id,
        domain_id=domain_id,
        action="runtime.promote",
        resource_type="runtime_stack",
        resource_sha256=stack.stack_sha256,
        now=now,
    )
    return store.promote(
        owner_id=owner_id,
        service_id=service_id,
        domain_id=domain_id,
        stack=stack,
        decision=promotion_decision,
        expected_authority_revision=expected_authority_revision,
        now=now,
    )


def rollback_runtime_stack_authorized(
    store: SQLiteRuntimeStackAuthorityStore,
    request: RuntimeRollbackRequest,
    *,
    expected_authority_revision: int,
    current_compatibility_sha256: str,
    authorization: OperatorAuthorizationDecision,
    authorization_request: OperatorAuthorizationRequest,
    operator_policy_sha256: str,
    now: float,
) -> RuntimeAuthorityRecord:
    _assert_exact_authorization(
        authorization,
        authorization_request,
        policy_sha256=operator_policy_sha256,
        owner_id=request.owner_id,
        domain_id=request.domain_id,
        action="runtime.rollback",
        resource_type="runtime_stack",
        resource_sha256=request.request_sha256,
        now=now,
    )
    return store.rollback(
        request,
        expected_authority_revision=expected_authority_revision,
        current_compatibility_sha256=current_compatibility_sha256,
        now=now,
    )


def promote_residency_policy_authorized(
    store: SQLiteResidencyPolicyStore,
    *,
    owner_id: str,
    service_id: str,
    domain_id: str,
    policy: DataResidencyPolicy,
    expected_revision: int | None,
    authorization: OperatorAuthorizationDecision,
    authorization_request: OperatorAuthorizationRequest,
    operator_policy_sha256: str,
    now: float,
) -> ResidencyPolicyRecord:
    _assert_exact_authorization(
        authorization,
        authorization_request,
        policy_sha256=operator_policy_sha256,
        owner_id=owner_id,
        domain_id=domain_id,
        action="residency.promote",
        resource_type="residency_policy",
        resource_sha256=policy.policy_sha256,
        now=now,
    )
    return store.promote(
        owner_id=owner_id,
        service_id=service_id,
        policy=policy,
        expected_revision=expected_revision,
        now=now,
    )


def apply_region_authority_authorized(
    store: SQLiteRegionAuthorityStore,
    decision: RegionAuthorityDecision,
    *,
    domain_id: str,
    expected_revision: int | None,
    authorization: OperatorAuthorizationDecision,
    authorization_request: OperatorAuthorizationRequest,
    operator_policy_sha256: str,
    now: float,
) -> RegionAuthorityRecord:
    if decision.action not in {"failover", "failback"}:
        raise ValueError("authorized region mutation entrypoint only accepts failover/failback decisions")
    _assert_exact_authorization(
        authorization,
        authorization_request,
        policy_sha256=operator_policy_sha256,
        owner_id=decision.owner_id,
        domain_id=domain_id,
        action=f"region.{decision.action}",
        resource_type="region_authority",
        resource_sha256=decision.decision_sha256,
        now=now,
    )
    return store.apply_decision(decision, expected_revision=expected_revision, now=now)


__all__ = [
    "apply_region_authority_authorized",
    "promote_residency_policy_authorized",
    "promote_runtime_stack_authorized",
    "rollback_runtime_stack_authorized",
]
