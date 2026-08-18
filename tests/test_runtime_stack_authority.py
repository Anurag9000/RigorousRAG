from __future__ import annotations

import hashlib

import pytest

from orchestration.runtime_stack_authority import (
    RuntimeComponent,
    RuntimePromotionEvidence,
    RuntimePromotionPolicy,
    RuntimeRollbackRequest,
    RuntimeStackArtifact,
    SQLiteRuntimeStackAuthorityStore,
    decide_runtime_promotion,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stack(name: str, *, compatibility: str = "compat") -> RuntimeStackArtifact:
    return RuntimeStackArtifact.build(
        stack_id=name,
        components=(
            RuntimeComponent("dense_retriever", f"dense-{name}", _sha(f"dense:{name}"), _sha("dense-contract")),
            RuntimeComponent("reranker", f"reranker-{name}", _sha(f"reranker:{name}"), _sha("reranker-contract")),
            RuntimeComponent("generator", f"generator-{name}", _sha(f"generator:{name}"), _sha("generator-contract")),
        ),
        retrieval_contract_sha256=_sha("retrieval-contract"),
        generation_contract_sha256=_sha("generation-contract"),
        compatibility_sha256=_sha(compatibility),
        source_revision="a" * 40,
    )


def _decision(stack: RuntimeStackArtifact, *, now: float = 10.0, ttl: float = 100.0):
    policy = RuntimePromotionPolicy(
        "promotion-v1",
        ("offline_quality", "security_review", "compatibility"),
        decision_ttl_seconds=ttl,
    )
    evidence = tuple(
        RuntimePromotionEvidence(kind, _sha(f"{kind}:{stack.stack_id}"), stack.stack_sha256, now - 1.0, now + 1000.0)
        for kind in policy.required_evidence_kinds
    )
    return decide_runtime_promotion(
        stack,
        evidence=evidence,
        policy=policy,
        now=now,
        current_compatibility_sha256=stack.compatibility_sha256,
    )


def test_missing_or_stale_required_evidence_blocks_promotion() -> None:
    stack = _stack("v1")
    policy = RuntimePromotionPolicy("policy", ("offline_quality", "security_review"))
    evidence = (RuntimePromotionEvidence("offline_quality", _sha("quality"), stack.stack_sha256, 0.0, 5.0),)
    decision = decide_runtime_promotion(
        stack,
        evidence=evidence,
        policy=policy,
        now=10.0,
        current_compatibility_sha256=stack.compatibility_sha256,
    )
    assert decision.eligible is False
    assert "missing_required_evidence:security_review" in decision.reason_codes
    assert "stale_or_not_yet_valid_evidence:offline_quality" in decision.reason_codes


def test_promotion_decision_expires_before_store_application(tmp_path) -> None:
    store = SQLiteRuntimeStackAuthorityStore(tmp_path / "runtime.sqlite")
    stack = _stack("v1")
    decision = _decision(stack, now=10.0, ttl=5.0)
    with pytest.raises(ValueError, match="stale or not yet valid"):
        store.promote(
            owner_id="owner",
            service_id="rag",
            domain_id="science",
            stack=stack,
            decision=decision,
            expected_authority_revision=0,
            now=16.0,
        )


def test_promotions_rotate_fence_and_stale_stack_is_rejected(tmp_path) -> None:
    store = SQLiteRuntimeStackAuthorityStore(tmp_path / "runtime.sqlite")
    first_stack = _stack("v1")
    second_stack = _stack("v2")
    first = store.promote(
        owner_id="owner",
        service_id="rag",
        domain_id="science",
        stack=first_stack,
        decision=_decision(first_stack),
        expected_authority_revision=0,
        now=11.0,
    )
    second = store.promote(
        owner_id="owner",
        service_id="rag",
        domain_id="science",
        stack=second_stack,
        decision=_decision(second_stack, now=12.0),
        expected_authority_revision=first.authority_revision,
        now=13.0,
    )
    assert second.authority_revision == first.authority_revision + 1
    assert second.fencing_token == first.fencing_token + 1
    with pytest.raises(RuntimeError, match="stale or non-authoritative"):
        store.assert_runtime_authority(
            owner_id="owner",
            service_id="rag",
            domain_id="science",
            stack_sha256=first_stack.stack_sha256,
            fencing_token=first.fencing_token,
        )
    assert store.assert_runtime_authority(
        owner_id="owner",
        service_id="rag",
        domain_id="science",
        stack_sha256=second_stack.stack_sha256,
        fencing_token=second.fencing_token,
    ).stack_sha256 == second_stack.stack_sha256


def test_rollback_advances_history_instead_of_rewinding(tmp_path) -> None:
    store = SQLiteRuntimeStackAuthorityStore(tmp_path / "runtime.sqlite")
    first_stack = _stack("v1")
    second_stack = _stack("v2")
    first = store.promote(
        owner_id="owner", service_id="rag", domain_id="science",
        stack=first_stack, decision=_decision(first_stack), expected_authority_revision=0, now=11.0,
    )
    second = store.promote(
        owner_id="owner", service_id="rag", domain_id="science",
        stack=second_stack, decision=_decision(second_stack, now=12.0), expected_authority_revision=first.authority_revision, now=13.0,
    )
    request = RuntimeRollbackRequest(
        "owner", "rag", "science", first.authority_revision, _sha("reason"), _sha("actor"), 14.0
    )
    rolled = store.rollback(
        request,
        expected_authority_revision=second.authority_revision,
        current_compatibility_sha256=first_stack.compatibility_sha256,
        now=15.0,
    )
    assert rolled.stack_sha256 == first_stack.stack_sha256
    assert rolled.authority_revision == second.authority_revision + 1
    assert rolled.fencing_token == second.fencing_token + 1
    assert [row.authority_revision for row in store.history(owner_id="owner", service_id="rag", domain_id="science")] == [3, 2, 1]


def test_rollback_refuses_historical_stack_incompatible_with_current_environment(tmp_path) -> None:
    store = SQLiteRuntimeStackAuthorityStore(tmp_path / "runtime.sqlite")
    old = _stack("old", compatibility="old-layout")
    current = _stack("current", compatibility="new-layout")
    first = store.promote(
        owner_id="owner", service_id="rag", domain_id="science",
        stack=old, decision=_decision(old), expected_authority_revision=0, now=11.0,
    )
    second = store.promote(
        owner_id="owner", service_id="rag", domain_id="science",
        stack=current, decision=_decision(current, now=12.0), expected_authority_revision=first.authority_revision, now=13.0,
    )
    request = RuntimeRollbackRequest("owner", "rag", "science", 1, _sha("reason"), _sha("actor"), 14.0)
    with pytest.raises(ValueError, match="incompatible"):
        store.rollback(
            request,
            expected_authority_revision=second.authority_revision,
            current_compatibility_sha256=current.compatibility_sha256,
            now=15.0,
        )


def test_promotion_is_compare_and_swap_bound(tmp_path) -> None:
    store = SQLiteRuntimeStackAuthorityStore(tmp_path / "runtime.sqlite")
    first_stack = _stack("v1")
    first = store.promote(
        owner_id="owner", service_id="rag", domain_id="science",
        stack=first_stack, decision=_decision(first_stack), expected_authority_revision=0, now=11.0,
    )
    second_stack = _stack("v2")
    with pytest.raises(RuntimeError, match="promotion CAS failed"):
        store.promote(
            owner_id="owner", service_id="rag", domain_id="science",
            stack=second_stack, decision=_decision(second_stack, now=12.0), expected_authority_revision=first.authority_revision + 1, now=13.0,
        )
