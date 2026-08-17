from __future__ import annotations

import hashlib

import pytest

from orchestration.continual_adaptation import (
    BenchmarkEvidence,
    ContinualWorkflowSpec,
    PromotionReceipt,
    RollbackReceipt,
    SQLiteContinualWorkflowStore,
    advance_continual_workflow,
    rollback_promoted_workflow,
)
from tools.continual_promotion import ContinualEvidence
from tools.feedback_promotion import CandidateMetrics, FeedbackBatchManifest, PromotionDecision
from tools.index_drift import IndexAdaptationDecision
from tools.training_lineage import TrainingOutcome, TrainingRequest


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def batch() -> FeedbackBatchManifest:
    return FeedbackBatchManifest(
        owner_id="alice",
        batch_id=sha("feedback-batch"),
        example_count=100,
        positive_weight=70.0,
        negative_weight=30.0,
        neutral_weight=0.0,
        subject_count=80,
        event_fingerprint=sha("events"),
    )


def spec(*, action: str = "shadow_rebuild") -> ContinualWorkflowSpec:
    reasons = () if action == "stable" else ("distribution_shift_detected",)
    return ContinualWorkflowSpec(
        owner_id="alice",
        baseline_version="retriever-v1",
        candidate_version="retriever-v2",
        drift_evidence_sha256=sha("drift"),
        adaptation_policy_sha256=sha("adaptation-policy"),
        adaptation_decision=IndexAdaptationDecision(action, reasons),
        training_request=TrainingRequest(
            run_id="continual-run-1",
            parent_artifact_sha256=sha("baseline-artifact"),
            dataset_sha256=sha("training-dataset"),
            code_revision="0123456789abcdef0123456789abcdef01234567",
            seed=7,
            config={"epochs": 3, "objective": "contrastive"},
        ),
        feedback_batch=batch(),
        benchmark_contract_sha256=sha("benchmark-contract"),
    )


def base_decision(workflow: ContinualWorkflowSpec, *, eligible: bool = True) -> PromotionDecision:
    baseline = CandidateMetrics(quality=0.70, p95_latency_ms=100.0, estimated_cost=1.0)
    candidate = CandidateMetrics(quality=0.75, p95_latency_ms=101.0, estimated_cost=1.0)
    return PromotionDecision(
        decision_id=sha("base-promotion-decision"),
        eligible=eligible,
        reason_codes=() if eligible else ("quality_regression",),
        owner_id=workflow.owner_id,
        batch_id=workflow.feedback_batch.batch_id,
        baseline_version=workflow.baseline_version,
        candidate_version=workflow.candidate_version,
        baseline=baseline,
        candidate=candidate,
        quality_delta=0.05,
        latency_ratio=1.01,
        cost_ratio=1.0,
        policy_fingerprint=sha("base-policy"),
    )


class BuildBackend:
    def __init__(self, *, fail_once: bool = False) -> None:
        self.calls: list[str] = []
        self.fail_once = fail_once

    def build(self, request, *, workflow_id, fencing_token):
        del request, fencing_token
        self.calls.append(workflow_id)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("transient build outage")
        return TrainingOutcome(
            output_artifact_sha256=sha("candidate-artifact"),
            evaluation_sha256=(sha("training-eval"),),
            provider_run_ref="trainer/job-7",
        )


class BenchmarkBackend:
    def __init__(self, *, wrong_artifact: bool = False, forgetting: float = 0.0) -> None:
        self.calls: list[str] = []
        self.wrong_artifact = wrong_artifact
        self.forgetting = forgetting

    def evaluate(self, workflow, lineage, *, workflow_id, fencing_token):
        del fencing_token
        self.calls.append(workflow_id)
        artifact = sha("wrong-artifact") if self.wrong_artifact else lineage.outcome.output_artifact_sha256
        return BenchmarkEvidence(
            workflow_id=workflow_id,
            benchmark_receipt_sha256=sha("benchmark-receipt"),
            verified_dataset_manifest_sha256=sha("verified-dataset"),
            experiment_sha256=sha("experiment"),
            candidate_artifact_sha256=artifact,
            base_decision=base_decision(workflow),
            continual_evidence=ContinualEvidence(
                drift_score=0.1,
                forgetting_delta=self.forgetting,
                forward_transfer_delta=0.01,
                privacy_safe_replay=True,
                independent_rollback_ready=True,
                adapter_version="retriever-v2",
            ),
        )


class PromotionBackend:
    def __init__(self) -> None:
        self.promotions: list[tuple[str, str]] = []
        self.rollbacks: list[str] = []

    def promote(
        self,
        workflow,
        lineage,
        evidence,
        decision,
        *,
        workflow_id,
        expected_baseline_version,
        fencing_token,
    ):
        del evidence, fencing_token
        self.promotions.append((workflow_id, expected_baseline_version))
        return PromotionReceipt(
            workflow_id=workflow_id,
            previous_version=workflow.baseline_version,
            promoted_version=workflow.candidate_version,
            candidate_artifact_sha256=lineage.outcome.output_artifact_sha256,
            decision_id=decision.decision_id,
            publication_sha256=sha("publication"),
        )

    def rollback(
        self,
        workflow,
        promotion,
        *,
        workflow_id,
        expected_current_version,
        fencing_token,
    ):
        del fencing_token
        assert expected_current_version == workflow.candidate_version
        self.rollbacks.append(workflow_id)
        return RollbackReceipt(
            workflow_id=workflow_id,
            rolled_back_from_version=workflow.candidate_version,
            restored_version=workflow.baseline_version,
            promotion_publication_sha256=promotion.publication_sha256,
            rollback_sha256=sha("rollback"),
        )


def test_stable_drift_is_held_without_build_or_benchmark(tmp_path) -> None:
    workflow = spec(action="stable")
    store = SQLiteContinualWorkflowStore(tmp_path / "continual.sqlite3")
    build = BuildBackend()
    benchmark = BenchmarkBackend()
    promotion = PromotionBackend()

    result = advance_continual_workflow(
        workflow,
        store=store,
        build_backend=build,
        benchmark_backend=benchmark,
        promotion_backend=promotion,
        worker_id="worker-a",
        now=10.0,
    )

    assert result.state == "stable_held"
    assert result.terminal_receipt_sha256 is not None
    assert build.calls == []
    assert benchmark.calls == []
    assert promotion.promotions == []


def test_transient_build_failure_retries_same_workflow_id(tmp_path) -> None:
    workflow = spec()
    store = SQLiteContinualWorkflowStore(tmp_path / "continual.sqlite3")
    build = BuildBackend(fail_once=True)
    benchmark = BenchmarkBackend()
    promotion = PromotionBackend()

    with pytest.raises(RuntimeError, match="transient build outage"):
        advance_continual_workflow(
            workflow,
            store=store,
            build_backend=build,
            benchmark_backend=benchmark,
            promotion_backend=promotion,
            worker_id="worker-a",
            now=10.0,
        )
    assert store.get(workflow.workflow_id).state == "build_requested"

    result = advance_continual_workflow(
        workflow,
        store=store,
        build_backend=build,
        benchmark_backend=benchmark,
        promotion_backend=promotion,
        worker_id="worker-b",
        now=20.0,
    )
    assert result.state == "promoted"
    assert build.calls == [workflow.workflow_id, workflow.workflow_id]


def test_benchmark_of_wrong_candidate_fails_closed_without_promotion(tmp_path) -> None:
    workflow = spec()
    store = SQLiteContinualWorkflowStore(tmp_path / "continual.sqlite3")
    promotion = PromotionBackend()

    result = advance_continual_workflow(
        workflow,
        store=store,
        build_backend=BuildBackend(),
        benchmark_backend=BenchmarkBackend(wrong_artifact=True),
        promotion_backend=promotion,
        worker_id="worker-a",
        now=10.0,
    )

    assert result.state == "failed"
    assert result.failure_type == "ContinualValidationError"
    assert result.terminal_receipt_sha256 is not None
    assert promotion.promotions == []


def test_continual_forgetting_gate_holds_candidate(tmp_path) -> None:
    workflow = spec()
    store = SQLiteContinualWorkflowStore(tmp_path / "continual.sqlite3")
    promotion = PromotionBackend()

    result = advance_continual_workflow(
        workflow,
        store=store,
        build_backend=BuildBackend(),
        benchmark_backend=BenchmarkBackend(forgetting=0.50),
        promotion_backend=promotion,
        worker_id="worker-a",
        now=10.0,
    )

    assert result.state == "held"
    assert "forgetting_budget_exceeded" in tuple(result.decision_payload["reason_codes"])
    assert promotion.promotions == []


def test_eligible_candidate_promotes_and_can_roll_back_independently(tmp_path) -> None:
    workflow = spec()
    store = SQLiteContinualWorkflowStore(tmp_path / "continual.sqlite3")
    promotion = PromotionBackend()

    promoted = advance_continual_workflow(
        workflow,
        store=store,
        build_backend=BuildBackend(),
        benchmark_backend=BenchmarkBackend(),
        promotion_backend=promotion,
        worker_id="worker-a",
        now=10.0,
    )
    assert promoted.state == "promoted"
    assert promotion.promotions == [(workflow.workflow_id, workflow.baseline_version)]
    assert promoted.promotion_payload["candidate_artifact_sha256"] == sha("candidate-artifact")

    rolled_back = rollback_promoted_workflow(
        workflow,
        store=store,
        promotion_backend=promotion,
        worker_id="worker-b",
        now=20.0,
    )
    assert rolled_back.state == "rolled_back"
    assert rolled_back.rollback_payload["restored_version"] == workflow.baseline_version
    assert promotion.rollbacks == [workflow.workflow_id]


def test_stale_executor_is_fenced_after_lease_takeover(tmp_path) -> None:
    workflow = spec()
    store = SQLiteContinualWorkflowStore(tmp_path / "continual.sqlite3")
    store.ensure(workflow, now=1.0)
    stale = store.claim(workflow.workflow_id, worker_id="worker-a", now=1.0, lease_seconds=2.0)
    current = store.claim(workflow.workflow_id, worker_id="worker-b", now=4.0, lease_seconds=10.0)

    assert current.fencing_token > stale.fencing_token
    with pytest.raises(RuntimeError, match="stale or fenced"):
        store.assert_claim(stale, now=4.0)
