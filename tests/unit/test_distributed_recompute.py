from __future__ import annotations

import sqlite3

import pytest

from tools.dependency_invalidation import DependencyInvalidationStore, DependencyRef, RecomputeTask
from tools.distributed_recompute import DistributedRecomputeBridge, claim_exact_recompute_task
from tools.durable_queue import InMemoryDurableQueue
from tools.recompute_executor import RecomputeOutcome, ResearchRecomputeExecutor


OWNER = "owner-distributed-recompute"
OTHER_OWNER = "owner-distributed-recompute-other"


def _seed_task(
    store: DependencyInvalidationStore,
    *,
    owner: str = OWNER,
    suffix: str = "one",
    kind: str = "result",
) -> RecomputeTask:
    root = DependencyRef("source", f"source-{suffix}")
    artifact = DependencyRef(kind, f"artifact-{suffix}")
    store.register_dependency(
        owner,
        upstream=root,
        downstream=artifact,
        relation="derived-from",
    )
    impact = store.invalidate(
        owner,
        root=root,
        reason=f"source {suffix} changed",
        event_type="test-change",
    )
    assert len(impact.recompute_tasks) == 1
    return impact.recompute_tasks[0]


def _task(store: DependencyInvalidationStore, owner: str, task_id: str) -> RecomputeTask:
    matches = [item for item in store.list_recompute(owner, limit=100) if item.task_id == task_id]
    assert len(matches) == 1
    return matches[0]


class _RecordingQueue(InMemoryDurableQueue):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.payloads: list[dict[str, object]] = []
        self.keys: list[str] = []

    def enqueue(self, payload, *, idempotency_key):
        self.payloads.append(dict(payload))
        self.keys.append(idempotency_key)
        return super().enqueue(payload, idempotency_key=idempotency_key)


class _FakeExecutor:
    def __init__(self, store: DependencyInvalidationStore, *, success: bool = True):
        self.store = store
        self.success = success
        self.calls: list[tuple[str, str]] = []

    def execute_claimed(self, owner_id: str, task_id: str) -> RecomputeOutcome:
        self.calls.append((owner_id, task_id))
        task = _task(self.store, owner_id, task_id)
        assert task.status == "claimed"
        error = "SyntheticFailure" if not self.success else ""
        self.store.finish_recompute(
            owner_id,
            task_id,
            success=self.success,
            error_type=error,
            acknowledge_stale=self.success,
        )
        return RecomputeOutcome(task, self.success, None, error)


def _bridge(
    store: DependencyInvalidationStore,
    queue: InMemoryDurableQueue,
    *,
    success: bool = True,
    max_attempts: int = 5,
    claim_timeout_seconds: float = 60.0,
):
    executor = _FakeExecutor(store, success=success)
    bridge = DistributedRecomputeBridge(
        owner_id=OWNER,
        invalidations=store,
        executor=executor,  # type: ignore[arg-type]
        queue=queue,
        max_attempts=max_attempts,
        claim_timeout_seconds=claim_timeout_seconds,
    )
    return bridge, executor


def test_exact_claim_is_owner_and_task_scoped(tmp_path):
    store = DependencyInvalidationStore(tmp_path / "invalidation.sqlite3")
    first = _seed_task(store, suffix="first")
    second = _seed_task(store, suffix="second")

    assert claim_exact_recompute_task(store, OTHER_OWNER, first.task_id).state == "missing"
    decision = claim_exact_recompute_task(store, OWNER, second.task_id)

    assert decision.state == "claimed"
    assert decision.task is not None
    assert decision.task.task_id == second.task_id
    assert _task(store, OWNER, first.task_id).status == "queued"
    assert _task(store, OWNER, second.task_id).status == "claimed"


def test_publish_payload_is_opaque_and_idempotent(tmp_path):
    store = DependencyInvalidationStore(tmp_path / "invalidation.sqlite3")
    task = _seed_task(store)
    queue = _RecordingQueue()
    bridge, _ = _bridge(store, queue)

    first = bridge.publish_ready()
    second = bridge.publish_ready()

    assert first == second
    assert len(first) == 1
    assert queue.payloads == [{"task_id": task.task_id}, {"task_id": task.task_id}]
    assert set(queue.payloads[0]) == {"task_id"}
    assert OWNER not in str(queue.payloads[0])
    assert queue.keys[0] == queue.keys[1]
    assert OWNER not in queue.keys[0]


def test_worker_executes_exact_authoritative_task_once(tmp_path):
    store = DependencyInvalidationStore(tmp_path / "invalidation.sqlite3")
    task = _seed_task(store)
    queue = InMemoryDurableQueue()
    bridge, executor = _bridge(store, queue)
    bridge.publish_ready()

    result = bridge.work_one(worker_id="worker-a")

    assert result.state == "completed"
    assert result.task_id == task.task_id
    assert executor.calls == [(OWNER, task.task_id)]
    assert _task(store, OWNER, task.task_id).status == "completed"
    assert bridge.work_one(worker_id="worker-a").state == "idle"


def test_terminal_duplicate_is_acknowledged_without_execution(tmp_path):
    store = DependencyInvalidationStore(tmp_path / "invalidation.sqlite3")
    task = _seed_task(store)
    claimed = claim_exact_recompute_task(store, OWNER, task.task_id)
    assert claimed.state == "claimed"
    store.finish_recompute(OWNER, task.task_id, success=True)
    queue = InMemoryDurableQueue()
    queue.enqueue({"task_id": task.task_id}, idempotency_key="duplicate")
    bridge, executor = _bridge(store, queue)

    result = bridge.work_one(worker_id="worker-a")

    assert result.state == "duplicate"
    assert executor.calls == []
    assert bridge.work_one(worker_id="worker-a").state == "idle"


def test_fresh_competing_claim_is_retried_not_lost(tmp_path):
    store = DependencyInvalidationStore(tmp_path / "invalidation.sqlite3")
    task = _seed_task(store)
    assert claim_exact_recompute_task(store, OWNER, task.task_id).state == "claimed"
    queue = InMemoryDurableQueue(max_attempts=3)
    queue.enqueue({"task_id": task.task_id}, idempotency_key="busy")
    bridge, executor = _bridge(store, queue, claim_timeout_seconds=3600.0)

    result = bridge.work_one(worker_id="worker-b", busy_retry_delay=0.0)

    assert result.state == "busy"
    assert executor.calls == []
    # Nack made the handoff available again instead of acknowledging it.
    assert queue.claim("observer", visibility_timeout=10.0) is not None


def test_expired_claim_is_recovered_and_executed(tmp_path):
    store = DependencyInvalidationStore(tmp_path / "invalidation.sqlite3")
    task = _seed_task(store)
    first = claim_exact_recompute_task(store, OWNER, task.task_id, claim_timeout_seconds=1.0)
    assert first.state == "claimed"
    connection = sqlite3.connect(str(store.path))
    try:
        connection.execute(
            "UPDATE recompute_tasks SET claimed_at=0 WHERE owner_id=? AND task_id=?",
            (OWNER, task.task_id),
        )
        connection.commit()
    finally:
        connection.close()
    queue = InMemoryDurableQueue()
    queue.enqueue({"task_id": task.task_id}, idempotency_key="recover")
    bridge, executor = _bridge(store, queue, claim_timeout_seconds=1.0)

    result = bridge.work_one(worker_id="worker-recovery")

    assert result.state == "completed"
    assert executor.calls == [(OWNER, task.task_id)]
    completed = _task(store, OWNER, task.task_id)
    assert completed.status == "completed"
    assert completed.attempts == 2


def test_exhausted_stale_claim_fails_closed_and_acks_duplicate(tmp_path):
    store = DependencyInvalidationStore(tmp_path / "invalidation.sqlite3")
    task = _seed_task(store)
    connection = sqlite3.connect(str(store.path))
    try:
        connection.execute(
            """UPDATE recompute_tasks
               SET status='claimed',attempts=2,claimed_at=0
               WHERE owner_id=? AND task_id=?""",
            (OWNER, task.task_id),
        )
        connection.commit()
    finally:
        connection.close()
    queue = InMemoryDurableQueue()
    queue.enqueue({"task_id": task.task_id}, idempotency_key="exhausted")
    bridge, executor = _bridge(
        store,
        queue,
        max_attempts=2,
        claim_timeout_seconds=1.0,
    )

    result = bridge.work_one(worker_id="worker-a")

    assert result.state == "duplicate"
    assert executor.calls == []
    failed = _task(store, OWNER, task.task_id)
    assert failed.status == "failed"
    assert failed.error_type == "ClaimAttemptsExhausted"


def test_malformed_transport_payload_never_reaches_ledger_handler(tmp_path):
    store = DependencyInvalidationStore(tmp_path / "invalidation.sqlite3")
    task = _seed_task(store)
    queue = InMemoryDurableQueue(max_attempts=1)
    queue.enqueue(
        {"task_id": task.task_id, "owner_id": OWNER},
        idempotency_key="malformed",
    )
    bridge, executor = _bridge(store, queue)

    result = bridge.work_one(worker_id="worker-a", busy_retry_delay=0.0)

    assert result.state == "invalid"
    assert executor.calls == []
    assert _task(store, OWNER, task.task_id).status == "queued"
    assert len(queue.dead_letters()) == 1


def test_handler_failure_is_authoritative_and_transport_is_settled(tmp_path):
    store = DependencyInvalidationStore(tmp_path / "invalidation.sqlite3")
    task = _seed_task(store)
    queue = InMemoryDurableQueue()
    bridge, executor = _bridge(store, queue, success=False)
    bridge.publish_ready()

    result = bridge.work_one(worker_id="worker-a")

    assert result.state == "failed"
    assert executor.calls == [(OWNER, task.task_id)]
    failed = _task(store, OWNER, task.task_id)
    assert failed.status == "failed"
    assert failed.error_type == "SyntheticFailure"
    assert bridge.work_one(worker_id="worker-a").state == "idle"


def test_executor_execute_claimed_reloads_authoritative_row(tmp_path):
    store = DependencyInvalidationStore(tmp_path / "invalidation.sqlite3")
    task = _seed_task(store, suffix="matrix", kind="matrix")
    calls: list[tuple[str, str]] = []

    def handler(owner_id: str, claimed: RecomputeTask):
        calls.append((owner_id, claimed.task_id))
        return None

    executor = ResearchRecomputeExecutor(
        invalidations=store,
        replacements=None,  # type: ignore[arg-type]
        results=None,  # type: ignore[arg-type]
        reports=None,  # type: ignore[arg-type]
        workspace=None,  # type: ignore[arg-type]
        composition=None,  # type: ignore[arg-type]
        agent_factory=lambda _owner, _model: None,
        custom_handlers={"matrix": handler},
    )

    with pytest.raises(RuntimeError, match="must be claimed"):
        executor.execute_claimed(OWNER, task.task_id)
    assert calls == []

    claimed = claim_exact_recompute_task(store, OWNER, task.task_id)
    assert claimed.state == "claimed"
    outcome = executor.execute_claimed(OWNER, task.task_id)

    assert outcome.success is True
    assert calls == [(OWNER, task.task_id)]
    assert _task(store, OWNER, task.task_id).status == "completed"
