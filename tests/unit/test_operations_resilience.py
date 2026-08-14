from __future__ import annotations

from tools.admission_control import AdmissionAction, AdmissionController, TokenBucket
from tools.backup_retention import BackupRecord, RetentionPolicy, plan_retention


class Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_backup_retention_preserves_minimum_immutability_hold_and_unverified() -> None:
    digest = "a" * 64
    records = [
        BackupRecord("old-delete", 0.0, digest),
        BackupRecord("old-hold", 1.0, digest, legal_hold=True),
        BackupRecord("old-immutable", 2.0, digest, immutable_until=200.0),
        BackupRecord("old-unverified", 3.0, digest, verified=False),
        BackupRecord("recent-a", 90.0, digest),
        BackupRecord("recent-b", 91.0, digest),
    ]
    plan = plan_retention(
        records,
        now=100.0,
        policy=RetentionPolicy(minimum_recovery_points=2, max_age_seconds=5.0),
    )
    assert plan.delete == ("old-delete",)
    assert set(plan.retain) == {
        "old-hold",
        "old-immutable",
        "old-unverified",
        "recent-a",
        "recent-b",
    }
    reasons = dict(plan.protected_reasons)
    assert "legal_hold" in reasons["old-hold"]
    assert "immutability_window" in reasons["old-immutable"]
    assert "unverified_backup" in reasons["old-unverified"]
    assert "minimum_recovery_point" in reasons["recent-a"]
    assert "minimum_recovery_point" in reasons["recent-b"]


def test_token_bucket_refills_deterministically() -> None:
    clock = Clock()
    bucket = TokenBucket(capacity=2.0, refill_per_second=0.5, clock=clock)
    assert bucket.allow()
    assert bucket.allow()
    assert not bucket.allow()
    clock.advance(2.0)
    assert bucket.allow()
    assert bucket.available() == 0.0


def test_admission_controller_backpressures_and_sheds_without_leaking_slots() -> None:
    controller = AdmissionController(
        max_inflight=1,
        backpressure_queue_depth=3,
        shed_queue_depth=5,
    )
    admitted, lease = controller.acquire(queue_depth=0)
    assert admitted.action == AdmissionAction.ADMIT and lease is not None
    blocked, blocked_lease = controller.acquire(queue_depth=0)
    assert blocked.action == AdmissionAction.BACKPRESSURE and blocked_lease is None
    shed, shed_lease = controller.acquire(queue_depth=5)
    assert shed.action == AdmissionAction.SHED and shed_lease is None
    assert controller.inflight == 1
    assert controller.release(lease)
    assert not controller.release(lease)
    queued, queued_lease = controller.acquire(queue_depth=3)
    assert queued.action == AdmissionAction.BACKPRESSURE and queued_lease is None
