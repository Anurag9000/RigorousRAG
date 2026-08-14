from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tools.distributed_coordination import InMemoryLeaseCoordinator, RedisLeaseCoordinator
from tools.distributed_jobs import FencedJobRunner, LeaseLostError
from tools.feedback_promotion import (
    CandidateMetrics,
    PromotionPolicy,
    build_feedback_batch,
    evaluate_promotion,
)
from tools.feedback_store import ActiveLearningExample
from tools.operations_cli import main as operations_main
from tools.promotion_journal import PromotionJournal
from tools.release_inventory import (
    build_cyclonedx_sbom,
    build_spdx_sbom,
    canonical_sha256,
    reproducible_timestamp,
)


def _example(kind: str, subject: str) -> ActiveLearningExample:
    return ActiveLearningExample(
        kind=kind,
        subject_id=subject,
        weight=1.0,
        metadata={"source": "review"},
        query_sha256="a" * 64,
        evidence_sha256="b" * 64,
    )


def _decision():
    batch = build_feedback_batch(
        owner_id="owner",
        examples=[_example("answer_correct", f"good-{index}") for index in range(18)]
        + [_example("answer_incorrect", f"bad-{index}") for index in range(2)],
    )
    return evaluate_promotion(
        batch=batch,
        baseline_version="v1",
        candidate_version="v2",
        baseline=CandidateMetrics(0.80, 100, 1.0),
        candidate=CandidateMetrics(0.83, 105, 1.02),
        policy=PromotionPolicy(
            min_examples=20,
            min_negative_weight_fraction=0.10,
            min_quality_gain=0.01,
        ),
    )


def test_promotion_journal_persists_chain_and_idempotency(tmp_path: Path) -> None:
    path = tmp_path / "promotion.sqlite3"
    decision = _decision()
    journal = PromotionJournal(path, clock=lambda: 123.0)
    eligible = journal.append(decision=decision, action="eligible", actor="evaluator")
    assert journal.append(decision=decision, action="eligible", actor="other") == eligible
    promoted = journal.append(decision=decision, action="promoted", actor="release-controller")
    assert promoted.previous_hash == eligible.record_hash

    reopened = PromotionJournal(path)
    assert [entry.action for entry in reopened.entries()] == ["eligible", "promoted"]
    assert reopened.verify_chain().valid


def test_promotion_journal_rejects_invalid_transition(tmp_path: Path) -> None:
    journal = PromotionJournal(tmp_path / "promotion.sqlite3")
    with pytest.raises(ValueError, match="invalid promotion transition"):
        journal.append(decision=_decision(), action="promoted", actor="operator")


def test_promotion_journal_detects_database_tamper(tmp_path: Path) -> None:
    path = tmp_path / "promotion.sqlite3"
    journal = PromotionJournal(path, clock=lambda: 123.0)
    journal.append(decision=_decision(), action="eligible", actor="evaluator")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE promotion_journal SET actor='intruder' WHERE sequence=1")
    verification = journal.verify_chain()
    assert not verification.valid
    assert verification.first_invalid_sequence == 1


def test_fenced_job_runner_skips_non_leader_and_releases_after_run() -> None:
    coordinator = InMemoryLeaseCoordinator()
    blocker = coordinator.acquire(name="job:reconcile", holder="other", ttl_seconds=30)
    assert blocker is not None
    runner = FencedJobRunner(coordinator=coordinator, holder="worker", ttl_seconds=30)
    skipped = runner.run("reconcile", lambda context: context.fencing_token)
    assert not skipped.acquired
    assert coordinator.release(blocker)

    result = runner.run("reconcile", lambda context: context.fencing_token)
    assert result.acquired
    assert result.value == result.fencing_token
    assert coordinator.acquire(name="job:reconcile", holder="next", ttl_seconds=30) is not None


def test_fenced_job_runner_fails_when_lease_expires() -> None:
    now = [10.0]
    coordinator = InMemoryLeaseCoordinator(clock=lambda: now[0])
    runner = FencedJobRunner(coordinator=coordinator, holder="worker", ttl_seconds=1)

    def expire(context):
        now[0] = 12.0
        context.heartbeat()

    with pytest.raises(LeaseLostError):
        runner.run("compact", expire)


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.counters: dict[str, int] = {}

    def incr(self, key: str) -> int:
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    def set(self, key: str, value: str, *, nx: bool, px: int) -> bool:
        del px
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    def eval(self, script: str, count: int, key: str, value: str, *args: object) -> int:
        assert count == 1
        if self.values.get(key) != value:
            return 0
        if "pexpire" in script:
            assert args
            return 1
        if "del" in script:
            del self.values[key]
            return 1
        raise AssertionError("unexpected script")


def test_redis_lease_adapter_uses_cas_renew_and_release() -> None:
    client = _FakeRedis()
    coordinator = RedisLeaseCoordinator(client, clock=lambda: 100.0)
    first = coordinator.acquire(name="promotion", holder="one", ttl_seconds=5)
    assert first is not None
    assert coordinator.acquire(name="promotion", holder="two", ttl_seconds=5) is None
    renewed = coordinator.renew(first, ttl_seconds=5)
    assert renewed is not None
    assert coordinator.release(renewed)
    second = coordinator.acquire(name="promotion", holder="two", ttl_seconds=5)
    assert second is not None
    assert second.token > first.token


def test_release_inventories_are_deterministic_and_standard_shaped() -> None:
    records = [{"name": "zeta", "version": "2"}, {"name": "alpha", "version": "1"}]
    cyclonedx = build_cyclonedx_sbom(records)
    spdx = build_spdx_sbom(
        reversed(records),
        namespace="https://example.invalid/rigorousrag/sbom/1",
        created=reproducible_timestamp(0),
    )
    assert cyclonedx["bomFormat"] == "CycloneDX"
    assert cyclonedx["specVersion"] == "1.6"
    assert spdx["spdxVersion"] == "SPDX-2.3"
    assert spdx["creationInfo"]["created"] == "1970-01-01T00:00:00Z"
    assert [item["name"] for item in cyclonedx["components"]] == ["alpha", "zeta"]
    assert canonical_sha256(cyclonedx) == canonical_sha256(build_cyclonedx_sbom(records))


def test_operations_cli_builds_inventory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "pip-list.json"
    source.write_text(json.dumps([{"name": "alpha", "version": "1.2.3"}]), encoding="utf-8")
    output = tmp_path / "sbom.json"
    status = operations_main(
        [
            "inventory-build",
            "--pip-list",
            str(source),
            "--output",
            str(output),
            "--format",
            "spdx",
            "--source-date-epoch",
            "0",
        ]
    )
    assert status == 0
    assert json.loads(output.read_text(encoding="utf-8"))["spdxVersion"] == "SPDX-2.3"
    reported = json.loads(capsys.readouterr().out)
    assert reported["component_count"] == 1


def test_operations_cli_backup_verify_restore_roundtrip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "state.sqlite3"
    source.write_bytes(b"state")
    backup = tmp_path / "backup"
    assert (
        operations_main(
            [
                "backup-create",
                "--source",
                str(source),
                "--destination",
                str(backup),
                "--generation",
                "g1",
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest = backup / "manifest.json"
    assert operations_main(["backup-verify", "--source", str(backup), "--manifest", str(manifest)]) == 0
    capsys.readouterr()
    restored = tmp_path / "restored"
    assert (
        operations_main(
            [
                "backup-restore",
                "--source",
                str(backup),
                "--manifest",
                str(manifest),
                "--destination",
                str(restored),
            ]
        )
        == 0
    )
    assert (restored / "state.sqlite3").read_bytes() == b"state"


def test_operations_cli_returns_nonzero_for_canary_regression(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = operations_main(
        [
            "canary-evaluate",
            "--requests",
            "100",
            "--errors",
            "2",
            "--baseline-p95-latency-ms",
            "100",
            "--canary-p95-latency-ms",
            "130",
            "--quality-delta",
            "-0.01",
        ]
    )
    assert status == 3
    assert json.loads(capsys.readouterr().out)["rollback"] is True
