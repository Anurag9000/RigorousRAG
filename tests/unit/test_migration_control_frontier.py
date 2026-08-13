from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tools.control_api import build_control_router
from tools.feedback_store import FeedbackStore
from tools.migration_coordinator import MigrationCoordinator
from tools.review_routing import ReviewDecision
from tools.review_store import ReviewStore
from tools.security import Principal
from tools.versioned_state import CallbackVersionedState, SQLiteVersionedState, VersionedValue


def test_sqlite_versioned_state_compare_and_set_and_owner_isolation(tmp_path):
    store = SQLiteVersionedState(tmp_path / "state.sqlite3")
    first = store.compare_and_set(
        owner_id="owner-a", key="policy", expected_version=None, value={"v": 1}
    )
    assert first is not None and first.version == 1
    assert store.compare_and_set(
        owner_id="owner-a", key="policy", expected_version=None, value={"v": 2}
    ) is None
    second = store.compare_and_set(
        owner_id="owner-a", key="policy", expected_version=1, value={"v": 2}
    )
    assert second is not None and second.version == 2 and second.value == {"v": 2}
    assert store.get(owner_id="owner-b", key="policy") is None
    assert not store.delete(owner_id="owner-a", key="policy", expected_version=1)
    assert store.delete(owner_id="owner-a", key="policy", expected_version=2)


def test_callback_state_validates_external_owner_key_contract():
    adapter = CallbackVersionedState(
        get=lambda owner, key: VersionedValue("other", key, 1, {}, 0.0),
        compare_and_set=lambda owner, key, version, value: None,
        delete=lambda owner, key, version: False,
    )
    try:
        adapter.get(owner_id="owner-a", key="x")
    except RuntimeError as exc:
        assert "invariants" in str(exc)
    else:
        raise AssertionError("invalid external backend response was accepted")


@dataclass
class FakeParticipant:
    name: str
    fail_commit: bool = False
    events: list[str] = field(default_factory=list)

    def prepare(
        self,
        *,
        transaction_id: str,
        fencing_token: int,
        payload: Mapping[str, Any],
    ) -> None:
        self.events.append(f"prepare:{transaction_id}:{fencing_token}")

    def commit(
        self,
        *,
        transaction_id: str,
        fencing_token: int,
        payload: Mapping[str, Any],
    ) -> None:
        self.events.append(f"commit:{transaction_id}:{fencing_token}")
        if self.fail_commit:
            raise RuntimeError("commit failure")

    def rollback(
        self,
        *,
        transaction_id: str,
        fencing_token: int,
        payload: Mapping[str, Any],
    ) -> None:
        self.events.append(f"rollback:{transaction_id}:{fencing_token}")


def test_migration_coordinator_commits_idempotently(tmp_path):
    first, second = FakeParticipant("first"), FakeParticipant("second")
    coordinator = MigrationCoordinator(tmp_path / "migration.sqlite3")
    created = coordinator.begin(
        owner_id="owner-a",
        transaction_id="tx-1",
        resource_id="corpus",
        participants=(first, second),
        payload={"generation": 7},
    )
    assert created.state == "created"
    committed = coordinator.execute(
        owner_id="owner-a",
        transaction_id="tx-1",
        participants=(first, second),
        coordinator_id="worker-1",
    )
    assert committed.state == "committed" and committed.fencing_token >= 1
    replay = coordinator.execute(
        owner_id="owner-a",
        transaction_id="tx-1",
        participants=(first, second),
        coordinator_id="worker-2",
    )
    assert replay.state == "committed"
    assert sum(event.startswith("commit:") for event in first.events) == 1
    assert sum(event.startswith("commit:") for event in second.events) == 1


def test_migration_coordinator_compensates_commit_failure(tmp_path):
    first = FakeParticipant("first")
    second = FakeParticipant("second", fail_commit=True)
    coordinator = MigrationCoordinator(tmp_path / "migration.sqlite3")
    coordinator.begin(
        owner_id="owner-a",
        transaction_id="tx-2",
        resource_id="index",
        participants=(first, second),
        payload={"from": 1, "to": 2},
    )
    result = coordinator.execute(
        owner_id="owner-a",
        transaction_id="tx-2",
        participants=(first, second),
        coordinator_id="worker-1",
    )
    assert result.state == "rolled_back"
    assert any(event.startswith("rollback:") for event in first.events)
    assert any(event.startswith("rollback:") for event in second.events)


def test_migration_recovery_resumes_rollback_without_reentering_commit(tmp_path):
    first = FakeParticipant("first")
    second = FakeParticipant("second")
    coordinator = MigrationCoordinator(tmp_path / "migration.sqlite3")
    record = coordinator.begin(
        owner_id="owner-a",
        transaction_id="tx-recover",
        resource_id="index",
        participants=(first, second),
        payload={"generation": 9},
    )
    coordinator._done(record, "first", "prepared")
    coordinator._done(record, "second", "prepared")
    coordinator._done(record, "second", "rolled_back")
    coordinator._state(record, "rolling_back", 1, "simulated interrupted rollback")

    recovered = coordinator.recover(
        participants_by_name={"first": first, "second": second},
        coordinator_id="recovery-worker",
    )

    assert len(recovered) == 1 and recovered[0].state == "rolled_back"
    assert not any(event.startswith("prepare:") for event in first.events + second.events)
    assert not any(event.startswith("commit:") for event in first.events + second.events)
    assert sum(event.startswith("rollback:") for event in first.events) == 1
    assert not any(event.startswith("rollback:") for event in second.events)


def test_control_router_claim_resolve_feedback_and_owner_scope(tmp_path):
    reviews = ReviewStore(tmp_path / "reviews.sqlite3")
    feedback = FeedbackStore(tmp_path / "feedback.sqlite3")
    reviews.enqueue(
        owner_id="owner-a",
        request_id="req-1",
        decision=ReviewDecision("human_review", 0.8, ("high_uncertainty",)),
        query="secret query",
    )
    reviews.enqueue(
        owner_id="owner-b",
        request_id="req-2",
        decision=ReviewDecision("human_review", 0.9, ("evidence_conflict",)),
        query="other query",
    )

    async def principal() -> Principal:
        return Principal(owner_id="owner-a", authenticated=True)

    app = FastAPI()
    app.include_router(
        build_control_router(
            principal_dependency=principal,
            review_store=reviews,
            feedback_store=feedback,
        )
    )
    client = TestClient(app)

    listed = client.get("/reviews").json()
    assert [item["request_id"] for item in listed] == ["req-1"]
    claimed = client.post(
        "/reviews/claim",
        json={"reviewer_id": "reviewer-1", "ttl_seconds": 60},
    ).json()
    assert claimed["request_id"] == "req-1"
    assert claimed["query_sha256"] and "secret query" not in str(claimed)
    resolved = client.post(
        "/reviews/req-1/resolve",
        json={
            "reviewer_id": "reviewer-1",
            "lease_token": claimed["lease_token"],
            "resolution": "approved",
        },
    )
    assert resolved.status_code == 200 and resolved.json() == {"resolved": True}

    event = client.post(
        "/feedback",
        json={
            "event_id": "fb-1",
            "kind": "answer_correct",
            "subject_id": "req-1",
            "query": "secret query",
        },
    )
    assert event.status_code == 200
    body = event.json()
    assert body["query_sha256"] and "secret query" not in str(body)
    assert [item["event_id"] for item in client.get("/feedback").json()] == ["fb-1"]
    assert reviews.get(owner_id="owner-b", request_id="req-2") is not None
