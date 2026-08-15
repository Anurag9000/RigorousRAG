from tools.agent_trajectory import AgentTrajectoryStore


def test_trajectory_is_owner_scoped_ordered_and_idempotent(tmp_path):
    store = AgentTrajectoryStore(tmp_path / "trajectory.sqlite3", clock=lambda: 10.0)
    first = store.append(
        owner_id="owner-a",
        trajectory_id="t1",
        event_type="tool_result",
        agent="researcher",
        payload={"value": 1},
        idempotency_key="event-1",
    )
    replay = store.append(
        owner_id="owner-a",
        trajectory_id="t1",
        event_type="tool_result",
        agent="researcher",
        payload={"value": 999},
        idempotency_key="event-1",
    )
    store.append(
        owner_id="owner-a",
        trajectory_id="t1",
        event_type="checkpoint",
        agent="planner",
        payload={"step": 2},
        idempotency_key="checkpoint-2",
    )
    store.append(
        owner_id="owner-b",
        trajectory_id="t1",
        event_type="checkpoint",
        agent="planner",
        payload={"step": 88},
        idempotency_key="other-owner",
    )

    assert first.sequence == 1
    assert replay.sequence == 1
    assert replay.payload == {"value": 1}
    events = store.list_events(owner_id="owner-a", trajectory_id="t1")
    assert [item.sequence for item in events] == [1, 2]
    assert [item.payload for item in events] == [{"value": 1}, {"step": 2}]
    assert store.latest_checkpoint(owner_id="owner-a", trajectory_id="t1").payload == {"step": 2}
    assert store.latest_checkpoint(owner_id="owner-b", trajectory_id="t1").payload == {"step": 88}


def test_trajectory_default_redactor_removes_nested_credentials_before_storage(tmp_path):
    path = tmp_path / "trajectory.sqlite3"
    store = AgentTrajectoryStore(path)
    event = store.append(
        owner_id="o",
        trajectory_id="t",
        event_type="checkpoint",
        agent="planner",
        payload={
            "api_key": "sk-super-secret",
            "nested": {"token": "bearer-secret", "safe": "kept"},
            "items": [{"password": "pw"}],
        },
        idempotency_key="1",
    )

    assert event.payload["api_key"] == "[REDACTED]"
    assert event.payload["nested"] == {"safe": "kept", "token": "[REDACTED]"}
    assert event.payload["items"] == [{"password": "[REDACTED]"}]
    raw = path.read_bytes()
    assert b"sk-super-secret" not in raw
    assert b"bearer-secret" not in raw


def test_custom_redactor_can_enforce_application_specific_privacy(tmp_path):
    store = AgentTrajectoryStore(
        tmp_path / "trajectory.sqlite3",
        redactor=lambda payload: {"document_count": len(payload.get("documents", []))},
    )
    event = store.append(
        owner_id="o",
        trajectory_id="t",
        event_type="checkpoint",
        agent="planner",
        payload={"documents": ["private-a", "private-b"]},
        idempotency_key="x",
    )
    assert event.payload == {"document_count": 2}


def test_missing_checkpoint_is_explicit(tmp_path):
    store = AgentTrajectoryStore(tmp_path / "trajectory.sqlite3")
    assert store.latest_checkpoint(owner_id="o", trajectory_id="t") is None
