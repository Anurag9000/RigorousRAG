from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from orchestration.target_population_reconciliation import (
    AliasBinding,
    DesiredTarget,
    PhysicalTarget,
    PopulationReconciliationJob,
    PopulationSnapshot,
    build_population_plan,
    execute_population_plan,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def desired(owner: str = "alice", logical: str = "primary") -> DesiredTarget:
    return DesiredTarget(
        owner_id=owner,
        kind="dense",
        logical_name=logical,
        generation_id="generation-7",
        profile_sha256=sha("profile"),
        schema_sha256=sha("schema"),
        source_sha256=sha("source"),
        expected_count=3,
    )


def physical(
    target: DesiredTarget,
    physical_id: str,
    *,
    state: str = "ready",
    generation_id: str | None = None,
    profile_sha256: str | None = None,
    schema_sha256: str | None = None,
    source_sha256: str | None = None,
    count: int | None = None,
    created_at: datetime | None = None,
    population_key: str | None = None,
) -> PhysicalTarget:
    return PhysicalTarget(
        owner_id=target.owner_id,
        physical_id=physical_id,
        kind=target.kind,
        logical_name=target.logical_name,
        generation_id=generation_id or target.generation_id,
        profile_sha256=profile_sha256 or target.profile_sha256,
        schema_sha256=schema_sha256 or target.schema_sha256,
        source_sha256=source_sha256 or target.source_sha256,
        observed_count=target.expected_count if count is None else count,
        state=state,
        created_at=created_at or datetime(2026, 8, 1, tzinfo=timezone.utc),
        population_key=population_key,
    )


class Backend:
    def __init__(self, targets=(), aliases=()) -> None:
        self.targets = {item.physical_id: item for item in targets}
        self.aliases = {item.key: item for item in aliases}
        self.protected: set[str] = set()
        self.fences: list[int] = []
        self.population_calls: list[str] = []
        self.orphan_calls: list[str] = []

    def assert_fence(self, fencing_token: int) -> None:
        self.fences.append(fencing_token)

    def begin_population(self, target, *, population_key, fencing_token):
        self.population_calls.append(population_key)
        created = physical(
            target,
            f"staged-{population_key[:12]}",
            state="building",
            population_key=population_key,
        )
        self.targets[created.physical_id] = created
        return created

    def inspect_physical(self, owner_id, physical_id):
        selected = self.targets.get(physical_id)
        if selected is not None and selected.owner_id != owner_id:
            raise AssertionError("cross-owner lookup")
        return selected

    def current_alias(self, owner_id, kind, logical_name):
        return self.aliases.get(
            (kind, logical_name),
            AliasBinding(owner_id, kind, logical_name, None, 0),
        )

    def compare_and_swap_alias(
        self,
        target,
        *,
        expected_physical_id,
        expected_revision,
        new_physical_id,
        fencing_token,
    ):
        key = target.key
        current = self.current_alias(target.owner_id, target.kind, target.logical_name)
        assert current.physical_id == expected_physical_id
        assert current.revision == expected_revision
        updated = AliasBinding(
            target.owner_id,
            target.kind,
            target.logical_name,
            new_physical_id,
            current.revision + 1,
        )
        self.aliases[key] = updated
        return updated

    def aliases_for_physical(self, owner_id, physical_id):
        return tuple(
            item
            for item in self.aliases.values()
            if item.owner_id == owner_id and item.physical_id == physical_id
        )

    def is_protected(self, owner_id, physical_id):
        del owner_id
        return physical_id in self.protected

    def record_orphan_candidate(self, target, *, fencing_token, plan_sha256):
        del fencing_token, plan_sha256
        self.orphan_calls.append(target.physical_id)
        return sha(f"candidate:{target.observation_sha256}")


class Inventory:
    def __init__(self, snapshot: PopulationSnapshot) -> None:
        self.value = snapshot

    def snapshot(self, owner_id: str) -> PopulationSnapshot:
        assert owner_id == self.value.owner_id
        return self.value


def test_exact_ready_live_target_is_healthy_and_action_free() -> None:
    target = desired()
    ready = physical(target, "dense-live")
    alias = AliasBinding("alice", "dense", "primary", "dense-live", 4)
    plan = build_population_plan(
        PopulationSnapshot("alice", (target,), (ready,), (alias,)),
        observed_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
    )

    assert [item.status for item in plan.findings] == ["healthy"]
    assert plan.actions == ()


def test_missing_target_submits_one_idempotent_hidden_population_without_alias_change() -> None:
    target = desired()
    plan = build_population_plan(
        PopulationSnapshot("alice", (target,), (), ()),
        observed_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
    )
    backend = Backend()

    receipts = execute_population_plan(plan, backend=backend, fencing_token=11)

    assert [item.action for item in plan.actions] == ["populate"]
    assert receipts[0].status == "population_submitted"
    assert backend.population_calls == [target.deterministic_population_key]
    assert backend.aliases == {}
    assert backend.fences


def test_matching_build_in_flight_prevents_duplicate_population() -> None:
    target = desired()
    staged = physical(
        target,
        "dense-stage",
        state="building",
        population_key=target.deterministic_population_key,
    )
    plan = build_population_plan(
        PopulationSnapshot(
            "alice",
            (target,),
            (staged,),
            (),
            in_flight_physical_ids=frozenset({"dense-stage"}),
        ),
        observed_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
    )

    assert [item.status for item in plan.findings] == ["population_in_flight"]
    assert plan.actions == ()


def test_exact_ready_population_uses_alias_cas_only_after_revalidation() -> None:
    target = desired()
    ready = physical(target, "dense-new")
    old_alias = AliasBinding("alice", "dense", "primary", "dense-old", 8)
    plan = build_population_plan(
        PopulationSnapshot("alice", (target,), (ready,), (old_alias,)),
        observed_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
    )
    backend = Backend((ready,), (old_alias,))

    receipts = execute_population_plan(plan, backend=backend, fencing_token=12)

    assert [item.action for item in plan.actions] == ["bind_alias"]
    assert receipts[0].status == "alias_bound"
    assert backend.aliases[target.key].physical_id == "dense-new"
    assert backend.aliases[target.key].revision == 9


def test_alias_change_after_plan_fails_closed() -> None:
    target = desired()
    ready = physical(target, "dense-new")
    observed_alias = AliasBinding("alice", "dense", "primary", "dense-old", 8)
    plan = build_population_plan(
        PopulationSnapshot("alice", (target,), (ready,), (observed_alias,)),
        observed_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
    )
    changed_alias = AliasBinding("alice", "dense", "primary", "dense-other", 9)
    backend = Backend((ready,), (changed_alias,))

    with pytest.raises(RuntimeError, match="alias changed"):
        execute_population_plan(plan, backend=backend, fencing_token=13)


def test_only_old_unaliased_unprotected_population_becomes_orphan_candidate() -> None:
    target = desired()
    live = physical(target, "dense-live")
    old = physical(
        target,
        "dense-old",
        generation_id="generation-6",
        created_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    alias = AliasBinding("alice", "dense", "primary", "dense-live", 4)
    plan = build_population_plan(
        PopulationSnapshot("alice", (target,), (live, old), (alias,)),
        observed_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
        orphan_grace_seconds=60,
    )
    backend = Backend((live, old), (alias,))

    assert "orphan_candidate" in {item.status for item in plan.findings}
    receipts = execute_population_plan(plan, backend=backend, fencing_token=14)
    assert any(item.status == "orphan_candidate_recorded" for item in receipts)
    assert backend.orphan_calls == ["dense-old"]
    assert "dense-old" in backend.targets  # reconciliation never deletes physical data


def test_live_protected_and_grace_window_populations_are_not_gc_candidates() -> None:
    target = desired()
    live = physical(target, "dense-live")
    protected = physical(target, "dense-protected", generation_id="generation-5")
    young = physical(
        target,
        "dense-young",
        generation_id="generation-4",
        created_at=datetime(2026, 8, 17, 11, 59, 50, tzinfo=timezone.utc),
    )
    alias = AliasBinding("alice", "dense", "primary", "dense-live", 2)
    snapshot = PopulationSnapshot(
        "alice",
        (target,),
        (live, protected, young),
        (alias,),
        protected_physical_ids=frozenset({"dense-protected"}),
    )
    plan = build_population_plan(
        snapshot,
        observed_at=datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc),
        orphan_grace_seconds=60,
    )

    assert "protected_orphan" in {item.status for item in plan.findings}
    assert not any(
        action.action == "record_orphan_candidate"
        and action.physical_id in {"dense-protected", "dense-young"}
        for action in plan.actions
    )


def test_snapshot_fails_closed_on_cross_owner_inventory() -> None:
    with pytest.raises(ValueError, match="crosses owner"):
        PopulationSnapshot("alice", (desired("bob"),), (), ())


def test_periodic_adapter_bounds_mutations_and_returns_continuation_signal() -> None:
    first = desired(logical="one")
    second = desired(logical="two")
    snapshot = PopulationSnapshot("alice", (first, second), (), ())
    backend = Backend()
    job = PopulationReconciliationJob(
        owner_id="alice",
        inventory=Inventory(snapshot),
        backend=backend,
        clock=lambda: datetime(2026, 8, 17, tzinfo=timezone.utc),
        max_actions=1,
    )

    result = job(fencing_token=21, continuation_token=None)

    assert result.examined == 2
    assert result.repaired == 1
    assert result.continuation_token is not None
    assert len(backend.population_calls) == 1
