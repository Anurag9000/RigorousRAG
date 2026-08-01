from types import SimpleNamespace

from tools.migration_planner import (
    inventory_profile_migrations,
    migration_task_id,
)


class Generations:
    def list_current(self, *, owner_id, limit):
        assert owner_id == "alice"
        assert limit == 10
        return (
            SimpleNamespace(
                doc_id="ready",
                sequence=1,
                state="active",
                profile_fingerprint="a" * 64,
            ),
            SimpleNamespace(
                doc_id="already",
                sequence=2,
                state="active",
                profile_fingerprint="b" * 64,
            ),
            SimpleNamespace(
                doc_id="missing",
                sequence=3,
                state="restored",
                profile_fingerprint="a" * 64,
            ),
            SimpleNamespace(
                doc_id="deleted",
                sequence=4,
                state="deleted",
                profile_fingerprint="a" * 64,
            ),
            SimpleNamespace(
                doc_id="broken-registry",
                sequence=5,
                state="active",
                profile_fingerprint="a" * 64,
            ),
        )


class Documents:
    def get(self, *, owner_id, doc_id, verify_visual):
        assert owner_id == "alice"
        assert verify_visual is False
        if doc_id == "ready":
            return {
                "source_retained": True,
                "source_path": "/private/not-returned.pdf",
            }
        if doc_id == "broken-registry":
            raise RuntimeError("failed at /private/registry.sqlite3")
        return None


def test_inventory_classifies_profile_drift_without_returning_paths(monkeypatch):
    target = SimpleNamespace(name="target", fingerprint="b" * 64)
    monkeypatch.setattr(
        "tools.migration_planner.resolve_embedding_profile",
        lambda value: target,
    )

    candidates = inventory_profile_migrations(
        owner_id="alice",
        target_profile="target",
        generations=Generations(),
        document_store=Documents(),
        limit=10,
    )

    by_id = {item.doc_id: item for item in candidates}
    assert by_id["ready"].eligible is True
    assert by_id["ready"].reason == "ready"
    assert by_id["already"].reason == "already_target_profile"
    assert by_id["missing"].reason == "retained_source_unavailable"
    assert by_id["deleted"].reason == "deleted"
    assert by_id["broken-registry"].reason == "registry_inspection_failed"
    assert "/private" not in repr(candidates)


def test_task_id_is_stable_and_changes_with_source_or_target(monkeypatch):
    profiles = {
        "one": SimpleNamespace(name="one", fingerprint="b" * 64),
        "two": SimpleNamespace(name="two", fingerprint="c" * 64),
    }
    monkeypatch.setattr(
        "tools.migration_planner.resolve_embedding_profile",
        lambda value: profiles[value],
    )
    first = inventory_profile_migrations(
        owner_id="alice",
        target_profile="one",
        generations=Generations(),
        document_store=Documents(),
        limit=10,
    )[0]
    repeated = inventory_profile_migrations(
        owner_id="alice",
        target_profile="one",
        generations=Generations(),
        document_store=Documents(),
        limit=10,
    )[0]
    second_target = inventory_profile_migrations(
        owner_id="alice",
        target_profile="two",
        generations=Generations(),
        document_store=Documents(),
        limit=10,
    )[0]
    assert migration_task_id(first) == migration_task_id(repeated)
    assert migration_task_id(first) != migration_task_id(second_target)
    assert len(migration_task_id(first)) == 64
