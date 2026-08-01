from __future__ import annotations

from types import SimpleNamespace

from tools.migration_journal import MigrationJournal
from tools.migration_shadow_runtime import (
    claim_shadow_build_task,
    clear_migration_shadow_store_cache,
    execute_next_shadow_build,
    get_migration_shadow_store,
)
from tools.migration_shadow_store import MigrationShadowStore, ShadowBuild
from tools.migration_types import MigrationCandidate

SOURCE = "a" * 64
TARGET = "b" * 64
CONTENT = "c" * 64
PARSER = "d" * 64


def candidate(doc_id, sequence):
    return MigrationCandidate(
        owner_id="alice",
        doc_id=doc_id,
        source_sequence=sequence,
        source_profile_fingerprint=SOURCE,
        target_profile_name="bge-m3",
        target_profile_fingerprint=TARGET,
        retained_source=True,
        eligible=True,
        reason="profile_drift",
    )


def build():
    return ShadowBuild(
        content_sha256=CONTENT,
        parser_fingerprint=PARSER,
        vector_rows=({"row_id": "one", "embedding": [0.1]},),
        sparse_rows=({"field_id": "one", "text": "source"},),
    )


def test_shadow_claim_excludes_validated_tasks_and_claims_planned_work(tmp_path):
    journal = MigrationJournal(tmp_path / "journal.sqlite3")
    first, second = journal.seed((candidate("doc-1", 1), candidate("doc-2", 2)), now=1.0)
    claimed = journal.claim(
        owner_id="alice",
        worker_id="validator",
        lease_seconds=10,
        now=2.0,
    )
    assert claimed.task_id == first.task_id
    journal.mark_validated(
        task_id=claimed.task_id,
        worker_id="validator",
        validation_digest="e" * 64,
        now=3.0,
    )
    shadow_claim = claim_shadow_build_task(
        journal,
        owner_id="alice",
        worker_id="builder",
        lease_seconds=20,
        now=4.0,
    )
    assert shadow_claim.task_id == second.task_id
    assert shadow_claim.state == "running"
    assert shadow_claim.lease_owner == "builder"
    assert journal.get(first.task_id).state == "validated"


def test_expired_running_and_failed_tasks_are_reclaimed_with_attempt_ceiling(tmp_path):
    journal = MigrationJournal(tmp_path / "journal.sqlite3")
    seeded = journal.seed((candidate("doc-1", 1),), now=1.0)[0]
    first = claim_shadow_build_task(
        journal,
        owner_id="alice",
        worker_id="one",
        lease_seconds=5,
        max_attempts=2,
        now=2.0,
    )
    assert first.attempt == 1
    second = claim_shadow_build_task(
        journal,
        owner_id="alice",
        worker_id="two",
        lease_seconds=5,
        max_attempts=2,
        now=8.0,
    )
    assert second.task_id == seeded.task_id
    assert second.attempt == 2
    assert claim_shadow_build_task(
        journal,
        owner_id="alice",
        worker_id="three",
        lease_seconds=5,
        max_attempts=2,
        now=20.0,
    ) is None


def test_execute_next_builds_and_validates_without_cutover(tmp_path, monkeypatch):
    journal = MigrationJournal(tmp_path / "journal.sqlite3")
    seeded = journal.seed((candidate("doc-1", 1),), now=1.0)[0]
    shadows = MigrationShadowStore(tmp_path / "shadows")
    generations = SimpleNamespace(
        current=lambda **kwargs: SimpleNamespace(
            sequence=1,
            state="active",
            profile_fingerprint=SOURCE,
            content_sha256=CONTENT,
        )
    )
    monkeypatch.setattr(
        "tools.sparse_runtime.get_generation_store",
        lambda: generations,
    )
    result = execute_next_shadow_build(
        owner_id="alice",
        worker_id="builder",
        lease_seconds=30,
        journal=journal,
        shadows=shadows,
        builder=lambda task: build(),
        now=2.0,
    )
    assert result.outcome == "validated"
    assert journal.get(seeded.task_id).state == "validated"
    assert shadows.validate(seeded.task_id).validation_digest == result.validation_digest
    assert execute_next_shadow_build(
        owner_id="alice",
        worker_id="builder",
        journal=journal,
        shadows=shadows,
        builder=lambda task: build(),
        now=3.0,
    ) is None


def test_shadow_store_runtime_cache_is_path_scoped(tmp_path):
    clear_migration_shadow_store_cache()
    first = get_migration_shadow_store(tmp_path / "one")
    second = get_migration_shadow_store(tmp_path / "one")
    third = get_migration_shadow_store(tmp_path / "two")
    assert first is second
    assert first is not third
