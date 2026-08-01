import json
from types import SimpleNamespace

from tools import migration_cutover_preflight_cli as cli
from tools.migration_cutover_preflight_store import MigrationCutoverPreflightStore
from tools.migration_promotion import PromotionReport

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64
F = "f" * 64


def promotion():
    return PromotionReport(
        task_id=E,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=4,
        source_profile_fingerprint=A,
        target_profile_fingerprint=B,
        validation_digest=D,
        benchmark_fingerprint=F,
        evidence_digest="1" * 64,
        policy_id="paired-promotion-v1",
        policy_digest="2" * 64,
        decision="eligible",
        reason_codes=(),
        quality_deltas={
            name: 0.0
            for name in (
                "recall_at_k",
                "ndcg_at_k",
                "mrr",
                "support_recall",
                "citation_precision",
                "abstention_accuracy",
            )
        },
        resource_ratios={
            "p95_latency_ms": 1.0,
            "peak_memory_bytes": 1.0,
            "index_bytes": 1.0,
            "estimated_cost_units": 1.0,
        },
        evaluated_at=1,
    )


def task(state="validated"):
    return SimpleNamespace(
        task_id=E,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=4,
        source_profile_fingerprint=A,
        target_profile_fingerprint=B,
        validation_digest=D,
        state=state,
    )


def manifest():
    return SimpleNamespace(
        task_id=E,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=4,
        source_profile_fingerprint=A,
        target_profile_fingerprint=B,
        validation_digest=D,
        content_sha256=C,
        vector_count=1,
        sparse_count=1,
        vector_sha256="3" * 64,
        sparse_sha256="4" * 64,
    )


def snapshot():
    vector = SimpleNamespace(
        owner_id="alice",
        doc_id="doc-1",
        ids=("v1",),
        documents=("one",),
        metadatas=({"owner_id": "alice", "doc_id": "doc-1"},),
    )
    field = SimpleNamespace(
        field_id="f1",
        field_type="body",
        text="one",
        position=0,
        token_count=1,
        page_number=1,
        section="A",
        metadata={},
    )
    sparse = SimpleNamespace(
        owner_id="alice",
        doc_id="doc-1",
        generation=7,
        profile_fingerprint=A,
        metadata={},
        fields=(field,),
    )
    generation = SimpleNamespace(
        sequence=4,
        state="active",
        content_sha256=C,
        profile_fingerprint=A,
        vector_rows=1,
        sparse_generation=7,
    )
    return SimpleNamespace(
        owner_id="alice",
        doc_id="doc-1",
        stores=SimpleNamespace(vector=vector, sparse=sparse),
        generation=generation,
    )


def install(monkeypatch, tmp_path, state="validated"):
    current_task = task(state)
    store = MigrationCutoverPreflightStore(tmp_path / "preflights")
    monkeypatch.setattr(
        cli,
        "get_migration_journal",
        lambda: SimpleNamespace(
            get=lambda task_id: current_task if task_id == E else None
        ),
    )
    monkeypatch.setattr(
        cli,
        "get_migration_shadow_store",
        lambda: SimpleNamespace(validate=lambda task_id: manifest()),
    )
    monkeypatch.setattr(
        cli,
        "get_migration_promotion_store",
        lambda: SimpleNamespace(read=lambda task_id: promotion()),
    )
    monkeypatch.setattr(cli, "get_migration_cutover_preflight_store", lambda: store)
    monkeypatch.setattr(cli, "_capture_snapshot", lambda current: snapshot())
    return current_task, store


def parse(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_plan_is_nonmutating_path_free_and_persisted(tmp_path, monkeypatch, capsys):
    _task, store = install(monkeypatch, tmp_path)
    assert cli.main(["plan", E]) == 0
    output, error = parse(capsys)
    assert error is None
    assert output["mutation_performed"] is False
    assert output["preflight_digest"] == store.read(E).preflight_digest
    assert "source_path" not in json.dumps(output)
    assert "one" not in json.dumps(output)


def test_status_history_and_bounded_not_found(tmp_path, monkeypatch, capsys):
    install(monkeypatch, tmp_path)
    assert cli.main(["plan", E]) == 0
    parse(capsys)
    assert cli.main(["status", E]) == 0
    status, _ = parse(capsys)
    assert status["task_id"] == E
    assert cli.main(["history", E]) == 0
    history, _ = parse(capsys)
    assert history["count"] == 1
    assert cli.main(["status", "9" * 64]) == 1
    _output, error = parse(capsys)
    assert error == {"error": "not_found", "task_id": "9" * 64}


def test_remove_requires_failed_or_cancelled_and_exact_confirmation(
    tmp_path, monkeypatch, capsys
):
    current_task, _store = install(monkeypatch, tmp_path)
    assert cli.main(["plan", E]) == 0
    parse(capsys)
    assert cli.main(["remove-task", E, "--confirm-task-id", E]) == 2
    parse(capsys)
    current_task.state = "failed"
    assert cli.main(["remove-task", E, "--confirm-task-id", "9" * 64]) == 2
    parse(capsys)
    assert cli.main(["remove-task", E, "--confirm-task-id", E]) == 0
    output, error = parse(capsys)
    assert error is None and output["removed"] is True


def test_plan_rejects_missing_task_with_bounded_error(tmp_path, monkeypatch, capsys):
    install(monkeypatch, tmp_path)
    assert cli.main(["plan", "9" * 64]) == 1
    _output, error = parse(capsys)
    assert error == {"error": "not_found", "task_id": "9" * 64}
