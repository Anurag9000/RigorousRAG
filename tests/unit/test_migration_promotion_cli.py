from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools import migration_promotion_cli as cli
from tools.migration_promotion_store import MigrationPromotionStore

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64


def quality(recall=0.8):
    return {
        "query_count": 100,
        "recall_at_k": recall,
        "ndcg_at_k": 0.75,
        "mrr": 0.72,
        "support_recall": 0.82,
        "citation_precision": 0.96,
        "abstention_accuracy": 0.9,
    }


def resources(latency=100.0):
    return {
        "p95_latency_ms": latency,
        "peak_memory_bytes": 1000,
        "index_bytes": 2000,
        "estimated_cost_units": 10.0,
    }


def evidence(recall=0.81):
    return {
        "task_id": E,
        "validation_digest": D,
        "benchmark_fingerprint": B,
        "source_sequence": 4,
        "source_content_sha256": C,
        "vector_count": 3,
        "sparse_count": 3,
        "repeated_runs": 5,
        "seed_count": 5,
        "confidence_interval_level": 0.95,
        "current_quality": quality(),
        "shadow_quality": quality(recall),
        "current_resources": resources(),
        "shadow_resources": resources(120.0),
    }


def fixtures(state="validated"):
    task = SimpleNamespace(
        task_id=E,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=4,
        source_profile_fingerprint=A,
        target_profile_fingerprint=B,
        validation_digest=D,
        state=state,
    )
    manifest = SimpleNamespace(
        task_id=E,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=4,
        source_profile_fingerprint=A,
        target_profile_fingerprint=B,
        validation_digest=D,
        content_sha256=C,
        vector_count=3,
        sparse_count=3,
    )
    generation = SimpleNamespace(
        sequence=4,
        state="active",
        profile_fingerprint=A,
        content_sha256=C,
    )
    return task, manifest, generation


def install(monkeypatch, tmp_path, state="validated"):
    task, manifest, generation = fixtures(state)
    journal = SimpleNamespace(get=lambda task_id: task if task_id == E else None)
    shadows = SimpleNamespace(validate=lambda task_id: manifest)
    generations = SimpleNamespace(current=lambda **kwargs: generation)
    store = MigrationPromotionStore(tmp_path / "reports")
    monkeypatch.setattr(cli, "get_migration_journal", lambda: journal)
    monkeypatch.setattr(cli, "get_migration_shadow_store", lambda: shadows)
    monkeypatch.setattr(cli, "get_migration_promotion_store", lambda: store)
    import tools.sparse_runtime as sparse_runtime

    monkeypatch.setattr(sparse_runtime, "get_generation_store", lambda: generations)
    return task, manifest, generation, store


def parse(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_evaluate_persists_eligible_report_without_paths(
    tmp_path, monkeypatch, capsys
):
    _task, _manifest, _generation, store = install(monkeypatch, tmp_path)
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence()))
    assert cli.main(["evaluate", E, "--evidence-file", str(path)]) == 0
    output, error = parse(capsys)
    assert error is None
    assert output["decision"] == "eligible"
    assert output["reason_codes"] == []
    assert "source_path" not in json.dumps(output)
    assert store.read(E).report_digest == output["report_digest"]


def test_blocked_evaluation_returns_one_and_history_status_work(
    tmp_path, monkeypatch, capsys
):
    install(monkeypatch, tmp_path)
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence(0.4)))
    assert cli.main(["evaluate", E, "--evidence-file", str(path)]) == 1
    output, _error = parse(capsys)
    assert output["decision"] == "blocked"
    assert "recall_at_k_below_floor" in output["reason_codes"]
    assert cli.main(["status", E]) == 0
    status, _error = parse(capsys)
    assert status["report_digest"] == output["report_digest"]
    assert cli.main(["history", E, "--limit", "10"]) == 0
    history, _error = parse(capsys)
    assert history["count"] == 1


def test_strict_json_duplicate_keys_and_unknown_fields_fail_generic(
    tmp_path, monkeypatch, capsys
):
    install(monkeypatch, tmp_path)
    path = tmp_path / "bad.json"
    path.write_text('{"task_id":"x","task_id":"y"}')
    assert cli.main(["evaluate", E, "--evidence-file", str(path)]) == 2
    _output, error = parse(capsys)
    assert error == {"error": "invalid_or_unavailable"}

    payload = evidence()
    payload["raw_query"] = "private"
    path.write_text(json.dumps(payload))
    assert cli.main(["evaluate", E, "--evidence-file", str(path)]) == 2
    _output, error = parse(capsys)
    assert error == {"error": "invalid_or_unavailable"}


def test_policy_override_can_block_resource_ratio(tmp_path, monkeypatch, capsys):
    install(monkeypatch, tmp_path)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence()))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"max_latency_ratio": 1.1}))
    assert (
        cli.main(
            [
                "evaluate",
                E,
                "--evidence-file",
                str(evidence_path),
                "--policy-file",
                str(policy),
            ]
        )
        == 1
    )
    output, _error = parse(capsys)
    assert "p95_latency_ms_ratio_exceeds_limit" in output["reason_codes"]


def test_remove_requires_exact_confirmation_and_failed_state(
    tmp_path, monkeypatch, capsys
):
    task, _manifest, _generation, _store = install(monkeypatch, tmp_path)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence()))
    assert cli.main(["evaluate", E, "--evidence-file", str(evidence_path)]) == 0
    parse(capsys)
    assert cli.main(["remove-task", E, "--confirm-task-id", E]) == 2
    _output, error = parse(capsys)
    assert error == {"error": "invalid_or_unavailable"}
    task.state = "failed"
    assert cli.main(["remove-task", E, "--confirm-task-id", "f" * 64]) == 2
    parse(capsys)
    assert cli.main(["remove-task", E, "--confirm-task-id", E]) == 0
    output, error = parse(capsys)
    assert error is None and output["removed"] is True


def test_symlink_evidence_file_is_refused(tmp_path, monkeypatch, capsys):
    install(monkeypatch, tmp_path)
    target = tmp_path / "evidence.json"
    target.write_text(json.dumps(evidence()))
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert cli.main(["evaluate", E, "--evidence-file", str(link)]) == 2
    _output, error = parse(capsys)
    assert error == {"error": "invalid_or_unavailable"}


def test_missing_task_and_report_are_bounded_not_found(
    tmp_path, monkeypatch, capsys
):
    _task, _manifest, _generation, _store = install(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli,
        "get_migration_journal",
        lambda: SimpleNamespace(get=lambda _task_id: None),
    )
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence()))
    assert cli.main(["evaluate", E, "--evidence-file", str(path)]) == 1
    _output, error = parse(capsys)
    assert error == {"error": "not_found", "task_id": E}
    assert cli.main(["status", E]) == 1
    _output, error = parse(capsys)
    assert error == {"error": "not_found", "task_id": E}
