from __future__ import annotations

import json

import pytest

from tools import migration_benchmark_cli as cli

TASK = "a" * 64
VALIDATION = "b" * 64
CONTENT = "c" * 64


def fixture_payload():
    def case(
        query,
        current,
        shadow,
        relevant,
        should=False,
        current_abstained=False,
        shadow_abstained=False,
    ):
        return {
            "query_id": query,
            "relevant_ids": relevant,
            "current_ranked_ids": current,
            "shadow_ranked_ids": shadow,
            "support_total": 1 if relevant else 0,
            "current_support_found": 0,
            "shadow_support_found": 1 if relevant else 0,
            "current_citation_count": 1 if relevant else 0,
            "current_valid_citation_count": 1 if relevant else 0,
            "shadow_citation_count": 1 if relevant else 0,
            "shadow_valid_citation_count": 1 if relevant else 0,
            "should_abstain": should,
            "current_abstained": current_abstained,
            "shadow_abstained": shadow_abstained,
        }

    runs = []
    for seed in (1, 2, 3):
        runs.append(
            {
                "seed": seed,
                "cases": [
                    case("q1", ["d2"], ["d1"], ["d1"]),
                    case("q2", [], [], [], True, False, True),
                ],
                "current_resources": {
                    "p95_latency_ms": 100,
                    "peak_memory_bytes": 1000,
                    "index_bytes": 2000,
                    "estimated_cost_units": 10,
                },
                "shadow_resources": {
                    "p95_latency_ms": 120,
                    "peak_memory_bytes": 1100,
                    "index_bytes": 2200,
                    "estimated_cost_units": 11,
                },
            }
        )
    return {
        "task_id": TASK,
        "validation_digest": VALIDATION,
        "source_sequence": 4,
        "source_content_sha256": CONTENT,
        "vector_count": 4,
        "sparse_count": 4,
        "rank_cutoff": 2,
        "runs": runs,
    }


def parse(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_run_writes_promotion_evidence_and_optional_report(tmp_path, capsys):
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(fixture_payload()))
    evidence = tmp_path / "evidence.json"
    report = tmp_path / "report.json"
    assert (
        cli.main(
            [
                "run",
                "--fixture-file",
                str(fixture),
                "--evidence-output",
                str(evidence),
                "--report-output",
                str(report),
            ]
        )
        == 0
    )
    output, error = parse(capsys)
    assert error is None
    assert output["query_count"] == 2
    assert output["repeated_runs"] == 3
    payload = json.loads(evidence.read_text())
    assert payload["task_id"] == TASK
    assert (
        payload["shadow_quality"]["recall_at_k"]
        > payload["current_quality"]["recall_at_k"]
    )
    assert "raw_query" not in evidence.read_text()
    detailed = json.loads(report.read_text())
    assert detailed["delta_intervals"]["recall_at_k"]["mean"] > 0


def test_inspect_prints_contract_only(tmp_path, capsys):
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(fixture_payload()))
    assert cli.main(["inspect", "--fixture-file", str(fixture)]) == 0
    output, error = parse(capsys)
    assert error is None
    assert set(output) == {
        "task_id",
        "benchmark_fingerprint",
        "rank_cutoff",
        "query_count",
        "repeated_runs",
        "seed_count",
    }


def test_duplicate_keys_unknown_fields_and_raw_queries_are_refused(
    tmp_path, capsys
):
    fixture = tmp_path / "fixture.json"
    fixture.write_text('{"task_id":"x","task_id":"y"}')
    assert cli.main(["inspect", "--fixture-file", str(fixture)]) == 2
    _output, error = parse(capsys)
    assert error == {"error": "invalid_or_unavailable"}

    payload = fixture_payload()
    payload["raw_query"] = "private"
    fixture.write_text(json.dumps(payload))
    assert cli.main(["inspect", "--fixture-file", str(fixture)]) == 2
    _output, error = parse(capsys)
    assert error == {"error": "invalid_or_unavailable"}


def test_symlink_fixture_and_output_are_refused(tmp_path, capsys):
    target = tmp_path / "fixture.json"
    target.write_text(json.dumps(fixture_payload()))
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert cli.main(["inspect", "--fixture-file", str(link)]) == 2
    parse(capsys)

    output_target = tmp_path / "real-output.json"
    output_target.write_text("{}")
    output_link = tmp_path / "output.json"
    output_link.symlink_to(output_target)
    assert (
        cli.main(
            [
                "run",
                "--fixture-file",
                str(target),
                "--evidence-output",
                str(output_link),
            ]
        )
        == 2
    )
    _output, error = parse(capsys)
    assert error == {"error": "invalid_or_unavailable"}


def test_invalid_contract_or_parent_is_generic(tmp_path, capsys):
    payload = fixture_payload()
    payload["runs"][1]["cases"][0]["relevant_ids"] = ["other"]
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(payload))
    assert (
        cli.main(
            [
                "run",
                "--fixture-file",
                str(fixture),
                "--evidence-output",
                str(tmp_path / "out.json"),
            ]
        )
        == 2
    )
    _output, error = parse(capsys)
    assert error == {"error": "invalid_or_unavailable"}
