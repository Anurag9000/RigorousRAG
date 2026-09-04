from __future__ import annotations

import json
from pathlib import Path

from training.authoritative_classical_training_cli import SCHEMA, run_config


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_fusion_training_is_content_bound_and_repeatable_from_state(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    rows = [
        {"probabilities": {"dense": 0.9, "sparse": 0.2}, "relevant": True},
        {"probabilities": {"dense": 0.2, "sparse": 0.8}, "relevant": False},
        {"probabilities": {"dense": 0.7, "sparse": 0.4}, "relevant": True},
        {"probabilities": {"dense": 0.3, "sparse": 0.7}, "relevant": False},
    ]
    _write_jsonl(train, rows)
    _write_jsonl(validation, rows[:2])
    config = tmp_path / "fusion.json"
    _write_json(
        config,
        {
            "schema": SCHEMA,
            "kind": "fusion_weight",
            "train_data": "train.jsonl",
            "validation_data": "validation.jsonl",
            "output_dir": "out/fusion",
            "profile_ids": ["dense", "sparse"],
            "calibration_contract_sha256": "a" * 64,
            "calibration_artifact_sha256s": {"dense": "b" * 64, "sparse": "c" * 64},
            "source_revision": "d" * 40,
            "training": {"epochs": 3, "batch_size": 1, "patience": 2, "learning_rate": 0.05},
        },
    )

    first = run_config(config)
    second = run_config(config)

    assert first["result_sha256"] == second["result_sha256"]
    assert first["result"]["artifact"]["artifact_sha256"] == second["result"]["artifact"]["artifact_sha256"]
    assert (tmp_path / "out/fusion/state/fusion-weight-latest.json").is_file()
    assert (tmp_path / "out/fusion/training_result.json").is_file()


def test_listwise_training_resumes_from_content_addressed_state(tmp_path: Path) -> None:
    train = tmp_path / "listwise-train.jsonl"
    validation = tmp_path / "listwise-validation.jsonl"
    query = {
        "query_sha256": "1" * 64,
        "candidates": [
            {"candidate_id": "good", "probabilities": {"dense": 0.9, "sparse": 0.5}, "relevance_grade": 2.0},
            {"candidate_id": "bad", "probabilities": {"dense": 0.2, "sparse": 0.6}, "relevance_grade": 0.0},
        ],
    }
    _write_jsonl(train, [query, {**query, "query_sha256": "2" * 64}])
    _write_jsonl(validation, [{**query, "query_sha256": "3" * 64}])
    config = tmp_path / "listwise.json"
    _write_json(
        config,
        {
            "schema": SCHEMA,
            "kind": "listwise_fusion",
            "train_data": "listwise-train.jsonl",
            "validation_data": "listwise-validation.jsonl",
            "output_dir": "out/listwise",
            "profile_ids": ["dense", "sparse"],
            "calibration_contract_sha256": "a" * 64,
            "calibration_artifact_sha256s": {"dense": "b" * 64, "sparse": "c" * 64},
            "source_revision": "d" * 40,
            "training": {"epochs": 3, "batch_size": 1, "patience": 2, "learning_rate": 0.05},
        },
    )

    first = run_config(config)
    second = run_config(config)

    assert first["result_sha256"] == second["result_sha256"]
    assert first["result"]["artifact"]["artifact_sha256"] == second["result"]["artifact"]["artifact_sha256"]
    assert (tmp_path / "out/listwise/state/listwise-fusion-latest.json").is_file()


def test_domain_classifier_reuses_exact_resume_store(tmp_path: Path) -> None:
    train = tmp_path / "domain-train.jsonl"
    validation = tmp_path / "domain-validation.jsonl"
    rows = [
        {"features": {"schema": ["length", "math"], "values": [1.0, 0.0]}, "label": "short"},
        {"features": {"schema": ["length", "math"], "values": [0.0, 1.0]}, "label": "math"},
        {"features": {"schema": ["length", "math"], "values": [0.8, 0.1]}, "label": "short"},
        {"features": {"schema": ["length", "math"], "values": [0.1, 0.8]}, "label": "math"},
    ]
    _write_jsonl(train, rows)
    _write_jsonl(validation, rows[:2])
    config = tmp_path / "domain.json"
    _write_json(
        config,
        {
            "schema": SCHEMA,
            "kind": "domain_classifier",
            "train_data": "domain-train.jsonl",
            "validation_data": "domain-validation.jsonl",
            "output_dir": "out/domain",
            "labels": ["short", "math"],
            "fallback_label": "short",
            "training": {"epochs": 3, "batch_size": 1, "patience": 2, "learning_rate": 0.05},
        },
    )

    first = run_config(config)
    second = run_config(config)

    assert first["result_sha256"] == second["result_sha256"]
    assert first["result"]["artifact"] == second["result"]["artifact"]
    assert (tmp_path / "out/domain/state/domain-classifier-latest.json").is_file()


def test_plan_ranker_reuses_exact_resume_store(tmp_path: Path) -> None:
    train = tmp_path / "plan-train.jsonl"
    validation = tmp_path / "plan-validation.jsonl"
    rows = [
        {
            "preferred_features": {"schema": ["quality", "cost"], "values": [1.0, 0.1]},
            "rejected_features": {"schema": ["quality", "cost"], "values": [0.2, 0.9]},
            "weight": 1.0,
        },
        {
            "preferred_features": {"schema": ["quality", "cost"], "values": [0.9, 0.2]},
            "rejected_features": {"schema": ["quality", "cost"], "values": [0.3, 0.8]},
            "weight": 1.0,
        },
    ]
    _write_jsonl(train, rows)
    _write_jsonl(validation, [rows[0]])
    config = tmp_path / "plan.json"
    _write_json(
        config,
        {
            "schema": SCHEMA,
            "kind": "plan_ranker",
            "train_data": "plan-train.jsonl",
            "validation_data": "plan-validation.jsonl",
            "output_dir": "out/plan",
            "latency_penalty": 0.1,
            "cost_penalty": 0.2,
            "risk_penalty": 0.3,
            "training": {"epochs": 3, "batch_size": 1, "patience": 2, "learning_rate": 0.05},
        },
    )

    first = run_config(config)
    second = run_config(config)

    assert first["result_sha256"] == second["result_sha256"]
    assert first["result"]["artifact"] == second["result"]["artifact"]
    assert (tmp_path / "out/plan/state/plan-ranker-latest.json").is_file()
