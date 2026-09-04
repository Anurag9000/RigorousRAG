from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.authoritative_classical_training_cli_v3 import SCHEMA, run_config

SOURCE = "d" * 40


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
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
    contract = tmp_path / "calibration-contract.json"
    dense = tmp_path / "dense-calibration.json"
    sparse = tmp_path / "sparse-calibration.json"
    _write_json(contract, {"schema": "test-calibration-contract/v1", "profiles": ["dense", "sparse"]})
    _write_json(dense, {"schema": "test-calibration-artifact/v1", "profile": "dense", "temperature": 1.1})
    _write_json(sparse, {"schema": "test-calibration-artifact/v1", "profile": "sparse", "temperature": 0.9})
    return train, validation, contract, dense, sparse


def _config(tmp_path: Path, *, expected: bool = False) -> Path:
    _, _, contract, dense, sparse = _fixture(tmp_path)
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "kind": "fusion_weight",
        "source_revision": SOURCE,
        "train_data": "train.jsonl",
        "validation_data": "validation.jsonl",
        "output_dir": "out/fusion",
        "profile_ids": ["dense", "sparse"],
        "calibration_contract": contract.name,
        "calibration_artifacts": {"dense": dense.name, "sparse": sparse.name},
        "training": {"epochs": 3, "batch_size": 1, "patience": 2, "learning_rate": 0.05},
    }
    if expected:
        payload["calibration_contract_sha256"] = _sha(contract)
        payload["calibration_artifact_sha256s"] = {"dense": _sha(dense), "sparse": _sha(sparse)}
    path = tmp_path / "fusion.json"
    _write_json(path, payload)
    return path


def test_real_calibration_bytes_are_bound_and_repeatable(tmp_path: Path) -> None:
    config = _config(tmp_path, expected=True)
    first = run_config(config)
    second = run_config(config)

    identity = first["input_artifacts"]
    assert first["source_revision"] == SOURCE
    assert first["result_sha256"] == second["result_sha256"]
    assert identity["contract"]["sha256"] == _sha(tmp_path / "calibration-contract.json")
    assert identity["artifacts"]["dense"]["sha256"] == _sha(tmp_path / "dense-calibration.json")
    assert identity["artifacts"]["sparse"]["sha256"] == _sha(tmp_path / "sparse-calibration.json")
    assert len(identity["identity_sha256"]) == 64


def test_missing_calibration_artifact_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["calibration_artifacts"]["dense"] = "missing.json"
    _write_json(config, payload)

    with pytest.raises(FileNotFoundError):
        run_config(config)


def test_placeholder_expected_digest_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["calibration_contract_sha256"] = "0" * 64
    _write_json(config, payload)

    with pytest.raises(ValueError, match="all-zero placeholder"):
        run_config(config)


def test_declared_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["calibration_artifact_sha256s"] = {"dense": "a" * 64, "sparse": _sha(tmp_path / "sparse-calibration.json")}
    _write_json(config, payload)

    with pytest.raises(ValueError, match="digest mismatch"):
        run_config(config)


def test_calibration_mutation_refuses_incompatible_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_config(config)
    _write_json(
        tmp_path / "dense-calibration.json",
        {"schema": "test-calibration-artifact/v1", "profile": "dense", "temperature": 99.0},
    )

    with pytest.raises(ValueError, match="different spec/config/data identity"):
        run_config(config)


def test_profile_artifact_coverage_must_be_exact(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["calibration_artifacts"] = {"dense": "dense-calibration.json"}
    _write_json(config, payload)

    with pytest.raises(ValueError, match="cover profile_ids exactly"):
        run_config(config)


def test_profile_ids_may_not_be_bare_string(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["profile_ids"] = "dense"
    payload["calibration_artifacts"] = {"d": "dense-calibration.json", "e": "dense-calibration.json", "n": "dense-calibration.json", "s": "sparse-calibration.json"}
    _write_json(config, payload)

    with pytest.raises(ValueError, match="must be an array"):
        run_config(config)


def test_calibration_artifact_keys_must_be_strings(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    # JSON object keys are always strings, so exercise the stricter helper indirectly via
    # an invalid normalized key that cannot be silently coerced into a profile id.
    payload["calibration_artifacts"] = {" dense ": "dense-calibration.json", "sparse": "sparse-calibration.json"}
    _write_json(config, payload)

    with pytest.raises(ValueError, match="cover profile_ids exactly"):
        run_config(config)


def test_zero_source_revision_is_not_an_alias_for_auto(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["source_revision"] = "0" * 40
    _write_json(config, payload)

    with pytest.raises(ValueError, match="all-zero placeholder"):
        run_config(config)
