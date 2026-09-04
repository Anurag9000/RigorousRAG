from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from training.authoritative_retrieval_training_cli_v2 import SCHEMA, _preflight


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _fixture(tmp_path: Path) -> Path:
    _text(tmp_path / "train.jsonl", '{"query":"q","positive":"p"}\n')
    _text(tmp_path / "validation.jsonl", '{"query":"vq","positive":"vp"}\n')
    _text(tmp_path / "model" / "config.json", '{"hidden_size":8}\n')
    _text(tmp_path / "model" / "weights.bin", "model-bytes\n")
    _text(tmp_path / "tokenizer" / "tokenizer.json", '{"version":1}\n')
    config = tmp_path / "retrieval.json"
    _json(
        config,
        {
            "schema": SCHEMA,
            "architecture": "dense",
            "source_commit": "auto",
            "train_data": "train.jsonl",
            "validation_data": "validation.jsonl",
            "model_root": "model",
            "tokenizer_root": "tokenizer",
            "output_dir": "out",
            "model": {},
        },
    )
    return config


def _with_expected(config: Path) -> dict[str, str]:
    actual = dict(_preflight(config)["actual_inputs"])
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["expected_inputs"] = actual
    _json(config, payload)
    return actual


def test_present_bytes_are_hashed_before_heavy_training_imports(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    first = _preflight(config)
    second = _preflight(config)
    assert first["actual_inputs"] == second["actual_inputs"]
    assert first["expected_inputs_verified"] is False
    assert set(first["actual_inputs"]) == {
        "train_data_sha256",
        "validation_data_sha256",
        "model_tree_sha256",
        "tokenizer_tree_sha256",
    }
    assert all(len(value) == 64 for value in first["actual_inputs"].values())


def test_closed_world_expected_contract_accepts_exact_bytes(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    expected = _with_expected(config)
    result = _preflight(config)
    assert result["expected_inputs_verified"] is True
    assert result["actual_inputs"] == expected


def test_all_zero_source_commit_is_rejected(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["source_commit"] = "0" * 40
    _json(config, payload)
    with pytest.raises(ValueError, match="all-zero placeholder"):
        _preflight(config)


def test_expected_contract_must_cover_every_admitted_input(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    expected = _with_expected(config)
    expected.pop("tokenizer_tree_sha256")
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["expected_inputs"] = expected
    _json(config, payload)
    with pytest.raises(ValueError, match="cover admitted retrieval inputs exactly"):
        _preflight(config)


def test_zero_expected_digest_is_rejected(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    expected = _with_expected(config)
    expected["train_data_sha256"] = "0" * 64
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["expected_inputs"] = expected
    _json(config, payload)
    with pytest.raises(ValueError, match="all-zero placeholder"):
        _preflight(config)


def test_input_mutation_breaks_expected_contract(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    _with_expected(config)
    _text(tmp_path / "model" / "weights.bin", "mutated-model-bytes\n")
    with pytest.raises(ValueError, match="retrieval input digest mismatch"):
        _preflight(config)


def test_untied_document_model_is_part_of_closed_world_contract(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    _text(tmp_path / "document-model" / "weights.bin", "document-model\n")
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["model"] = {"untied_document_model_root": "document-model"}
    _json(config, payload)
    actual = dict(_preflight(config)["actual_inputs"])
    assert "untied_document_model_tree_sha256" in actual
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["expected_inputs"] = {key: value for key, value in actual.items() if key != "untied_document_model_tree_sha256"}
    _json(config, payload)
    with pytest.raises(ValueError, match="cover admitted retrieval inputs exactly"):
        _preflight(config)


def test_symlink_inside_model_tree_fails_closed(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    target = tmp_path / "external.bin"
    _text(target, "external\n")
    link = tmp_path / "model" / "linked.bin"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="contains a symlink"):
        _preflight(config)
