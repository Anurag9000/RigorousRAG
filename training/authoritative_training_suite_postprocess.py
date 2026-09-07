#!/usr/bin/env python3
"""Artifact-bound validation, readiness testing and metrics materialization for training suites.

This module is intentionally post-training only. It never optimizes model parameters and it
never invents a held-out dataset. The authoritative trainers remain responsible for their
native validation loops and early stopping. This layer gives the central DAG stable executable
post-training phases by verifying the trainers' persisted result/checkpoint contracts and by
materializing a normalized metrics index from the native result payload.

``validate`` verifies result identity against the exact config bytes and, for advanced RAG,
re-runs the existing authoritative config validation and verifies the best/latest checkpoint.
``test`` performs artifact/readiness verification. It is deliberately not mislabeled as a
held-out scientific benchmark: benchmark/test jobs that require external governed cohorts are
supplied separately through the suite lifecycle manifest.
``metrics`` extracts every finite numeric metric already emitted by the native trainer and writes
one atomic, content-addressed metrics manifest. It does not replace domain-native metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

RESULT_SCHEMA = "rigorousrag-training-suite-postprocess/v1"
METRICS_SCHEMA = "rigorousrag-training-suite-metrics/v1"
_FAMILIES = frozenset({"classical", "retrieval", "advanced"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.expanduser().absolute()
    parts = absolute.parts
    if not parts:
        raise ValueError(f"{label} path is invalid")
    current = Path(parts[0])
    if current.is_symlink():
        raise ValueError(f"{label} traverses a symlink: {current}")
    for part in parts[1:]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"{label} traverses a symlink: {current}")


def _regular(path: str | Path, label: str) -> Path:
    raw = Path(path).expanduser()
    _reject_symlink_components(raw, label)
    selected = raw.resolve(strict=True)
    if not selected.is_file() or selected.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return selected


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.stat().st_size <= 0 or path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError(f"{label} exceeds the JSON safety bound")
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _reject_symlink_components(path, "output")
    path = path.expanduser().resolve(strict=False)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise ValueError("output must be a regular file path")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except Exception:
            directory = None
        if directory is not None:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _family(value: str) -> str:
    selected = str(value).strip().lower()
    if selected not in _FAMILIES:
        raise ValueError(f"family must be one of {sorted(_FAMILIES)}")
    return selected


def _config_and_result(family: str, config_path: str | Path) -> tuple[Path, Mapping[str, Any], Path, Mapping[str, Any]]:
    config = _regular(config_path, "training config")
    raw = _read_json(config, "training config")
    if family in {"classical", "retrieval"}:
        output = raw.get("output_dir")
        if not isinstance(output, str) or not output.strip():
            raise ValueError(f"{family} config must declare output_dir")
        candidate = Path(output).expanduser()
        root = candidate if candidate.is_absolute() else config.parent / candidate
    else:
        # Advanced-RAG path semantics are authoritative in advanced_rag_config:
        # relative checkpoint roots resolve from the repository working directory,
        # not from the configuration file directory. Reuse that loader rather than
        # duplicating or subtly changing its path contract here.
        from training.advanced_rag_config import load_advanced_run_config

        configured = load_advanced_run_config(config)
        root = Path(configured.checkpoint_root)
    result_path = root.resolve(strict=False) / "training_result.json"
    result = _read_json(_regular(result_path, "training result"), "training result")
    return config, raw, result_path.resolve(strict=True), result


def _verify_self_digest(result: Mapping[str, Any]) -> None:
    expected = result.get("result_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("training result does not contain a SHA-256 result identity")
    unsigned = dict(result)
    unsigned.pop("result_sha256", None)
    actual = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if expected.lower() != actual:
        raise ValueError(f"training result digest mismatch: expected {expected}, actual {actual}")


def _verify_result_identity(family: str, config: Path, result: Mapping[str, Any]) -> Mapping[str, Any]:
    _verify_self_digest(result)
    config_sha = _sha_file(config)
    observed = str(result.get("config_sha256") or "").lower()
    if observed != config_sha:
        raise ValueError(f"training result config identity mismatch: expected {config_sha}, got {observed}")
    complete = result.get("complete")
    if family in {"retrieval", "advanced"} and complete is not True:
        raise ValueError(f"{family} training result is not marked complete")
    expected_schema = {
        "classical": "rigorousrag-authoritative-classical-training-result/v3",
        "retrieval": "rigorousrag-authoritative-retrieval-training-result/v1",
        "advanced": "rigorousrag-authoritative-advanced-training-result/v1",
    }[family]
    if result.get("schema") != expected_schema:
        raise ValueError(f"unexpected {family} training result schema {result.get('schema')!r}")
    return {"schema": expected_schema, "config_sha256": config_sha, "result_sha256": str(result["result_sha256"]).lower()}


def _advanced_checkpoint(config_path: Path) -> Mapping[str, Any]:
    from training.advanced_checkpoint_authority import AdvancedCheckpointManager
    from training.advanced_rag_config import load_advanced_run_config
    from training.advanced_rag_operator import validate_config, verify_checkpoint_from_config

    configured = load_advanced_run_config(config_path)
    validation = validate_config(config_path, load_models=False)
    manager = AdvancedCheckpointManager(configured.checkpoint_root)
    pointer = "best" if (manager.root / "best.json").is_file() else "latest"
    if not (manager.root / f"{pointer}.json").is_file():
        raise ValueError("advanced training has neither a best nor latest checkpoint pointer")
    digest = manager.resolve_pointer(pointer)
    binding = verify_checkpoint_from_config(config_path, checkpoint_digest=digest)
    return {
        "validation": validation,
        "checkpoint_pointer": pointer,
        "checkpoint_digest": digest,
        "checkpoint_binding": {
            key: getattr(binding, key)
            for key in (
                "kind", "checkpoint_digest", "plan_sha256", "training_input_sha256",
                "training_config_sha256", "source_commit"
            )
            if hasattr(binding, key)
        },
    }


def _retrieval_checkpoint(config: Path, raw: Mapping[str, Any]) -> Mapping[str, Any]:
    from training.checkpointing import CheckpointManager

    output = Path(str(raw["output_dir"])).expanduser()
    output = output if output.is_absolute() else config.parent / output
    manager = CheckpointManager(output.resolve(strict=False) / "checkpoints")
    pointer = "best" if (manager.root / "best.json").is_file() else "latest"
    if not (manager.root / f"{pointer}.json").is_file():
        raise ValueError("retrieval training has neither a best nor latest checkpoint pointer")
    digest = manager.resolve_pointer(pointer)
    path, manifest = manager.verify(digest)
    return {
        "checkpoint_digest": digest,
        "checkpoint_path": str(path),
        "checkpoint_run_id": manifest.run_id,
        "checkpoint_source_commit": manifest.source_commit,
        "checkpoint_training_config_digest": manifest.training_config_digest,
        "checkpoint_dataset_manifest_digest": manifest.dataset_manifest_digest,
        "checkpoint_metric_snapshot": dict(manifest.metric_snapshot),
    }


def _finite_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_finite_metrics(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            result.update(_finite_metrics(item, child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            result[prefix or "value"] = number
    return result


def validate_result(family: str, config_path: str | Path, training_job_id: str) -> Mapping[str, Any]:
    selected_family = _family(family)
    config, raw, result_path, result = _config_and_result(selected_family, config_path)
    identity = _verify_result_identity(selected_family, config, result)
    family_validation: Mapping[str, Any] = {}
    if selected_family == "advanced":
        family_validation = _advanced_checkpoint(config)
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "operation": "validate",
        "training_job_id": training_job_id,
        "family": selected_family,
        "config": str(config),
        "training_result": str(result_path),
        "identity": identity,
        "family_validation": family_validation,
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def test_artifact(family: str, config_path: str | Path, training_job_id: str) -> Mapping[str, Any]:
    selected_family = _family(family)
    config, raw, result_path, result = _config_and_result(selected_family, config_path)
    identity = _verify_result_identity(selected_family, config, result)
    if selected_family == "advanced":
        evidence: Mapping[str, Any] = _advanced_checkpoint(config)
    elif selected_family == "retrieval":
        evidence = _retrieval_checkpoint(config, raw)
    else:
        evidence = {"transactional_result_verified": True, "result_sha256": identity["result_sha256"]}
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "operation": "artifact-test",
        "scientific_test_dataset_used": False,
        "training_job_id": training_job_id,
        "family": selected_family,
        "config": str(config),
        "training_result": str(result_path),
        "identity": identity,
        "artifact_evidence": evidence,
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def metrics_result(family: str, config_path: str | Path, training_job_id: str) -> Mapping[str, Any]:
    selected_family = _family(family)
    config, _, result_path, result = _config_and_result(selected_family, config_path)
    identity = _verify_result_identity(selected_family, config, result)
    metrics = _finite_metrics(result)
    payload: dict[str, Any] = {
        "schema": METRICS_SCHEMA,
        "training_job_id": training_job_id,
        "family": selected_family,
        "config": str(config),
        "training_result": str(result_path),
        "identity": identity,
        "metric_count": len(metrics),
        "metrics": dict(sorted(metrics.items())),
    }
    payload["metrics_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for name in ("validate", "test", "metrics"):
        command = sub.add_parser(name)
        command.add_argument("--family", required=True, choices=sorted(_FAMILIES))
        command.add_argument("--config", required=True)
        command.add_argument("--training-job-id", required=True)
        command.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation == "validate":
        payload = validate_result(args.family, args.config, args.training_job_id)
    elif args.operation == "test":
        payload = test_artifact(args.family, args.config, args.training_job_id)
    elif args.operation == "metrics":
        payload = metrics_result(args.family, args.config, args.training_job_id)
    else:  # pragma: no cover
        raise RuntimeError("unreachable operation")
    _atomic_json(Path(args.output), payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["METRICS_SCHEMA", "RESULT_SCHEMA", "main", "metrics_result", "test_artifact", "validate_result"]
