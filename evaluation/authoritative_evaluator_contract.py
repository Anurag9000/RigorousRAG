"""Content-addressed evaluator contracts for promotion-grade benchmark cohorts.

Promotion evidence already binds an ``evaluator_contract_sha256``.  This module makes that
digest reconstructable instead of operator-invented: exact local config bytes, a full source
revision, implementation identifier and a closed metric schema produce a self-verifying
contract receipt.  No evaluator or model executes on import or verification.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.advanced_path_authority import safe_advanced_path

_MAX_CONFIG_BYTES = 16 * 1024 * 1024
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024
_MAX_METRICS = 10_000
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _commit(value: Any) -> str:
    selected = str(value).strip().lower()
    if len(selected) not in {40, 64} or any(ch not in _HEX for ch in selected):
        raise ValueError("source_commit must be a full Git object id")
    return selected


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if (
        not selected
        or len(selected) > maximum
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected)
    ):
        raise ValueError(f"{label} is invalid")
    return selected


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str, maximum_bytes: int) -> Mapping[str, Any]:
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{label} exceeds byte safety bound")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _atomic(path: Path, payload: bytes) -> None:
    if path.exists():
        raise ValueError("evaluator contract receipt destination must not already exist")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class EvaluatorMetricContract:
    name: str
    family: str
    direction: str
    scope: str
    definition: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "metric name", 300))
        family = _text(self.family, "metric family", 100).lower()
        if family not in {
            "retrieval",
            "generation",
            "citation",
            "grounding",
            "robustness",
            "calibration",
            "latency",
            "cost",
            "safety",
            "other",
        }:
            raise ValueError("unsupported metric family")
        object.__setattr__(self, "family", family)
        direction = _text(self.direction, "metric direction", 20).lower()
        if direction not in {"maximize", "minimize", "descriptive"}:
            raise ValueError("metric direction must be maximize/minimize/descriptive")
        object.__setattr__(self, "direction", direction)
        scope = _text(self.scope, "metric scope", 30).lower()
        if scope not in {"per_sample", "aggregate", "both"}:
            raise ValueError("metric scope must be per_sample/aggregate/both")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(
            self,
            "definition",
            _text(self.definition, "metric definition", 10_000),
        )


@dataclass(frozen=True)
class AuthoritativeEvaluatorContract:
    evaluator_id: str
    implementation_id: str
    source_commit: str
    config_path: str
    config_sha256: str
    metrics: tuple[EvaluatorMetricContract, ...]
    sample_semantics: str
    aggregation_semantics: str
    contract_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluator_id",
            _text(self.evaluator_id, "evaluator_id", 500),
        )
        object.__setattr__(
            self,
            "implementation_id",
            _text(self.implementation_id, "implementation_id", 2_000),
        )
        object.__setattr__(self, "source_commit", _commit(self.source_commit))
        config = safe_advanced_path(
            self.config_path,
            label="evaluator contract config",
            must_exist=True,
            require_file=True,
        )
        object.__setattr__(self, "config_path", str(config))
        object.__setattr__(
            self,
            "config_sha256",
            _sha(self.config_sha256, "config_sha256"),
        )
        metrics = tuple(self.metrics)
        if (
            not metrics
            or len(metrics) > _MAX_METRICS
            or any(not isinstance(item, EvaluatorMetricContract) for item in metrics)
        ):
            raise ValueError("metrics must be a bounded non-empty metric-contract sequence")
        if len({item.name for item in metrics}) != len(metrics):
            raise ValueError("evaluator metric names must be unique")
        object.__setattr__(self, "metrics", tuple(sorted(metrics, key=lambda item: item.name)))
        object.__setattr__(
            self,
            "sample_semantics",
            _text(self.sample_semantics, "sample_semantics", 20_000),
        )
        object.__setattr__(
            self,
            "aggregation_semantics",
            _text(self.aggregation_semantics, "aggregation_semantics", 20_000),
        )
        provided = _sha(self.contract_sha256, "contract_sha256")
        if _file_sha(config) != self.config_sha256:
            raise ValueError("evaluator contract config bytes differ from receipt")
        if _digest(self.unsigned()) != provided:
            raise ValueError("authoritative evaluator contract digest mismatch")
        object.__setattr__(self, "contract_sha256", provided)

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-authoritative-evaluator-contract/v1",
            "evaluator_id": self.evaluator_id,
            "implementation_id": self.implementation_id,
            "source_commit": self.source_commit,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "metrics": [asdict(item) for item in self.metrics],
            "sample_semantics": self.sample_semantics,
            "aggregation_semantics": self.aggregation_semantics,
        }


def _parse_config(
    path: str | Path,
) -> tuple[Path, Mapping[str, Any], tuple[EvaluatorMetricContract, ...]]:
    source = safe_advanced_path(
        path,
        label="evaluator contract config",
        must_exist=True,
        require_file=True,
    )
    raw = _read_json(source, "evaluator contract config", _MAX_CONFIG_BYTES)
    required = {
        "schema",
        "evaluator_id",
        "implementation_id",
        "source_commit",
        "metrics",
        "sample_semantics",
        "aggregation_semantics",
    }
    if (
        set(raw) != required
        or raw.get("schema") != "rigorousrag-evaluator-contract-config/v1"
        or not isinstance(raw.get("metrics"), list)
    ):
        raise ValueError("unsupported evaluator contract config schema")
    metric_fields = {"name", "family", "direction", "scope", "definition"}
    metrics = []
    for index, item in enumerate(raw["metrics"]):
        if not isinstance(item, Mapping) or set(item) != metric_fields:
            raise ValueError(f"evaluator metric contract {index} fields are invalid")
        metrics.append(EvaluatorMetricContract(**dict(item)))
    return source, raw, tuple(metrics)


def build_authoritative_evaluator_contract(
    config_path: str | Path,
) -> AuthoritativeEvaluatorContract:
    source, raw, metrics = _parse_config(config_path)
    config_sha = _file_sha(source)
    unsigned = {
        "schema": "rigorousrag-authoritative-evaluator-contract/v1",
        "evaluator_id": raw["evaluator_id"],
        "implementation_id": raw["implementation_id"],
        "source_commit": _commit(raw["source_commit"]),
        "config_path": str(source),
        "config_sha256": config_sha,
        "metrics": [asdict(item) for item in sorted(metrics, key=lambda item: item.name)],
        "sample_semantics": raw["sample_semantics"],
        "aggregation_semantics": raw["aggregation_semantics"],
    }
    return AuthoritativeEvaluatorContract(
        evaluator_id=raw["evaluator_id"],
        implementation_id=raw["implementation_id"],
        source_commit=raw["source_commit"],
        config_path=str(source),
        config_sha256=config_sha,
        metrics=metrics,
        sample_semantics=raw["sample_semantics"],
        aggregation_semantics=raw["aggregation_semantics"],
        contract_sha256=_digest(unsigned),
    )


def write_authoritative_evaluator_contract(
    path: str | Path,
    contract: AuthoritativeEvaluatorContract,
) -> None:
    if not isinstance(contract, AuthoritativeEvaluatorContract):
        raise ValueError("contract must be AuthoritativeEvaluatorContract")
    destination = safe_advanced_path(
        path,
        label="authoritative evaluator contract receipt",
        must_exist=False,
    )
    _atomic(
        destination,
        _canonical({**contract.unsigned(), "contract_sha256": contract.contract_sha256})
        + b"\n",
    )


def read_authoritative_evaluator_contract(
    path: str | Path,
) -> AuthoritativeEvaluatorContract:
    source = safe_advanced_path(
        path,
        label="authoritative evaluator contract receipt",
        must_exist=True,
        require_file=True,
    )
    raw = _read_json(source, "authoritative evaluator contract receipt", _MAX_RECEIPT_BYTES)
    required = {
        "schema",
        "evaluator_id",
        "implementation_id",
        "source_commit",
        "config_path",
        "config_sha256",
        "metrics",
        "sample_semantics",
        "aggregation_semantics",
        "contract_sha256",
    }
    if (
        set(raw) != required
        or raw.get("schema") != "rigorousrag-authoritative-evaluator-contract/v1"
        or not isinstance(raw.get("metrics"), list)
    ):
        raise ValueError("unsupported authoritative evaluator contract schema")
    metric_fields = {"name", "family", "direction", "scope", "definition"}
    metrics = []
    for index, item in enumerate(raw["metrics"]):
        if not isinstance(item, Mapping) or set(item) != metric_fields:
            raise ValueError(f"evaluator metric receipt {index} fields are invalid")
        metrics.append(EvaluatorMetricContract(**dict(item)))
    return AuthoritativeEvaluatorContract(
        evaluator_id=raw["evaluator_id"],
        implementation_id=raw["implementation_id"],
        source_commit=raw["source_commit"],
        config_path=raw["config_path"],
        config_sha256=raw["config_sha256"],
        metrics=tuple(metrics),
        sample_semantics=raw["sample_semantics"],
        aggregation_semantics=raw["aggregation_semantics"],
        contract_sha256=raw["contract_sha256"],
    )


def verify_authoritative_evaluator_contract(
    path: str | Path,
) -> AuthoritativeEvaluatorContract:
    persisted = read_authoritative_evaluator_contract(path)
    rebuilt = build_authoritative_evaluator_contract(persisted.config_path)
    if rebuilt.contract_sha256 != persisted.contract_sha256:
        raise ValueError("persisted evaluator contract differs from config reconstruction")
    return persisted


__all__ = [
    "AuthoritativeEvaluatorContract",
    "EvaluatorMetricContract",
    "build_authoritative_evaluator_contract",
    "read_authoritative_evaluator_contract",
    "verify_authoritative_evaluator_contract",
    "write_authoritative_evaluator_contract",
]
