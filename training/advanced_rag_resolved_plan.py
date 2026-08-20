"""Resolved, restart-verifiable advanced-RAG training plan artifacts.

Canonical recipes may intentionally omit stage overrides, in which case the source-owned
curriculum defaults are authoritative.  This module materializes the *resolved* plan after the
strict config parser has applied those defaults/overrides.  The artifact therefore records exact
stage order, step counts, checkpoint cadence, learning rates, objective weights, execution /
collator settings, trainability, split identities and cache contracts.

It also provides an atomic helper for adding bounded curriculum-stage hyperparameter overrides to
an already-emitted v1 recipe, followed by canonical parse-back and receipt regeneration.  No model
is loaded and no training executes.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_config import DynamicConfiguredRun, GroundedConfiguredRun, load_advanced_run_config
from training.advanced_rag_recipes import AdvancedRecipeReceipt

_MAX_CONFIG_BYTES = 16 * 1024 * 1024
_GROUNDED_STAGES = frozenset({"supervised", "attribution", "grounding", "reflection", "retriever_coupling", "preference", "joint"})
_DYNAMIC_STAGES = frozenset({"imitation", "need_selection", "value", "off_policy", "cost_aware", "joint"})
_STAGE_FIELDS = frozenset({"max_steps", "checkpoint_every_steps", "learning_rate"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _read(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    source = safe_advanced_path(path, label="advanced RAG recipe", must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("advanced RAG recipe exceeds byte safety bound")
    try:
        raw = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError("advanced RAG recipe is not strict JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("advanced RAG recipe must contain an object")
    return source, raw


def _atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _normalized_stage_overrides(value: Any, *, kind: str) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("curriculum stage overrides must be an object")
    allowed_stages = _GROUNDED_STAGES if kind == "grounded_generation" else _DYNAMIC_STAGES
    unknown_stages = set(value) - allowed_stages
    if unknown_stages:
        raise ValueError(f"curriculum stage overrides contain unknown stages: {sorted(unknown_stages)}")
    result: dict[str, Mapping[str, Any]] = {}
    for stage, raw in value.items():
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError(f"curriculum stage override {stage!r} must be a non-empty object")
        unknown = set(raw) - _STAGE_FIELDS
        if unknown:
            raise ValueError(f"curriculum stage override {stage!r} contains unsupported fields: {sorted(unknown)}")
        normalized: dict[str, Any] = {}
        for name, item in raw.items():
            if name in {"max_steps", "checkpoint_every_steps"}:
                if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                    raise ValueError(f"{stage}.{name} must be a positive integer")
                normalized[name] = item
            else:
                if isinstance(item, bool) or not isinstance(item, (int, float)) or not 0.0 < float(item) <= 10.0:
                    raise ValueError(f"{stage}.learning_rate must be positive and bounded")
                normalized[name] = float(item)
        if "max_steps" in normalized and "checkpoint_every_steps" in normalized and normalized["checkpoint_every_steps"] > normalized["max_steps"]:
            raise ValueError(f"{stage}.checkpoint_every_steps may not exceed max_steps")
        result[str(stage)] = normalized
    return result


def _recipe_receipt(path: Path) -> AdvancedRecipeReceipt:
    configured = load_advanced_run_config(path)
    if isinstance(configured, GroundedConfiguredRun):
        kind = "grounded_generation"
    elif isinstance(configured, DynamicConfiguredRun):
        kind = "dynamic_rag_policy"
    else:  # pragma: no cover
        raise RuntimeError("advanced recipe resolved to unsupported configured run")
    config_sha = _stream_sha(path)
    unsigned = {
        "schema": "rigorousrag-advanced-recipe-receipt/v1",
        "kind": kind,
        "run_id": configured.plan.run_id,
        "output_path": str(path),
        "config_sha256": config_sha,
        "plan_sha256": configured.plan.plan_sha256,
    }
    return AdvancedRecipeReceipt(
        kind=kind,
        run_id=configured.plan.run_id,
        output_path=str(path),
        config_sha256=config_sha,
        plan_sha256=configured.plan.plan_sha256,
        receipt_sha256=_digest(unsigned),
    )


def apply_curriculum_stage_overrides(
    recipe_path: str | Path,
    stage_overrides: Mapping[str, Mapping[str, Any]],
) -> AdvancedRecipeReceipt:
    source, raw = _read(recipe_path)
    if raw.get("schema") != "rigorousrag-advanced-training-config/v1":
        raise ValueError("curriculum overrides require advanced training config v1")
    kind = raw.get("kind")
    if kind not in {"grounded_generation", "dynamic_rag_policy"}:
        raise ValueError("unsupported advanced training kind")
    overrides = _normalized_stage_overrides(stage_overrides, kind=kind)
    selected = dict(raw)
    curriculum = selected.get("curriculum") or {}
    if not isinstance(curriculum, Mapping):
        raise ValueError("advanced training curriculum must be an object")
    curriculum = dict(curriculum)
    curriculum["stages"] = {stage: dict(values) for stage, values in overrides.items()}
    selected["curriculum"] = curriculum
    _atomic(source, selected)
    # Parse-back validates defaults + overrides together, including checkpoint<=max-step rules
    # when only one side of the pair was overridden.
    return _recipe_receipt(source)


def _cache_descriptor(spec: Any | None) -> Mapping[str, Any] | None:
    if spec is None:
        return None
    return {
        "root": spec.root,
        "identity": _jsonable(spec.identity),
        "contract_sha256": spec.contract_sha256,
    }


def resolved_training_plan_payload(recipe_path: str | Path) -> Mapping[str, Any]:
    configured = load_advanced_run_config(recipe_path)
    plan = configured.plan
    stages = []
    for stage in plan.stages:
        stages.append({
            "name": stage.name,
            "kind": stage.kind.value,
            "max_steps": stage.max_steps,
            "checkpoint_every_steps": stage.checkpoint_every_steps,
            "learning_rate": stage.learning_rate,
            "objective": _jsonable(stage.objective),
            "objective_sha256": stage.objective.objective_sha256,
            "stage_sha256": stage.stage_sha256,
        })
    common: dict[str, Any] = {
        "schema": "rigorousrag-resolved-advanced-training-plan/v1",
        "kind": "grounded_generation" if isinstance(configured, GroundedConfiguredRun) else "dynamic_rag_policy",
        "run_id": plan.run_id,
        "plan_sha256": plan.plan_sha256,
        "source_commit": plan.source_commit,
        "dataset_manifest_sha256": plan.dataset_manifest_sha256,
        "architecture": _jsonable(plan.architecture),
        "stages": stages,
        "execution": _jsonable(configured.execution),
        "collator": _jsonable(configured.collator),
        "trainability": {name: list(policy.trainable_prefixes) for name, policy in sorted(configured.trainability.items())},
        "train_split": _jsonable(configured.train_split),
        "validation_split": _jsonable(configured.validation_split),
        "checkpoint_root": configured.checkpoint_root,
        "resume_checkpoint_digest": configured.resume_checkpoint_digest,
    }
    if isinstance(configured, GroundedConfiguredRun):
        common.update({
            "base_model": _jsonable(configured.base_model),
            "tokenizer": _jsonable(configured.tokenizer),
            "teacher_cache": _cache_descriptor(configured.teacher_cache),
            "reference_cache": _cache_descriptor(configured.reference_cache),
            "retriever_model": None if configured.retriever_model is None else _jsonable(configured.retriever_model),
            "retriever_utility_cache": _cache_descriptor(configured.retriever_utility_cache),
            "retriever_coupling": _jsonable(configured.retriever_coupling),
        })
    else:
        common.update({
            "generator": _jsonable(configured.generator),
            "tokenizer": _jsonable(configured.tokenizer),
            "budget": _jsonable(plan.budget),
            "retrieval_stack_sha256": plan.retrieval_stack_sha256,
            "hidden_state_cache": _cache_descriptor(configured.hidden_state_cache),
        })
    unsigned = dict(common)
    common["resolved_plan_sha256"] = _digest(unsigned)
    return common


def write_resolved_training_plan(recipe_path: str | Path, output_path: str | Path) -> Mapping[str, Any]:
    payload = resolved_training_plan_payload(recipe_path)
    destination = safe_advanced_path(output_path, label="resolved advanced training plan output", must_exist=False)
    if destination.exists():
        raise ValueError("resolved advanced training plan output must not already exist")
    _atomic(destination, payload)
    _, read_back = _read(destination)
    if read_back != payload:
        raise RuntimeError("resolved advanced training plan changed during atomic write")
    return payload


__all__ = [
    "apply_curriculum_stage_overrides",
    "resolved_training_plan_payload",
    "write_resolved_training_plan",
]
