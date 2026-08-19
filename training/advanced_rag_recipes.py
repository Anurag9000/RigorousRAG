"""Deterministic recipe emitters for authoritative advanced-RAG training configs.

The training parser already defines the canonical JSON contract and staged curricula.  This
module removes the remaining hand-authored-envelope step: callers supply exact local
artifact/data/cache identities and receive a complete ``rigorousrag-advanced-training-config/v1``
file using the repository's canonical grounded or dynamic curriculum defaults.

Recipes never download models/data, create placeholder identities, or execute training.  The
written configuration is parsed back through :mod:`training.advanced_rag_config` before the
receipt is returned, proving that the emitted document is accepted by the same authority used
by ``validate`` and ``train``.
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
from training.advanced_rag_config import TensorCacheSpec, load_advanced_run_config
from training.advanced_rag_data import DynamicCollatorConfig, GroundedCollatorConfig
from training.advanced_rag_runner import LocalTrainingSplit, ParameterTrainabilityPolicy, TrainingExecutionConfig
from training.dynamic_retrieval_policy import DynamicPolicyArchitecture, DynamicRetrievalBudget
from training.grounded_generation import GroundedGenerationArchitectureConfig
from training.grounded_supervision_pipeline import RetrieverCouplingConfig
from training.local_artifact_loading import LocalArtifactTreeBinding

_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


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


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


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


def _artifact(binding: LocalArtifactTreeBinding) -> Mapping[str, Any]:
    if not isinstance(binding, LocalArtifactTreeBinding):
        raise ValueError("artifact recipe values must be LocalArtifactTreeBinding")
    return {"path": binding.path, "sha256": binding.expected_sha256, "kind": binding.artifact_kind}


def _split(value: LocalTrainingSplit) -> Mapping[str, Any]:
    if not isinstance(value, LocalTrainingSplit):
        raise ValueError("split recipe values must be LocalTrainingSplit")
    payload: dict[str, Any] = {"path": value.path, "sha256": value.content_sha256, "split_name": value.split_name}
    if value.expected_record_count is not None:
        payload["expected_record_count"] = value.expected_record_count
    return payload


def _cache(value: TensorCacheSpec | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, TensorCacheSpec):
        raise ValueError("cache recipe values must be TensorCacheSpec")
    return {"root": value.root, "identity": _jsonable(value.identity)}


def _trainability(values: Mapping[str, ParameterTrainabilityPolicy] | None) -> Mapping[str, Any] | None:
    if values is None:
        return None
    result: dict[str, Any] = {}
    for stage, policy in values.items():
        name = _identifier(stage, "trainability stage", 300)
        if not isinstance(policy, ParameterTrainabilityPolicy):
            raise ValueError("trainability values must be ParameterTrainabilityPolicy")
        result[name] = list(policy.trainable_prefixes)
    return result


def _atomic_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = safe_advanced_path(path, label="advanced RAG recipe output", must_exist=False)
    if destination.exists() and destination.is_dir():
        raise ValueError("advanced RAG recipe output must be a file path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


@dataclass(frozen=True)
class AdvancedRecipeReceipt:
    kind: str
    run_id: str
    output_path: str
    config_sha256: str
    plan_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in {"grounded_generation", "dynamic_rag_policy"}:
            raise ValueError("unsupported advanced recipe kind")
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id", 300))
        for name in ("config_sha256", "plan_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if _digest(self._unsigned()) != self.receipt_sha256:
            raise ValueError("advanced recipe receipt digest mismatch")

    def _unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-advanced-recipe-receipt/v1",
            "kind": self.kind,
            "run_id": self.run_id,
            "output_path": self.output_path,
            "config_sha256": self.config_sha256,
            "plan_sha256": self.plan_sha256,
        }


def _finalize(path: Path, *, kind: str, run_id: str) -> AdvancedRecipeReceipt:
    config_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    configured = load_advanced_run_config(path)
    actual_kind = "grounded_generation" if configured.__class__.__name__ == "GroundedConfiguredRun" else "dynamic_rag_policy"
    if actual_kind != kind:
        raise RuntimeError("generated recipe parsed as the wrong advanced run kind")
    if configured.plan.run_id != run_id:
        raise RuntimeError("generated recipe run_id changed during canonical parsing")
    unsigned = {
        "schema": "rigorousrag-advanced-recipe-receipt/v1",
        "kind": kind,
        "run_id": run_id,
        "output_path": str(path),
        "config_sha256": config_sha,
        "plan_sha256": configured.plan.plan_sha256,
    }
    return AdvancedRecipeReceipt(
        kind=kind,
        run_id=run_id,
        output_path=str(path),
        config_sha256=config_sha,
        plan_sha256=configured.plan.plan_sha256,
        receipt_sha256=_digest(unsigned),
    )


def write_grounded_training_recipe(
    output_path: str | Path,
    *,
    run_id: str,
    source_commit: str,
    dataset_manifest_sha256: str,
    base_model: LocalArtifactTreeBinding,
    tokenizer: LocalArtifactTreeBinding,
    architecture: GroundedGenerationArchitectureConfig,
    train_split: LocalTrainingSplit,
    validation_split: LocalTrainingSplit,
    checkpoint_root: str | Path,
    execution: TrainingExecutionConfig = TrainingExecutionConfig(),
    collator: GroundedCollatorConfig = GroundedCollatorConfig(),
    include_preference: bool = True,
    teacher_cache: TensorCacheSpec | None = None,
    reference_cache: TensorCacheSpec | None = None,
    retriever_model: LocalArtifactTreeBinding | None = None,
    retriever_utility_cache: TensorCacheSpec | None = None,
    retriever_coupling: RetrieverCouplingConfig = RetrieverCouplingConfig(),
    trainability: Mapping[str, ParameterTrainabilityPolicy] | None = None,
    resume_checkpoint_digest: str | None = None,
) -> AdvancedRecipeReceipt:
    """Write the canonical grounded SFT→attribution→grounding→reflection→coupling→preference→joint config."""
    selected_run = _identifier(run_id, "run_id", 300)
    commit = _commit(source_commit)
    dataset_sha = _sha(dataset_manifest_sha256, "dataset_manifest_sha256")
    if base_model.artifact_kind not in {"causal_lm", "seq2seq_lm"}:
        raise ValueError("grounded base_model must be causal_lm or seq2seq_lm")
    if tokenizer.artifact_kind != "tokenizer":
        raise ValueError("grounded tokenizer binding must have artifact_kind=tokenizer")
    if not isinstance(architecture, GroundedGenerationArchitectureConfig):
        raise ValueError("architecture must be GroundedGenerationArchitectureConfig")
    if not isinstance(execution, TrainingExecutionConfig) or not isinstance(collator, GroundedCollatorConfig):
        raise ValueError("execution/collator have incorrect types")
    if (retriever_model is None) != (retriever_utility_cache is None):
        raise ValueError("retriever_model and retriever_utility_cache must be configured together")
    if retriever_model is not None and retriever_model.artifact_kind != "sequence_classifier":
        raise ValueError("grounded retriever_model must be a sequence_classifier artifact")
    payload: dict[str, Any] = {
        "schema": "rigorousrag-advanced-training-config/v1",
        "kind": "grounded_generation",
        "run_id": selected_run,
        "source_commit": commit,
        "dataset_manifest_sha256": dataset_sha,
        "base_model": _artifact(base_model),
        "tokenizer": _artifact(tokenizer),
        "architecture": _jsonable(architecture),
        "curriculum": {"include_preference": bool(include_preference)},
        "execution": _jsonable(execution),
        "collator": _jsonable(collator),
        "train_split": _split(train_split),
        "validation_split": _split(validation_split),
        "checkpoint_root": str(safe_advanced_path(checkpoint_root, label="advanced checkpoint root", must_exist=False)),
        "resume_checkpoint_digest": None if resume_checkpoint_digest is None else _sha(resume_checkpoint_digest, "resume_checkpoint_digest"),
        "teacher_cache": _cache(teacher_cache),
        "reference_cache": _cache(reference_cache),
    }
    trainability_payload = _trainability(trainability)
    if trainability_payload is not None:
        payload["trainability"] = trainability_payload
    if retriever_model is not None:
        payload["retriever"] = {
            "model": _artifact(retriever_model),
            "utility_cache": _cache(retriever_utility_cache),
            "coupling": _jsonable(retriever_coupling),
        }
    destination = _atomic_json(output_path, payload)
    return _finalize(destination, kind="grounded_generation", run_id=selected_run)


def write_dynamic_training_recipe(
    output_path: str | Path,
    *,
    run_id: str,
    source_commit: str,
    dataset_manifest_sha256: str,
    generator: LocalArtifactTreeBinding,
    tokenizer: LocalArtifactTreeBinding,
    retrieval_stack_sha256: str,
    architecture: DynamicPolicyArchitecture,
    budget: DynamicRetrievalBudget,
    train_split: LocalTrainingSplit,
    validation_split: LocalTrainingSplit,
    checkpoint_root: str | Path,
    execution: TrainingExecutionConfig = TrainingExecutionConfig(),
    collator: DynamicCollatorConfig = DynamicCollatorConfig(),
    include_need_selection: bool = True,
    hidden_state_cache: TensorCacheSpec | None = None,
    trainability: Mapping[str, ParameterTrainabilityPolicy] | None = None,
    resume_checkpoint_digest: str | None = None,
) -> AdvancedRecipeReceipt:
    """Write the canonical imitation→need-selection→value→off-policy→cost-aware→joint config."""
    selected_run = _identifier(run_id, "run_id", 300)
    commit = _commit(source_commit)
    dataset_sha = _sha(dataset_manifest_sha256, "dataset_manifest_sha256")
    retrieval_sha = _sha(retrieval_stack_sha256, "retrieval_stack_sha256")
    if generator.artifact_kind not in {"causal_lm", "seq2seq_lm"}:
        raise ValueError("dynamic generator must be causal_lm or seq2seq_lm")
    if tokenizer.artifact_kind != "tokenizer":
        raise ValueError("dynamic tokenizer binding must have artifact_kind=tokenizer")
    if not isinstance(architecture, DynamicPolicyArchitecture) or not isinstance(budget, DynamicRetrievalBudget):
        raise ValueError("dynamic architecture/budget have incorrect types")
    if not isinstance(execution, TrainingExecutionConfig) or not isinstance(collator, DynamicCollatorConfig):
        raise ValueError("execution/collator have incorrect types")
    if include_need_selection and hidden_state_cache is None:
        raise ValueError("canonical dynamic recipe with need-selection requires hidden_state_cache")
    payload: dict[str, Any] = {
        "schema": "rigorousrag-advanced-training-config/v1",
        "kind": "dynamic_rag_policy",
        "run_id": selected_run,
        "source_commit": commit,
        "dataset_manifest_sha256": dataset_sha,
        "generator": _artifact(generator),
        "tokenizer": _artifact(tokenizer),
        "architecture": _jsonable(architecture),
        "budget": _jsonable(budget),
        "retrieval_stack_sha256": retrieval_sha,
        "curriculum": {"include_need_selection": bool(include_need_selection)},
        "execution": _jsonable(execution),
        "collator": _jsonable(collator),
        "train_split": _split(train_split),
        "validation_split": _split(validation_split),
        "checkpoint_root": str(safe_advanced_path(checkpoint_root, label="advanced checkpoint root", must_exist=False)),
        "resume_checkpoint_digest": None if resume_checkpoint_digest is None else _sha(resume_checkpoint_digest, "resume_checkpoint_digest"),
        "hidden_state_cache": _cache(hidden_state_cache),
    }
    trainability_payload = _trainability(trainability)
    if trainability_payload is not None:
        payload["trainability"] = trainability_payload
    destination = _atomic_json(output_path, payload)
    return _finalize(destination, kind="dynamic_rag_policy", run_id=selected_run)


__all__ = ["AdvancedRecipeReceipt", "write_dynamic_training_recipe", "write_grounded_training_recipe"]
