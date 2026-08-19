"""Checkpoint-to-inference artifact export and promotion contracts for advanced RAG.

Authoritative exports require a checkpoint/run-config verification receipt.  The resulting
manifest carries enough canonical architecture data to reconstruct the repository-owned
heads/controllers while still referring to separately admitted base-model/tokenizer trees by
digest.  Nothing is loaded or executed on import.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from training.advanced_rag_run_binding import VerifiedAdvancedCheckpointBinding
from training.checkpointing import CheckpointManager
from training.dynamic_retrieval_policy import DynamicPolicyArchitecture, DynamicPolicyTrainingPlan, DynamicRetrievalBudget
from training.grounded_generation import GroundedGenerationArchitectureConfig, GroundedTrainingPlan

_HEX = frozenset("0123456789abcdef")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def grounded_runtime_config(plan: GroundedTrainingPlan) -> Mapping[str, Any]:
    architecture = plan.architecture
    return {
        "architecture": {
            "hidden_size": architecture.hidden_size,
            "attribution_size": architecture.attribution_size,
            "reflection_actions": [action.value for action in architecture.reflection_actions],
        }
    }


def dynamic_runtime_config(plan: DynamicPolicyTrainingPlan) -> Mapping[str, Any]:
    architecture = plan.architecture
    budget = plan.budget
    return {
        "architecture": {
            "feature_names": list(architecture.feature_names),
            "hidden_size": architecture.hidden_size,
            "context_hidden_size": architecture.context_hidden_size,
            "need_projection_size": architecture.need_projection_size,
            "actions": [action.value for action in architecture.actions],
        },
        "budget": {
            "max_generation_tokens": budget.max_generation_tokens,
            "max_retrievals": budget.max_retrievals,
            "max_verifications": budget.max_verifications,
            "min_tokens_before_retrieval": budget.min_tokens_before_retrieval,
            "min_tokens_before_stop": budget.min_tokens_before_stop,
            "max_consecutive_retrievals": budget.max_consecutive_retrievals,
        },
    }


def _validate_runtime_config(kind: str, value: Mapping[str, Any], *, architecture_sha256: str, budget_sha256: str | None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("runtime_config must be a mapping")
    if kind == "grounded_generator":
        if set(value) != {"architecture"} or not isinstance(value["architecture"], Mapping):
            raise ValueError("grounded runtime_config must contain only architecture")
        architecture = GroundedGenerationArchitectureConfig(**dict(value["architecture"]))
        if architecture.architecture_sha256 != architecture_sha256:
            raise ValueError("grounded runtime architecture does not match architecture_sha256")
        if budget_sha256 is not None:
            raise ValueError("grounded artifact may not carry dynamic budget")
    elif kind == "dynamic_rag_policy":
        if set(value) != {"architecture", "budget"} or not isinstance(value["architecture"], Mapping) or not isinstance(value["budget"], Mapping):
            raise ValueError("dynamic runtime_config must contain architecture and budget")
        architecture = DynamicPolicyArchitecture(**dict(value["architecture"]))
        budget = DynamicRetrievalBudget(**dict(value["budget"]))
        if architecture.architecture_sha256 != architecture_sha256:
            raise ValueError("dynamic runtime architecture does not match architecture_sha256")
        if budget_sha256 is None or budget.budget_sha256 != budget_sha256:
            raise ValueError("dynamic runtime budget does not match budget_sha256")
    else:
        raise ValueError("unsupported advanced artifact kind")
    # Round-trip through strict canonical JSON types so callers cannot retain mutable aliases.
    return json.loads(_canonical(value).decode("utf-8"))


@dataclass(frozen=True)
class MetricQualificationPolicy:
    minimum: Mapping[str, float] = field(default_factory=dict)
    maximum: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum", {str(k): _finite(v, f"minimum[{k}]") for k, v in self.minimum.items()})
        object.__setattr__(self, "maximum", {str(k): _finite(v, f"maximum[{k}]") for k, v in self.maximum.items()})
        overlap = set(self.minimum) & set(self.maximum)
        for key in overlap:
            if self.minimum[key] > self.maximum[key]:
                raise ValueError(f"metric {key} minimum exceeds maximum")

    @property
    def policy_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-advanced-promotion-policy/v1", "minimum": self.minimum, "maximum": self.maximum})


@dataclass(frozen=True)
class AdvancedArtifactManifest:
    kind: str
    checkpoint_digest: str
    plan_sha256: str
    training_input_sha256: str
    training_config_sha256: str
    source_commit: str
    dataset_manifest_sha256: str
    architecture_sha256: str
    base_model_sha256: str
    generator_family: str | None
    tokenizer_sha256: str | None
    retrieval_stack_sha256: str | None
    budget_sha256: str | None
    runtime_config: Mapping[str, Any]
    weights_sha256: str
    weights_bytes: int
    included_prefixes: tuple[str, ...]
    evaluation_receipt_sha256: str | None
    artifact_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in {"grounded_generator", "dynamic_rag_policy"}:
            raise ValueError("unsupported advanced artifact kind")
        for name in (
            "checkpoint_digest", "plan_sha256", "training_input_sha256", "training_config_sha256",
            "dataset_manifest_sha256", "architecture_sha256", "base_model_sha256", "weights_sha256", "artifact_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        commit = str(self.source_commit).strip().lower()
        if len(commit) not in {40, 64} or any(ch not in _HEX for ch in commit):
            raise ValueError("source_commit must be a full Git object id")
        object.__setattr__(self, "source_commit", commit)
        if self.kind == "grounded_generator":
            if self.generator_family not in {"causal_lm", "seq2seq_lm"}:
                raise ValueError("grounded generator artifact requires causal_lm or seq2seq_lm generator_family")
            if self.tokenizer_sha256 is None:
                raise ValueError("grounded generator artifact requires tokenizer identity")
        elif self.generator_family is not None:
            raise ValueError("dynamic policy artifact does not own a generator family")
        for name in ("tokenizer_sha256", "retrieval_stack_sha256", "budget_sha256", "evaluation_receipt_sha256"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sha(value, name))
        object.__setattr__(self, "runtime_config", _validate_runtime_config(self.kind, self.runtime_config, architecture_sha256=self.architecture_sha256, budget_sha256=self.budget_sha256))
        if isinstance(self.weights_bytes, bool) or not isinstance(self.weights_bytes, int) or self.weights_bytes <= 0:
            raise ValueError("weights_bytes must be positive")
        prefixes = tuple(str(value).strip() for value in self.included_prefixes)
        if not prefixes or any(not value for value in prefixes):
            raise ValueError("included_prefixes must be non-empty")
        object.__setattr__(self, "included_prefixes", prefixes)


@dataclass(frozen=True)
class AdvancedArtifactPromotionReceipt:
    artifact_sha256: str
    policy_sha256: str
    evaluation_receipt_sha256: str
    promoted: bool
    reason_codes: tuple[str, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("artifact_sha256", "policy_sha256", "evaluation_receipt_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        reasons = tuple(str(value).strip() for value in self.reason_codes)
        if any(not value for value in reasons):
            raise ValueError("promotion reason codes are invalid")
        if self.promoted and reasons:
            raise ValueError("promoted receipt may not contain failure reasons")
        if not self.promoted and not reasons:
            raise ValueError("blocked promotion requires reason codes")
        object.__setattr__(self, "reason_codes", reasons)


class ArtifactAdmissionSink(Protocol):
    def admit(self, artifact_directory: str, *, artifact_sha256: str, promotion_receipt_sha256: str) -> Any: ...


def _save_filtered_safetensors(source: Path, destination: Path, prefixes: Sequence[str]) -> tuple[str, int]:
    try:
        from safetensors.torch import load_file, save_file
    except Exception as exc:
        raise RuntimeError("advanced artifact export requires safetensors") from exc
    state = load_file(str(source), device="cpu")
    selected = {name: tensor.contiguous() for name, tensor in state.items() if any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in prefixes)}
    if not selected:
        raise ValueError("artifact export prefixes selected no checkpoint tensors")
    save_file(selected, str(destination), metadata={"format": "pt", "rigorousrag": "advanced-inference-artifact"})
    return _file_sha(destination), destination.stat().st_size


def _write_manifest(directory: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical(payload) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, directory / "manifest.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _manifest_payload(**values: Any) -> tuple[dict[str, Any], str]:
    unsigned = {"schema": "rigorousrag-advanced-inference-artifact/v3", **values}
    artifact_sha = _digest(unsigned)
    return {**unsigned, "artifact_sha256": artifact_sha}, artifact_sha


def _assert_binding(binding: VerifiedAdvancedCheckpointBinding, *, kind: str, plan_sha256: str, checkpoint_digest: str) -> None:
    if not isinstance(binding, VerifiedAdvancedCheckpointBinding):
        raise ValueError("verified_binding must be VerifiedAdvancedCheckpointBinding")
    if binding.kind != kind or binding.plan_sha256 != plan_sha256 or binding.checkpoint_digest != checkpoint_digest:
        raise ValueError("verified checkpoint binding does not match requested artifact export")


def export_grounded_generator_artifact(
    *,
    checkpoint_manager: CheckpointManager,
    checkpoint_digest: str,
    plan: GroundedTrainingPlan,
    verified_binding: VerifiedAdvancedCheckpointBinding,
    destination_root: str | Path,
    evaluation_receipt_sha256: str | None = None,
    include_retriever: bool = True,
) -> AdvancedArtifactManifest:
    _assert_binding(verified_binding, kind="grounded_generation", plan_sha256=plan.plan_sha256, checkpoint_digest=checkpoint_digest)
    path, checkpoint = checkpoint_manager.verify(checkpoint_digest)
    if (
        checkpoint.source_commit != plan.source_commit
        or checkpoint.dataset_manifest_digest != plan.dataset_manifest_sha256
        or checkpoint.model_architecture != f"grounded_generation:{plan.plan_sha256}"
        or checkpoint.training_config_digest != verified_binding.training_config_sha256
        or checkpoint.run_id != verified_binding.bound_run_id
    ):
        raise ValueError("checkpoint no longer matches verified grounded training binding")
    prefixes = ["base_model.", "auxiliary."]
    if include_retriever and plan.retriever_stack_sha256 is not None:
        prefixes.append("retriever_model.")
    root = Path(destination_root).expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".grounded-export-", dir=root))
    try:
        weights_sha, weights_bytes = _save_filtered_safetensors(path / "model.safetensors", temporary / "model.safetensors", prefixes)
        values = {
            "kind": "grounded_generator", "checkpoint_digest": checkpoint_digest, "plan_sha256": plan.plan_sha256,
            "training_input_sha256": verified_binding.training_input_sha256,
            "training_config_sha256": verified_binding.training_config_sha256,
            "source_commit": plan.source_commit, "dataset_manifest_sha256": plan.dataset_manifest_sha256,
            "architecture_sha256": plan.architecture.architecture_sha256, "base_model_sha256": plan.base_model_sha256,
            "generator_family": verified_binding.generator_family,
            "tokenizer_sha256": plan.tokenizer_sha256,
            "retrieval_stack_sha256": plan.retriever_stack_sha256 if include_retriever else None,
            "budget_sha256": None, "runtime_config": grounded_runtime_config(plan),
            "weights_sha256": weights_sha, "weights_bytes": weights_bytes,
            "included_prefixes": prefixes, "evaluation_receipt_sha256": evaluation_receipt_sha256,
        }
        payload, artifact_sha = _manifest_payload(**values)
        _write_manifest(temporary, payload)
        destination = root / artifact_sha
        if destination.exists():
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
        return AdvancedArtifactManifest(**{**values, "included_prefixes": tuple(prefixes), "artifact_sha256": artifact_sha})
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def export_dynamic_policy_artifact(
    *,
    checkpoint_manager: CheckpointManager,
    checkpoint_digest: str,
    plan: DynamicPolicyTrainingPlan,
    verified_binding: VerifiedAdvancedCheckpointBinding,
    destination_root: str | Path,
    evaluation_receipt_sha256: str | None = None,
) -> AdvancedArtifactManifest:
    _assert_binding(verified_binding, kind="dynamic_rag_policy", plan_sha256=plan.plan_sha256, checkpoint_digest=checkpoint_digest)
    path, checkpoint = checkpoint_manager.verify(checkpoint_digest)
    if (
        checkpoint.source_commit != plan.source_commit
        or checkpoint.dataset_manifest_digest != plan.dataset_manifest_sha256
        or checkpoint.model_architecture != f"dynamic_retrieval_policy:{plan.plan_sha256}"
        or checkpoint.training_config_digest != verified_binding.training_config_sha256
        or checkpoint.run_id != verified_binding.bound_run_id
    ):
        raise ValueError("checkpoint no longer matches verified dynamic-policy training binding")
    prefixes = ("controller.", "need_selector.")
    root = Path(destination_root).expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".dynamic-export-", dir=root))
    try:
        weights_sha, weights_bytes = _save_filtered_safetensors(path / "model.safetensors", temporary / "model.safetensors", prefixes)
        values = {
            "kind": "dynamic_rag_policy", "checkpoint_digest": checkpoint_digest, "plan_sha256": plan.plan_sha256,
            "training_input_sha256": verified_binding.training_input_sha256,
            "training_config_sha256": verified_binding.training_config_sha256,
            "source_commit": plan.source_commit, "dataset_manifest_sha256": plan.dataset_manifest_sha256,
            "architecture_sha256": plan.architecture.architecture_sha256, "base_model_sha256": plan.base_generator_sha256,
            "generator_family": None, "tokenizer_sha256": None, "retrieval_stack_sha256": plan.retrieval_stack_sha256,
            "budget_sha256": plan.budget.budget_sha256, "runtime_config": dynamic_runtime_config(plan),
            "weights_sha256": weights_sha, "weights_bytes": weights_bytes,
            "included_prefixes": list(prefixes), "evaluation_receipt_sha256": evaluation_receipt_sha256,
        }
        payload, artifact_sha = _manifest_payload(**values)
        _write_manifest(temporary, payload)
        destination = root / artifact_sha
        if destination.exists():
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
        return AdvancedArtifactManifest(**{**values, "included_prefixes": prefixes, "artifact_sha256": artifact_sha})
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def qualify_advanced_artifact(manifest: AdvancedArtifactManifest, *, evaluation_receipt_sha256: str, metrics: Mapping[str, float], policy: MetricQualificationPolicy) -> AdvancedArtifactPromotionReceipt:
    evaluation_sha = _sha(evaluation_receipt_sha256, "evaluation_receipt_sha256")
    if manifest.evaluation_receipt_sha256 is not None and manifest.evaluation_receipt_sha256 != evaluation_sha:
        raise ValueError("artifact manifest is bound to a different evaluation receipt")
    selected = {str(key): _finite(value, f"metric {key}") for key, value in metrics.items()}
    reasons = []
    for key, threshold in policy.minimum.items():
        if key not in selected:
            reasons.append(f"missing_minimum_metric:{key}")
        elif selected[key] < threshold:
            reasons.append(f"below_minimum:{key}")
    for key, threshold in policy.maximum.items():
        if key not in selected:
            reasons.append(f"missing_maximum_metric:{key}")
        elif selected[key] > threshold:
            reasons.append(f"above_maximum:{key}")
    promoted = not reasons
    unsigned = {
        "schema": "rigorousrag-advanced-artifact-promotion/v1", "artifact_sha256": manifest.artifact_sha256,
        "policy_sha256": policy.policy_sha256, "evaluation_receipt_sha256": evaluation_sha,
        "promoted": promoted, "reason_codes": sorted(reasons), "metrics_sha256": _digest(selected),
    }
    receipt_sha = _digest(unsigned)
    return AdvancedArtifactPromotionReceipt(manifest.artifact_sha256, policy.policy_sha256, evaluation_sha, promoted, tuple(sorted(reasons)), receipt_sha)


def admit_promoted_artifact(directory: str | Path, manifest: AdvancedArtifactManifest, receipt: AdvancedArtifactPromotionReceipt, sink: ArtifactAdmissionSink) -> Any:
    if not receipt.promoted or receipt.artifact_sha256 != manifest.artifact_sha256:
        raise ValueError("only a matching promoted artifact may enter admission")
    selected = Path(directory).expanduser().resolve(strict=True)
    if selected.name != manifest.artifact_sha256 or not selected.is_dir() or selected.is_symlink():
        raise ValueError("artifact directory is not the content-addressed promoted artifact")
    return sink.admit(str(selected), artifact_sha256=manifest.artifact_sha256, promotion_receipt_sha256=receipt.receipt_sha256)


__all__ = [
    "AdvancedArtifactManifest", "AdvancedArtifactPromotionReceipt", "ArtifactAdmissionSink", "MetricQualificationPolicy",
    "admit_promoted_artifact", "dynamic_runtime_config", "export_dynamic_policy_artifact",
    "export_grounded_generator_artifact", "grounded_runtime_config", "qualify_advanced_artifact",
]
