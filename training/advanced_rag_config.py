"""Strict JSON configuration for reproducible advanced RAG training runs."""
from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_curricula import CurriculumStageHyperparameters, DynamicCurriculumHyperparameters, GroundedCurriculumHyperparameters, build_dynamic_curriculum, build_grounded_curriculum
from training.advanced_rag_data import DynamicCollatorConfig, GroundedCollatorConfig
from training.advanced_rag_runner import LocalTrainingSplit, ParameterTrainabilityPolicy, TrainingExecutionConfig
from training.advanced_rag_strict_cache import AuthoritativeSafetensorSupervisionCache
from training.advanced_rag_supervision import SafetensorSupervisionCache, SupervisionCacheIdentity
from training.dynamic_retrieval_policy import DynamicPolicyArchitecture, DynamicPolicyTrainingPlan, DynamicRetrievalBudget
from training.grounded_generation import GroundedGenerationArchitectureConfig, GroundedTrainingPlan
from training.grounded_supervision_pipeline import RetrieverCouplingConfig
from training.local_artifact_loading import LocalArtifactTreeBinding

_HEX = frozenset("0123456789abcdef")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _strict(value: Mapping[str, Any], *, allowed: set[str], required: set[str], label: str) -> Mapping[str, Any]:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{label} is missing required fields: {sorted(missing)}")
    return value


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _dataclass_kwargs(cls: Any, raw: Mapping[str, Any], label: str) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {sorted(unknown)}")
    return dict(raw)


def _read_json(path: str | Path) -> Mapping[str, Any]:
    selected = safe_advanced_path(path, label="advanced RAG config", must_exist=True, require_file=True)
    if selected.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("advanced RAG config exceeds byte safety bound")
    try:
        value = json.loads(selected.read_text(encoding="utf-8"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc:
        raise ValueError("advanced RAG configuration is not strict JSON") from exc
    return _mapping(value, "configuration")


def _artifact(raw: Any, *, kind: str | None = None, label: str) -> LocalArtifactTreeBinding:
    selected = _strict(_mapping(raw, label), allowed={"path", "sha256", "kind"}, required={"path", "sha256", "kind"}, label=label)
    binding = LocalArtifactTreeBinding(path=selected["path"], expected_sha256=selected["sha256"], artifact_kind=selected["kind"])
    if kind is not None and binding.artifact_kind != kind:
        raise ValueError(f"{label} must have kind={kind}")
    return binding


def _split(raw: Any, label: str) -> LocalTrainingSplit:
    selected = _strict(_mapping(raw, label), allowed={"path", "sha256", "split_name", "expected_record_count"}, required={"path", "sha256", "split_name"}, label=label)
    path = safe_advanced_path(selected["path"], label=f"{label}.path", must_exist=True, require_file=True)
    return LocalTrainingSplit(path=str(path), content_sha256=selected["sha256"], split_name=selected["split_name"], expected_record_count=selected.get("expected_record_count"))


@dataclass(frozen=True)
class TensorCacheSpec:
    """Exact immutable binding to one materialized supervision cache."""
    root: str
    identity: SupervisionCacheIdentity
    contract_sha256: str

    def __post_init__(self) -> None:
        root = safe_advanced_path(self.root, label="supervision cache root", must_exist=True, require_directory=True)
        object.__setattr__(self, "root", str(root))
        if not isinstance(self.identity, SupervisionCacheIdentity):
            raise ValueError("cache identity must be SupervisionCacheIdentity")
        object.__setattr__(self, "contract_sha256", _sha(self.contract_sha256, "cache contract_sha256"))

    def build(self) -> SafetensorSupervisionCache:
        cache = AuthoritativeSafetensorSupervisionCache(self.root, self.identity)
        actual = cache.seal()
        if actual != self.contract_sha256:
            raise ValueError("supervision cache content contract differs from training config")
        return cache


def _cache(raw: Any | None, label: str, *, expected_kind: str) -> TensorCacheSpec | None:
    if raw is None:
        return None
    selected = _strict(
        _mapping(raw, label),
        allowed={"root", "identity", "contract_sha256"},
        required={"root", "identity", "contract_sha256"},
        label=label,
    )
    identity_raw = _strict(
        _mapping(selected["identity"], f"{label}.identity"),
        allowed={"cache_kind", "producer_sha256", "tokenizer_sha256", "dataset_manifest_sha256", "source_commit", "config_sha256"},
        required={"cache_kind", "producer_sha256", "tokenizer_sha256", "dataset_manifest_sha256", "source_commit", "config_sha256"},
        label=f"{label}.identity",
    )
    identity = SupervisionCacheIdentity(**dict(identity_raw))
    if identity.cache_kind != expected_kind:
        raise ValueError(f"{label} must use cache_kind={expected_kind}")
    spec = TensorCacheSpec(root=selected["root"], identity=identity, contract_sha256=selected["contract_sha256"])
    spec.build()
    return spec


def _stage(raw: Any, default: CurriculumStageHyperparameters, label: str) -> CurriculumStageHyperparameters:
    if raw is None:
        return default
    return replace(default, **_dataclass_kwargs(CurriculumStageHyperparameters, _mapping(raw, label), label))


def _grounded_hyper(raw: Any | None) -> GroundedCurriculumHyperparameters:
    if raw is None:
        return GroundedCurriculumHyperparameters()
    selected = _mapping(raw, "curriculum.stages")
    allowed = {field.name for field in fields(GroundedCurriculumHyperparameters)}
    unknown = set(selected) - allowed
    if unknown:
        raise ValueError(f"grounded curriculum contains unknown stages: {sorted(unknown)}")
    defaults = GroundedCurriculumHyperparameters()
    return GroundedCurriculumHyperparameters(**{name: _stage(selected.get(name), getattr(defaults, name), f"curriculum.stages.{name}") for name in allowed})


def _dynamic_hyper(raw: Any | None) -> DynamicCurriculumHyperparameters:
    if raw is None:
        return DynamicCurriculumHyperparameters()
    selected = _mapping(raw, "curriculum.stages")
    allowed = {field.name for field in fields(DynamicCurriculumHyperparameters)}
    unknown = set(selected) - allowed
    if unknown:
        raise ValueError(f"dynamic curriculum contains unknown stages: {sorted(unknown)}")
    defaults = DynamicCurriculumHyperparameters()
    return DynamicCurriculumHyperparameters(**{name: _stage(selected.get(name), getattr(defaults, name), f"curriculum.stages.{name}") for name in allowed})


def _trainability(raw: Any | None, defaults: Mapping[str, ParameterTrainabilityPolicy]) -> Mapping[str, ParameterTrainabilityPolicy]:
    if raw is None:
        return dict(defaults)
    selected = _mapping(raw, "trainability")
    unknown = set(selected) - set(defaults)
    if unknown:
        raise ValueError(f"trainability overrides unknown stage names: {sorted(unknown)}")
    result = dict(defaults)
    for stage, prefixes in selected.items():
        if not isinstance(prefixes, list) or any(not isinstance(value, str) for value in prefixes):
            raise ValueError("trainability values must be arrays of parameter prefixes")
        result[stage] = ParameterTrainabilityPolicy(tuple(prefixes))
    return result


def _checkpoint_root(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("checkpoint_root must be a non-empty path string")
    selected = safe_advanced_path(value, label="advanced checkpoint root", must_exist=False)
    if selected.exists() and not selected.is_dir():
        raise ValueError("advanced checkpoint root must be a directory when it exists")
    return str(selected)


@dataclass(frozen=True)
class GroundedConfiguredRun:
    plan: GroundedTrainingPlan
    trainability: Mapping[str, ParameterTrainabilityPolicy]
    execution: TrainingExecutionConfig
    collator: GroundedCollatorConfig
    train_split: LocalTrainingSplit
    validation_split: LocalTrainingSplit
    checkpoint_root: str
    base_model: LocalArtifactTreeBinding
    tokenizer: LocalArtifactTreeBinding
    resume_checkpoint_digest: str | None
    teacher_cache: TensorCacheSpec | None
    reference_cache: TensorCacheSpec | None
    retriever_model: LocalArtifactTreeBinding | None
    retriever_utility_cache: TensorCacheSpec | None
    retriever_coupling: RetrieverCouplingConfig


@dataclass(frozen=True)
class DynamicConfiguredRun:
    plan: DynamicPolicyTrainingPlan
    trainability: Mapping[str, ParameterTrainabilityPolicy]
    execution: TrainingExecutionConfig
    collator: DynamicCollatorConfig
    train_split: LocalTrainingSplit
    validation_split: LocalTrainingSplit
    checkpoint_root: str
    generator: LocalArtifactTreeBinding
    tokenizer: LocalArtifactTreeBinding
    resume_checkpoint_digest: str | None
    hidden_state_cache: TensorCacheSpec | None


def load_grounded_run_config(path: str | Path) -> GroundedConfiguredRun:
    root = _strict(
        _read_json(path),
        allowed={"schema", "kind", "run_id", "source_commit", "dataset_manifest_sha256", "base_model", "tokenizer", "architecture", "curriculum", "execution", "collator", "train_split", "validation_split", "checkpoint_root", "resume_checkpoint_digest", "teacher_cache", "reference_cache", "retriever", "trainability"},
        required={"schema", "kind", "run_id", "source_commit", "dataset_manifest_sha256", "base_model", "tokenizer", "architecture", "train_split", "validation_split", "checkpoint_root"},
        label="grounded configuration",
    )
    if root["schema"] != "rigorousrag-advanced-training-config/v1" or root["kind"] != "grounded_generation":
        raise ValueError("configuration is not a grounded_generation v1 training config")
    base = _artifact(root["base_model"], label="base_model")
    if base.artifact_kind not in {"causal_lm", "seq2seq_lm"}:
        raise ValueError("grounded base_model must be causal_lm or seq2seq_lm")
    tokenizer = _artifact(root["tokenizer"], kind="tokenizer", label="tokenizer")
    architecture = GroundedGenerationArchitectureConfig(**_dataclass_kwargs(GroundedGenerationArchitectureConfig, _mapping(root["architecture"], "architecture"), "architecture"))
    curriculum_raw = _mapping(root.get("curriculum") or {}, "curriculum")
    _strict(curriculum_raw, allowed={"include_preference", "stages"}, required=set(), label="curriculum")
    retriever_raw = root.get("retriever")
    retriever_model = retriever_cache = None
    retriever_sha = None
    retriever_config = RetrieverCouplingConfig()
    if retriever_raw is not None:
        retriever_obj = _strict(_mapping(retriever_raw, "retriever"), allowed={"model", "utility_cache", "coupling"}, required={"model", "utility_cache"}, label="retriever")
        retriever_model = _artifact(retriever_obj["model"], kind="sequence_classifier", label="retriever.model")
        retriever_cache = _cache(retriever_obj["utility_cache"], "retriever.utility_cache", expected_kind="document_lm_utility")
        retriever_config = RetrieverCouplingConfig(**_dataclass_kwargs(RetrieverCouplingConfig, _mapping(retriever_obj.get("coupling") or {}, "retriever.coupling"), "retriever.coupling"))
        retriever_sha = retriever_model.expected_sha256
    teacher_cache = _cache(root.get("teacher_cache"), "teacher_cache", expected_kind="teacher_logits")
    reference_cache = _cache(root.get("reference_cache"), "reference_cache", expected_kind="reference_policy_log_probs")
    teacher_sha = teacher_cache.identity.producer_sha256 if teacher_cache is not None else None
    plan, default_trainability = build_grounded_curriculum(
        run_id=root["run_id"], architecture=architecture, base_model_sha256=base.expected_sha256,
        tokenizer_sha256=tokenizer.expected_sha256, dataset_manifest_sha256=root["dataset_manifest_sha256"],
        source_commit=root["source_commit"], retriever_stack_sha256=retriever_sha, teacher_model_sha256=teacher_sha,
        include_preference=bool(curriculum_raw.get("include_preference", True)), hyperparameters=_grounded_hyper(curriculum_raw.get("stages")),
    )
    return GroundedConfiguredRun(
        plan=plan, trainability=_trainability(root.get("trainability"), default_trainability),
        execution=TrainingExecutionConfig(**_dataclass_kwargs(TrainingExecutionConfig, _mapping(root.get("execution") or {}, "execution"), "execution")),
        collator=GroundedCollatorConfig(**_dataclass_kwargs(GroundedCollatorConfig, _mapping(root.get("collator") or {}, "collator"), "collator")),
        train_split=_split(root["train_split"], "train_split"), validation_split=_split(root["validation_split"], "validation_split"),
        checkpoint_root=_checkpoint_root(root["checkpoint_root"]), base_model=base, tokenizer=tokenizer,
        resume_checkpoint_digest=root.get("resume_checkpoint_digest"), teacher_cache=teacher_cache,
        reference_cache=reference_cache, retriever_model=retriever_model,
        retriever_utility_cache=retriever_cache, retriever_coupling=retriever_config,
    )


def load_dynamic_run_config(path: str | Path) -> DynamicConfiguredRun:
    root = _strict(
        _read_json(path),
        allowed={"schema", "kind", "run_id", "source_commit", "dataset_manifest_sha256", "generator", "tokenizer", "architecture", "budget", "retrieval_stack_sha256", "curriculum", "execution", "collator", "train_split", "validation_split", "checkpoint_root", "resume_checkpoint_digest", "hidden_state_cache", "trainability"},
        required={"schema", "kind", "run_id", "source_commit", "dataset_manifest_sha256", "generator", "tokenizer", "architecture", "budget", "retrieval_stack_sha256", "train_split", "validation_split", "checkpoint_root"},
        label="dynamic configuration",
    )
    if root["schema"] != "rigorousrag-advanced-training-config/v1" or root["kind"] != "dynamic_rag_policy":
        raise ValueError("configuration is not a dynamic_rag_policy v1 training config")
    generator = _artifact(root["generator"], label="generator")
    if generator.artifact_kind not in {"causal_lm", "seq2seq_lm"}:
        raise ValueError("dynamic generator must be causal_lm or seq2seq_lm")
    tokenizer = _artifact(root["tokenizer"], kind="tokenizer", label="tokenizer")
    architecture = DynamicPolicyArchitecture(**_dataclass_kwargs(DynamicPolicyArchitecture, _mapping(root["architecture"], "architecture"), "architecture"))
    budget = DynamicRetrievalBudget(**_dataclass_kwargs(DynamicRetrievalBudget, _mapping(root["budget"], "budget"), "budget"))
    curriculum_raw = _mapping(root.get("curriculum") or {}, "curriculum")
    _strict(curriculum_raw, allowed={"include_need_selection", "stages"}, required=set(), label="curriculum")
    plan, default_trainability = build_dynamic_curriculum(
        run_id=root["run_id"], architecture=architecture, budget=budget,
        dataset_manifest_sha256=root["dataset_manifest_sha256"], base_generator_sha256=generator.expected_sha256,
        retrieval_stack_sha256=root["retrieval_stack_sha256"], source_commit=root["source_commit"],
        include_need_selection=bool(curriculum_raw.get("include_need_selection", True)), hyperparameters=_dynamic_hyper(curriculum_raw.get("stages")),
    )
    return DynamicConfiguredRun(
        plan=plan, trainability=_trainability(root.get("trainability"), default_trainability),
        execution=TrainingExecutionConfig(**_dataclass_kwargs(TrainingExecutionConfig, _mapping(root.get("execution") or {}, "execution"), "execution")),
        collator=DynamicCollatorConfig(**_dataclass_kwargs(DynamicCollatorConfig, _mapping(root.get("collator") or {}, "collator"), "collator")),
        train_split=_split(root["train_split"], "train_split"), validation_split=_split(root["validation_split"], "validation_split"),
        checkpoint_root=_checkpoint_root(root["checkpoint_root"]), generator=generator, tokenizer=tokenizer,
        resume_checkpoint_digest=root.get("resume_checkpoint_digest"), hidden_state_cache=_cache(root.get("hidden_state_cache"), "hidden_state_cache", expected_kind="generator_hidden_states"),
    )


def load_advanced_run_config(path: str | Path) -> GroundedConfiguredRun | DynamicConfiguredRun:
    root = _read_json(path)
    kind = root.get("kind")
    if kind == "grounded_generation":
        return load_grounded_run_config(path)
    if kind == "dynamic_rag_policy":
        return load_dynamic_run_config(path)
    raise ValueError("advanced training config kind must be grounded_generation or dynamic_rag_policy")


__all__ = ["DynamicConfiguredRun", "GroundedConfiguredRun", "TensorCacheSpec", "load_advanced_run_config", "load_dynamic_run_config", "load_grounded_run_config"]
