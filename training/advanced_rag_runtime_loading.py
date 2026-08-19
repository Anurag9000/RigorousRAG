"""Strict reconstruction of promoted advanced-RAG inference artifacts.

The loader verifies the content-addressed export directory, re-hashes the manifest and
safetensor bytes, reconstructs repository-owned modules from the canonical runtime payload,
and requires separately admitted local base/tokenizer/retriever trees by exact digest. It
never falls back to network downloads or remote code.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from training.advanced_rag_artifacts import AdvancedArtifactManifest
from training.advanced_rag_models import DynamicRagPolicyModel, GroundedGeneratorTrainingModule
from training.dynamic_retrieval_policy import DynamicPolicyArchitecture, DynamicRetrievalBudget
from training.grounded_generation import GroundedGenerationArchitectureConfig
from training.grounded_supervision_pipeline import PairwiseCandidateRetriever
from training.local_artifact_loading import (
    LocalArtifactTreeBinding,
    load_local_language_model,
    load_local_sequence_classifier,
    load_local_tokenizer,
)
from training.seq2seq_grounded import Seq2SeqGroundedGeneratorTrainingModule

_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_WEIGHTS_BYTES = 2 * 1024 * 1024 * 1024 * 1024


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("advanced RAG runtime loading requires optional PyTorch")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_advanced_artifact_manifest(directory: str | Path) -> AdvancedArtifactManifest:
    """Verify directory/manifest/weight identity without instantiating any model."""
    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("advanced artifact root must be a non-symlink directory")
    children = {item.name: item for item in root.iterdir()}
    if set(children) != {"manifest.json", "model.safetensors"}:
        raise ValueError("advanced artifact directory must contain exactly manifest.json and model.safetensors")
    manifest_path = children["manifest.json"]
    weights_path = children["model.safetensors"]
    for path in (manifest_path, weights_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("advanced artifact files must be regular non-symlink files")
    if manifest_path.stat().st_size <= 0 or manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("advanced artifact manifest exceeds byte safety bound")
    if weights_path.stat().st_size <= 0 or weights_path.stat().st_size > _MAX_WEIGHTS_BYTES:
        raise ValueError("advanced artifact weights exceed byte safety bound")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc:
        raise ValueError("advanced artifact manifest is not strict JSON") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != "rigorousrag-advanced-inference-artifact/v3":
        raise ValueError("unsupported advanced artifact manifest schema")
    required = {field.name for field in fields(AdvancedArtifactManifest)} | {"schema"}
    if set(payload) != required:
        raise ValueError(f"advanced artifact manifest field mismatch: {sorted(set(payload) ^ required)}")
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    artifact_sha = str(payload["artifact_sha256"]).strip().lower()
    if _digest(unsigned) != artifact_sha:
        raise ValueError("advanced artifact manifest digest does not match payload")
    if root.name != artifact_sha:
        raise ValueError("advanced artifact directory name is not its content address")
    manifest_values = {key: value for key, value in payload.items() if key != "schema"}
    manifest_values["included_prefixes"] = tuple(manifest_values["included_prefixes"])
    manifest = AdvancedArtifactManifest(**manifest_values)
    if weights_path.stat().st_size != manifest.weights_bytes or _file_sha(weights_path) != manifest.weights_sha256:
        raise ValueError("advanced artifact safetensor bytes differ from manifest")
    return manifest


def _load_safetensors_strict(model: Any, path: Path) -> None:
    _require_torch()
    try:
        from safetensors.torch import load_file
    except Exception as exc:
        raise RuntimeError("advanced RAG runtime loading requires safetensors") from exc
    state = load_file(str(path), device="cpu")
    if not state:
        raise ValueError("advanced artifact contains no model tensors")
    incompatible = model.load_state_dict(state, strict=True)
    if getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None):
        raise RuntimeError("strict advanced artifact load reported incompatible keys")


@dataclass(frozen=True)
class LoadedGroundedArtifact:
    manifest: AdvancedArtifactManifest
    model: Any
    tokenizer: Any

    def __post_init__(self) -> None:
        if self.manifest.kind != "grounded_generator":
            raise ValueError("LoadedGroundedArtifact requires grounded_generator manifest")


@dataclass(frozen=True)
class LoadedDynamicPolicyArtifact:
    manifest: AdvancedArtifactManifest
    model: Any
    budget: DynamicRetrievalBudget

    def __post_init__(self) -> None:
        if self.manifest.kind != "dynamic_rag_policy":
            raise ValueError("LoadedDynamicPolicyArtifact requires dynamic_rag_policy manifest")


def load_grounded_artifact(
    directory: str | Path,
    *,
    base_model: LocalArtifactTreeBinding,
    tokenizer: LocalArtifactTreeBinding,
    retriever_model: LocalArtifactTreeBinding | None = None,
) -> LoadedGroundedArtifact:
    """Reconstruct a grounded causal/seq2seq wrapper from a verified advanced artifact."""
    manifest = read_advanced_artifact_manifest(directory)
    if manifest.kind != "grounded_generator":
        raise ValueError("artifact is not a grounded generator")
    if base_model.expected_sha256 != manifest.base_model_sha256 or base_model.artifact_kind != manifest.generator_family:
        raise ValueError("base-model binding differs from grounded artifact")
    if tokenizer.expected_sha256 != manifest.tokenizer_sha256 or tokenizer.artifact_kind != "tokenizer":
        raise ValueError("tokenizer binding differs from grounded artifact")
    base = load_local_language_model(base_model)
    tokenization = load_local_tokenizer(tokenizer)
    runtime = manifest.runtime_config
    architecture = GroundedGenerationArchitectureConfig(**dict(runtime["architecture"]))

    retriever = None
    adapter = runtime.get("retriever_adapter")
    if manifest.retrieval_stack_sha256 is not None:
        if retriever_model is None:
            raise ValueError("joint grounded artifact requires retriever_model binding")
        if retriever_model.expected_sha256 != manifest.retrieval_stack_sha256 or retriever_model.artifact_kind != "sequence_classifier":
            raise ValueError("retriever local binding differs from grounded artifact")
        if not isinstance(adapter, Mapping):
            raise ValueError("joint grounded artifact lacks retriever adapter configuration")
        pair_model = load_local_sequence_classifier(retriever_model)
        retriever = PairwiseCandidateRetriever(pair_model, positive_label_index=int(adapter["positive_label_index"]))
    elif retriever_model is not None:
        raise ValueError("retriever_model supplied for artifact that does not include retriever weights")

    if manifest.generator_family == "causal_lm":
        model = GroundedGeneratorTrainingModule(base_model=base, config=architecture, retriever_model=retriever)
    else:
        model = Seq2SeqGroundedGeneratorTrainingModule(base_model=base, config=architecture, retriever_model=retriever)
    _load_safetensors_strict(model, Path(directory).expanduser().resolve(strict=True) / "model.safetensors")
    return LoadedGroundedArtifact(manifest=manifest, model=model, tokenizer=tokenization)


def load_dynamic_policy_artifact(directory: str | Path) -> LoadedDynamicPolicyArtifact:
    """Reconstruct the complete dynamic controller/value/need-selector module."""
    manifest = read_advanced_artifact_manifest(directory)
    if manifest.kind != "dynamic_rag_policy":
        raise ValueError("artifact is not a dynamic RAG policy")
    runtime = manifest.runtime_config
    architecture = DynamicPolicyArchitecture(**dict(runtime["architecture"]))
    budget = DynamicRetrievalBudget(**dict(runtime["budget"]))
    model = DynamicRagPolicyModel(architecture)
    _load_safetensors_strict(model, Path(directory).expanduser().resolve(strict=True) / "model.safetensors")
    return LoadedDynamicPolicyArtifact(manifest=manifest, model=model, budget=budget)


__all__ = [
    "LoadedDynamicPolicyArtifact",
    "LoadedGroundedArtifact",
    "load_dynamic_policy_artifact",
    "load_grounded_artifact",
    "read_advanced_artifact_manifest",
]
