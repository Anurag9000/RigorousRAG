"""Restart-verifiable production bundle for promoted advanced-RAG runtime components.

This module does not replace the mature mutable runtime-stack authority. It closes the source
handoff immediately before it: exact promoted advanced artifacts are re-verified, converted
through ``advanced_rag_runtime_stack_bridge``, assembled into the existing ``RuntimeStackArtifact``
and ``RuntimePromotionEvidence`` types, and persisted in one closed content-addressed bundle.
Verification reconstructs every binding from source artifact/promotion files and rebuilds the
stack/evidence rather than trusting serialized dataclass fields.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from orchestration.advanced_rag_runtime_stack_bridge import (
    AdvancedRuntimeComponentBinding,
    advanced_offline_quality_evidence,
    bind_qualified_advanced_artifact,
    build_runtime_stack_with_advanced_bindings,
)
from orchestration.runtime_stack_authority import RuntimeComponent
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_runtime_loading import read_advanced_artifact_manifest
from training.authoritative_advanced_promotion import (
    read_authoritative_advanced_promotion_evidence,
)

_MAX_CONFIG_BYTES = 16 * 1024 * 1024
_MAX_RECEIPT_BYTES = 32 * 1024 * 1024
_MAX_COMPONENTS = 1_000
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


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path, label: str, maximum_bytes: int) -> Mapping[str, Any]:
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{label} exceeds byte safety bound")
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _atomic(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class AdvancedRuntimeBundleSource:
    artifact_directory: str
    promotion_evidence_path: str
    component_id: str
    binding_sha256: str

    def __post_init__(self) -> None:
        artifact = safe_advanced_path(self.artifact_directory, label="advanced artifact directory", must_exist=True, require_directory=True)
        promotion = safe_advanced_path(self.promotion_evidence_path, label="authoritative promotion evidence", must_exist=True, require_file=True)
        component = str(self.component_id).strip()
        if not component or len(component) > 1_000 or any(ord(ch) < 32 or ord(ch) == 127 for ch in component):
            raise ValueError("component_id is invalid")
        object.__setattr__(self, "artifact_directory", str(artifact))
        object.__setattr__(self, "promotion_evidence_path", str(promotion))
        object.__setattr__(self, "component_id", component)
        object.__setattr__(self, "binding_sha256", _sha(self.binding_sha256, "binding_sha256"))


@dataclass(frozen=True)
class AuthoritativeAdvancedRuntimeBundleReceipt:
    stack_sha256: str
    offline_quality_evidence_sha256: str
    source_set_sha256: str
    other_components_sha256: str
    stack_config_sha256: str
    sources: tuple[AdvancedRuntimeBundleSource, ...]
    stack_file_sha256: str
    offline_quality_file_sha256: str
    bindings_file_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "stack_sha256", "offline_quality_evidence_sha256", "source_set_sha256",
            "other_components_sha256", "stack_config_sha256", "stack_file_sha256",
            "offline_quality_file_sha256", "bindings_file_sha256", "receipt_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        sources = tuple(self.sources)
        if not sources or len(sources) > _MAX_COMPONENTS or any(not isinstance(item, AdvancedRuntimeBundleSource) for item in sources):
            raise ValueError("runtime bundle sources must be bounded and non-empty")
        if len({item.component_id for item in sources}) != len(sources):
            raise ValueError("advanced runtime component IDs must be unique")
        object.__setattr__(self, "sources", tuple(sorted(sources, key=lambda item: item.component_id)))
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("authoritative advanced runtime bundle receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-authoritative-advanced-runtime-bundle-receipt/v1",
            "stack_sha256": self.stack_sha256,
            "offline_quality_evidence_sha256": self.offline_quality_evidence_sha256,
            "source_set_sha256": self.source_set_sha256,
            "other_components_sha256": self.other_components_sha256,
            "stack_config_sha256": self.stack_config_sha256,
            "sources": [asdict(item) for item in self.sources],
            "stack_file_sha256": self.stack_file_sha256,
            "offline_quality_file_sha256": self.offline_quality_file_sha256,
            "bindings_file_sha256": self.bindings_file_sha256,
        }


def _runtime_component(raw: Mapping[str, Any]) -> RuntimeComponent:
    required = {"kind", "component_id", "artifact_sha256", "contract_sha256"}
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("other runtime component fields are invalid")
    return RuntimeComponent(
        kind=raw["kind"], component_id=raw["component_id"], artifact_sha256=raw["artifact_sha256"], contract_sha256=raw["contract_sha256"]
    )


def _binding(artifact_directory: str | Path, promotion_path: str | Path, component_id: str) -> AdvancedRuntimeComponentBinding:
    artifact = safe_advanced_path(artifact_directory, label="advanced artifact directory", must_exist=True, require_directory=True)
    manifest = read_advanced_artifact_manifest(artifact)
    promotion = read_authoritative_advanced_promotion_evidence(promotion_path)
    return bind_qualified_advanced_artifact(artifact, manifest, promotion, component_id=component_id)


def _config_payload(
    *, stack_id: str, retrieval_contract_sha256: str, generation_contract_sha256: str,
    compatibility_sha256: str, source_revision: str, valid_from: float, expires_at: float | None,
) -> Mapping[str, Any]:
    return {
        "schema": "rigorousrag-authoritative-advanced-runtime-stack-config/v1",
        "stack_id": stack_id,
        "retrieval_contract_sha256": retrieval_contract_sha256,
        "generation_contract_sha256": generation_contract_sha256,
        "compatibility_sha256": compatibility_sha256,
        "source_revision": source_revision,
        "valid_from": valid_from,
        "expires_at": expires_at,
    }


def build_authoritative_advanced_runtime_bundle(
    *,
    advanced_sources: Sequence[Mapping[str, Any]],
    other_components: Sequence[Mapping[str, Any]],
    stack_id: str,
    retrieval_contract_sha256: str,
    generation_contract_sha256: str,
    compatibility_sha256: str,
    source_revision: str,
    valid_from: float,
    expires_at: float | None,
    output_dir: str | Path,
) -> AuthoritativeAdvancedRuntimeBundleReceipt:
    selected_sources = tuple(advanced_sources)
    if not selected_sources or len(selected_sources) > _MAX_COMPONENTS:
        raise ValueError("advanced_sources must be bounded and non-empty")
    if len(other_components) > _MAX_COMPONENTS:
        raise ValueError("other_components exceeds safety bound")
    bindings = []
    source_receipts = []
    seen_components: set[str] = set()
    for index, raw in enumerate(selected_sources):
        if not isinstance(raw, Mapping) or set(raw) != {"artifact_directory", "promotion_evidence_path", "component_id"}:
            raise ValueError(f"advanced source {index} fields are invalid")
        component_id = str(raw["component_id"]).strip()
        if component_id in seen_components:
            raise ValueError(f"duplicate advanced runtime component_id {component_id!r}")
        seen_components.add(component_id)
        binding = _binding(raw["artifact_directory"], raw["promotion_evidence_path"], component_id)
        bindings.append(binding)
        source_receipts.append(AdvancedRuntimeBundleSource(str(raw["artifact_directory"]), str(raw["promotion_evidence_path"]), component_id, binding.binding_sha256))
    others = tuple(_runtime_component(raw) for raw in other_components)
    if any(item.component_id in seen_components for item in others):
        raise ValueError("other runtime component ID collides with advanced component ID")
    stack = build_runtime_stack_with_advanced_bindings(
        stack_id=stack_id,
        advanced_bindings=tuple(bindings),
        other_components=others,
        retrieval_contract_sha256=retrieval_contract_sha256,
        generation_contract_sha256=generation_contract_sha256,
        compatibility_sha256=compatibility_sha256,
        source_revision=source_revision,
    )
    quality = advanced_offline_quality_evidence(stack, tuple(bindings), valid_from=valid_from, expires_at=expires_at)
    config_payload = _config_payload(
        stack_id=stack_id, retrieval_contract_sha256=retrieval_contract_sha256,
        generation_contract_sha256=generation_contract_sha256, compatibility_sha256=compatibility_sha256,
        source_revision=source_revision, valid_from=valid_from, expires_at=expires_at,
    )
    source_set_sha = _digest([asdict(item) for item in sorted(source_receipts, key=lambda item: item.component_id)])
    other_sha = _digest([asdict(item) for item in others])
    config_sha = _digest(config_payload)
    root = safe_advanced_path(output_dir, label="advanced runtime bundle output", must_exist=False)
    if root.exists():
        raise ValueError("advanced runtime bundle output must not already exist")
    parent = safe_advanced_path(root.parent, label="advanced runtime bundle parent", must_exist=True, require_directory=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name or 'runtime'}-stage-", dir=parent))
    published = False
    try:
        _atomic(stage / "stack.json", _canonical({"schema": "rigorousrag-runtime-stack-artifact/v1", "stack": asdict(stack), "stack_sha256": stack.stack_sha256}) + b"\n")
        _atomic(stage / "offline_quality.json", _canonical({"schema": "rigorousrag-runtime-offline-quality-evidence/v1", "evidence": asdict(quality), "evidence_sha256": quality.evidence_sha256}) + b"\n")
        _atomic(stage / "bindings.json", _canonical({"schema": "rigorousrag-advanced-runtime-bindings/v1", "sources": [asdict(item) for item in sorted(source_receipts, key=lambda item: item.component_id)], "other_components": [asdict(item) for item in others], "stack_config": config_payload}) + b"\n")
        stack_sha = _file_sha(stage / "stack.json"); quality_sha = _file_sha(stage / "offline_quality.json"); bindings_sha = _file_sha(stage / "bindings.json")
        unsigned = {
            "schema": "rigorousrag-authoritative-advanced-runtime-bundle-receipt/v1",
            "stack_sha256": stack.stack_sha256,
            "offline_quality_evidence_sha256": quality.evidence_sha256,
            "source_set_sha256": source_set_sha,
            "other_components_sha256": other_sha,
            "stack_config_sha256": config_sha,
            "sources": [asdict(item) for item in sorted(source_receipts, key=lambda item: item.component_id)],
            "stack_file_sha256": stack_sha,
            "offline_quality_file_sha256": quality_sha,
            "bindings_file_sha256": bindings_sha,
        }
        receipt = AuthoritativeAdvancedRuntimeBundleReceipt(
            stack_sha256=stack.stack_sha256,
            offline_quality_evidence_sha256=quality.evidence_sha256,
            source_set_sha256=source_set_sha,
            other_components_sha256=other_sha,
            stack_config_sha256=config_sha,
            sources=tuple(source_receipts),
            stack_file_sha256=stack_sha,
            offline_quality_file_sha256=quality_sha,
            bindings_file_sha256=bindings_sha,
            receipt_sha256=_digest(unsigned),
        )
        _atomic(stage / "bundle_receipt.json", _canonical({**receipt.unsigned(), "receipt_sha256": receipt.receipt_sha256}) + b"\n")
        expected = {"stack.json", "offline_quality.json", "bindings.json", "bundle_receipt.json"}
        if {item.name for item in stage.iterdir()} != expected:
            raise RuntimeError("advanced runtime bundle staging directory is not closed")
        os.replace(stage, root); published = True
        return receipt
    except Exception:
        if published: shutil.rmtree(root, ignore_errors=True)
        else: shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_authoritative_advanced_runtime_bundle(
    receipt_path: str | Path,
) -> AuthoritativeAdvancedRuntimeBundleReceipt:
    source = safe_advanced_path(receipt_path, label="advanced runtime bundle receipt", must_exist=True, require_file=True)
    root = source.parent
    if source != root / "bundle_receipt.json":
        raise ValueError("advanced runtime bundle receipt must use canonical filename")
    expected_files = {"stack.json", "offline_quality.json", "bindings.json", "bundle_receipt.json"}
    if {item.name for item in root.iterdir()} != expected_files:
        raise ValueError("advanced runtime bundle directory is not closed")
    raw = _strict_json(source, "advanced runtime bundle receipt", _MAX_RECEIPT_BYTES)
    required = {"schema", "stack_sha256", "offline_quality_evidence_sha256", "source_set_sha256", "other_components_sha256", "stack_config_sha256", "sources", "stack_file_sha256", "offline_quality_file_sha256", "bindings_file_sha256", "receipt_sha256"}
    if set(raw) != required or raw.get("schema") != "rigorousrag-authoritative-advanced-runtime-bundle-receipt/v1" or not isinstance(raw.get("sources"), list):
        raise ValueError("unsupported advanced runtime bundle receipt schema")
    source_fields = {"artifact_directory", "promotion_evidence_path", "component_id", "binding_sha256"}
    source_receipts = []
    for item in raw["sources"]:
        if not isinstance(item, Mapping) or set(item) != source_fields:
            raise ValueError("advanced runtime bundle source fields are invalid")
        source_receipts.append(AdvancedRuntimeBundleSource(**dict(item)))
    receipt = AuthoritativeAdvancedRuntimeBundleReceipt(
        stack_sha256=raw["stack_sha256"], offline_quality_evidence_sha256=raw["offline_quality_evidence_sha256"],
        source_set_sha256=raw["source_set_sha256"], other_components_sha256=raw["other_components_sha256"],
        stack_config_sha256=raw["stack_config_sha256"], sources=tuple(source_receipts), stack_file_sha256=raw["stack_file_sha256"],
        offline_quality_file_sha256=raw["offline_quality_file_sha256"], bindings_file_sha256=raw["bindings_file_sha256"], receipt_sha256=raw["receipt_sha256"],
    )
    if _file_sha(root / "stack.json") != receipt.stack_file_sha256 or _file_sha(root / "offline_quality.json") != receipt.offline_quality_file_sha256 or _file_sha(root / "bindings.json") != receipt.bindings_file_sha256:
        raise ValueError("advanced runtime bundle component bytes differ from receipt")
    bindings_raw = _strict_json(root / "bindings.json", "advanced runtime bindings", _MAX_RECEIPT_BYTES)
    if set(bindings_raw) != {"schema", "sources", "other_components", "stack_config"} or bindings_raw.get("schema") != "rigorousrag-advanced-runtime-bindings/v1" or not isinstance(bindings_raw["sources"], list) or not isinstance(bindings_raw["other_components"], list) or not isinstance(bindings_raw["stack_config"], Mapping):
        raise ValueError("advanced runtime bindings file is malformed")
    if _digest(bindings_raw["sources"]) != receipt.source_set_sha256 or _digest(bindings_raw["other_components"]) != receipt.other_components_sha256 or _digest(bindings_raw["stack_config"]) != receipt.stack_config_sha256:
        raise ValueError("advanced runtime binding/config digests differ from receipt")
    reconstructed_bindings = []
    if len(bindings_raw["sources"]) != len(receipt.sources):
        raise ValueError("advanced runtime source count differs from receipt")
    by_component = {item.component_id: item for item in receipt.sources}
    for raw_source in bindings_raw["sources"]:
        if not isinstance(raw_source, Mapping) or set(raw_source) != source_fields:
            raise ValueError("advanced runtime source binding is malformed")
        expected_source = by_component.get(str(raw_source["component_id"]))
        if expected_source is None or asdict(expected_source) != dict(raw_source):
            raise ValueError("advanced runtime source differs from receipt")
        binding = _binding(raw_source["artifact_directory"], raw_source["promotion_evidence_path"], raw_source["component_id"])
        if binding.binding_sha256 != raw_source["binding_sha256"]:
            raise ValueError("reconstructed advanced runtime binding differs")
        reconstructed_bindings.append(binding)
    others = tuple(_runtime_component(item) for item in bindings_raw["other_components"])
    config = bindings_raw["stack_config"]
    required_config = {"schema", "stack_id", "retrieval_contract_sha256", "generation_contract_sha256", "compatibility_sha256", "source_revision", "valid_from", "expires_at"}
    if set(config) != required_config or config.get("schema") != "rigorousrag-authoritative-advanced-runtime-stack-config/v1":
        raise ValueError("advanced runtime stack config is malformed")
    stack = build_runtime_stack_with_advanced_bindings(
        stack_id=config["stack_id"], advanced_bindings=tuple(reconstructed_bindings), other_components=others,
        retrieval_contract_sha256=config["retrieval_contract_sha256"], generation_contract_sha256=config["generation_contract_sha256"],
        compatibility_sha256=config["compatibility_sha256"], source_revision=config["source_revision"],
    )
    quality = advanced_offline_quality_evidence(stack, tuple(reconstructed_bindings), valid_from=config["valid_from"], expires_at=config["expires_at"])
    if stack.stack_sha256 != receipt.stack_sha256 or quality.evidence_sha256 != receipt.offline_quality_evidence_sha256:
        raise ValueError("reconstructed runtime stack/offline quality differs from receipt")
    stack_raw = _strict_json(root / "stack.json", "runtime stack artifact", _MAX_RECEIPT_BYTES)
    quality_raw = _strict_json(root / "offline_quality.json", "runtime offline quality evidence", _MAX_RECEIPT_BYTES)
    if stack_raw.get("stack_sha256") != stack.stack_sha256 or stack_raw.get("stack") != asdict(stack):
        raise ValueError("persisted runtime stack differs from reconstruction")
    if quality_raw.get("evidence_sha256") != quality.evidence_sha256 or quality_raw.get("evidence") != asdict(quality):
        raise ValueError("persisted runtime quality evidence differs from reconstruction")
    return receipt


__all__ = ["AdvancedRuntimeBundleSource", "AuthoritativeAdvancedRuntimeBundleReceipt", "build_authoritative_advanced_runtime_bundle", "verify_authoritative_advanced_runtime_bundle"]
