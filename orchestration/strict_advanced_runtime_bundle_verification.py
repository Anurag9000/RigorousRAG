"""Filesystem- and serialization-hardened production runtime-bundle verification.

The historical bundle verifier remains available for compatibility. Production verification is
independent: it rejects symlinked children, validates the closed four-file publication, re-hashes
all bytes, reopens every advanced artifact/promotion source through the strict runtime bridge,
rebuilds the runtime stack/offline-quality evidence, and compares JSON-normalized structures so
tuple-backed dataclasses round-trip correctly through JSON arrays.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from orchestration.advanced_rag_runtime_stack_bridge import (
    advanced_offline_quality_evidence,
    bind_qualified_advanced_artifact,
    build_runtime_stack_with_advanced_bindings,
)
from orchestration.authoritative_advanced_runtime_bundle import (
    AdvancedRuntimeBundleSource,
    AuthoritativeAdvancedRuntimeBundleReceipt,
)
from orchestration.runtime_stack_authority import RuntimeComponent
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_runtime_loading import read_advanced_artifact_manifest
from training.authoritative_advanced_promotion import (
    read_authoritative_advanced_promotion_evidence,
)

_EXPECTED = (
    "stack.json",
    "offline_quality.json",
    "bindings.json",
    "bundle_receipt.json",
)
_MAX_BYTES = 32 * 1024 * 1024


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


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_normalized(value: Any) -> Any:
    return json.loads(_canonical(value).decode("utf-8"))


def _strict_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink():
        raise ValueError(f"{label} may not be a symlink")
    selected = safe_advanced_path(
        path,
        label=label,
        must_exist=True,
        require_file=True,
    )
    size = selected.stat().st_size
    if size <= 0 or size > _MAX_BYTES:
        raise ValueError(f"{label} exceeds byte safety bound")
    try:
        value = json.loads(
            selected.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _runtime_component(raw: Any) -> RuntimeComponent:
    required = {"kind", "component_id", "artifact_sha256", "contract_sha256"}
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("runtime bundle other component fields are invalid")
    return RuntimeComponent(
        kind=raw["kind"],
        component_id=raw["component_id"],
        artifact_sha256=raw["artifact_sha256"],
        contract_sha256=raw["contract_sha256"],
    )


def _source(raw: Any) -> AdvancedRuntimeBundleSource:
    required = {
        "artifact_directory",
        "promotion_evidence_path",
        "component_id",
        "binding_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("advanced runtime bundle source fields are invalid")
    return AdvancedRuntimeBundleSource(**dict(raw))


def verify_strict_authoritative_advanced_runtime_bundle(
    receipt_path: str | Path,
) -> AuthoritativeAdvancedRuntimeBundleReceipt:
    raw_receipt_path = Path(receipt_path).expanduser()
    if raw_receipt_path.is_symlink():
        raise ValueError("advanced runtime bundle receipt may not be a symlink")
    receipt_path_selected = safe_advanced_path(
        raw_receipt_path,
        label="advanced runtime bundle receipt",
        must_exist=True,
        require_file=True,
    )
    root = receipt_path_selected.parent
    if receipt_path_selected != root / "bundle_receipt.json":
        raise ValueError("advanced runtime bundle receipt must use canonical filename")

    children = {item.name for item in root.iterdir()}
    if children != set(_EXPECTED):
        raise ValueError("advanced runtime bundle directory is not closed")
    for name in _EXPECTED:
        raw_child = root / name
        if raw_child.is_symlink():
            raise ValueError(f"advanced runtime bundle child {name} may not be a symlink")
        child = safe_advanced_path(
            raw_child,
            label=f"advanced runtime bundle {name}",
            must_exist=True,
            require_file=True,
        )
        if child.parent != root or child.name != name:
            raise ValueError(f"advanced runtime bundle child {name} escapes canonical root")

    raw = _strict_json(root / "bundle_receipt.json", "advanced runtime bundle receipt")
    required = {
        "schema",
        "stack_sha256",
        "offline_quality_evidence_sha256",
        "source_set_sha256",
        "other_components_sha256",
        "stack_config_sha256",
        "sources",
        "stack_file_sha256",
        "offline_quality_file_sha256",
        "bindings_file_sha256",
        "receipt_sha256",
    }
    if (
        set(raw) != required
        or raw.get("schema")
        != "rigorousrag-authoritative-advanced-runtime-bundle-receipt/v1"
        or not isinstance(raw.get("sources"), list)
    ):
        raise ValueError("unsupported advanced runtime bundle receipt schema")
    sources = tuple(_source(item) for item in raw["sources"])
    receipt = AuthoritativeAdvancedRuntimeBundleReceipt(
        stack_sha256=raw["stack_sha256"],
        offline_quality_evidence_sha256=raw["offline_quality_evidence_sha256"],
        source_set_sha256=raw["source_set_sha256"],
        other_components_sha256=raw["other_components_sha256"],
        stack_config_sha256=raw["stack_config_sha256"],
        sources=sources,
        stack_file_sha256=raw["stack_file_sha256"],
        offline_quality_file_sha256=raw["offline_quality_file_sha256"],
        bindings_file_sha256=raw["bindings_file_sha256"],
        receipt_sha256=raw["receipt_sha256"],
    )

    if _file_sha(root / "stack.json") != receipt.stack_file_sha256:
        raise ValueError("runtime stack bytes differ from bundle receipt")
    if _file_sha(root / "offline_quality.json") != receipt.offline_quality_file_sha256:
        raise ValueError("runtime offline-quality bytes differ from bundle receipt")
    if _file_sha(root / "bindings.json") != receipt.bindings_file_sha256:
        raise ValueError("runtime bindings bytes differ from bundle receipt")

    bindings_raw = _strict_json(root / "bindings.json", "advanced runtime bindings")
    if (
        set(bindings_raw) != {"schema", "sources", "other_components", "stack_config"}
        or bindings_raw.get("schema") != "rigorousrag-advanced-runtime-bindings/v1"
        or not isinstance(bindings_raw.get("sources"), list)
        or not isinstance(bindings_raw.get("other_components"), list)
        or not isinstance(bindings_raw.get("stack_config"), Mapping)
    ):
        raise ValueError("advanced runtime bindings file is malformed")
    if _digest(bindings_raw["sources"]) != receipt.source_set_sha256:
        raise ValueError("runtime source-set digest differs from receipt")
    if _digest(bindings_raw["other_components"]) != receipt.other_components_sha256:
        raise ValueError("runtime other-component digest differs from receipt")
    if _digest(bindings_raw["stack_config"]) != receipt.stack_config_sha256:
        raise ValueError("runtime stack-config digest differs from receipt")

    by_component = {item.component_id: item for item in receipt.sources}
    if len(bindings_raw["sources"]) != len(by_component):
        raise ValueError("runtime source count differs from receipt")
    reconstructed = []
    for raw_source in bindings_raw["sources"]:
        selected_source = _source(raw_source)
        expected = by_component.get(selected_source.component_id)
        if expected is None or _json_normalized(asdict(expected)) != dict(raw_source):
            raise ValueError("advanced runtime source differs from receipt")
        artifact = safe_advanced_path(
            selected_source.artifact_directory,
            label="advanced runtime source artifact",
            must_exist=True,
            require_directory=True,
        )
        promotion_path = safe_advanced_path(
            selected_source.promotion_evidence_path,
            label="advanced runtime source promotion",
            must_exist=True,
            require_file=True,
        )
        manifest = read_advanced_artifact_manifest(artifact)
        promotion = read_authoritative_advanced_promotion_evidence(promotion_path)
        binding = bind_qualified_advanced_artifact(
            artifact,
            manifest,
            promotion,
            component_id=selected_source.component_id,
        )
        if binding.binding_sha256 != selected_source.binding_sha256:
            raise ValueError("reconstructed advanced runtime binding differs from receipt")
        reconstructed.append(binding)

    others = tuple(_runtime_component(item) for item in bindings_raw["other_components"])
    config = bindings_raw["stack_config"]
    required_config = {
        "schema",
        "stack_id",
        "retrieval_contract_sha256",
        "generation_contract_sha256",
        "compatibility_sha256",
        "source_revision",
        "valid_from",
        "expires_at",
    }
    if (
        set(config) != required_config
        or config.get("schema")
        != "rigorousrag-authoritative-advanced-runtime-stack-config/v1"
    ):
        raise ValueError("advanced runtime stack config is malformed")

    stack = build_runtime_stack_with_advanced_bindings(
        stack_id=config["stack_id"],
        advanced_bindings=tuple(reconstructed),
        other_components=others,
        retrieval_contract_sha256=config["retrieval_contract_sha256"],
        generation_contract_sha256=config["generation_contract_sha256"],
        compatibility_sha256=config["compatibility_sha256"],
        source_revision=config["source_revision"],
    )
    quality = advanced_offline_quality_evidence(
        stack,
        tuple(reconstructed),
        valid_from=config["valid_from"],
        expires_at=config["expires_at"],
    )
    if stack.stack_sha256 != receipt.stack_sha256:
        raise ValueError("reconstructed runtime stack identity differs from receipt")
    if quality.evidence_sha256 != receipt.offline_quality_evidence_sha256:
        raise ValueError("reconstructed runtime quality identity differs from receipt")

    stack_raw = _strict_json(root / "stack.json", "runtime stack artifact")
    if set(stack_raw) != {"schema", "stack", "stack_sha256"} or stack_raw.get(
        "schema"
    ) != "rigorousrag-runtime-stack-artifact/v1":
        raise ValueError("runtime stack artifact file is malformed")
    if stack_raw["stack_sha256"] != stack.stack_sha256:
        raise ValueError("persisted runtime stack digest differs from reconstruction")
    if stack_raw["stack"] != _json_normalized(asdict(stack)):
        raise ValueError("persisted runtime stack fields differ from reconstruction")

    quality_raw = _strict_json(
        root / "offline_quality.json",
        "runtime offline quality evidence",
    )
    if set(quality_raw) != {"schema", "evidence", "evidence_sha256"} or quality_raw.get(
        "schema"
    ) != "rigorousrag-runtime-offline-quality-evidence/v1":
        raise ValueError("runtime offline-quality evidence file is malformed")
    if quality_raw["evidence_sha256"] != quality.evidence_sha256:
        raise ValueError("persisted runtime quality digest differs from reconstruction")
    if quality_raw["evidence"] != _json_normalized(asdict(quality)):
        raise ValueError("persisted runtime quality fields differ from reconstruction")
    return receipt


__all__ = ["verify_strict_authoritative_advanced_runtime_bundle"]
