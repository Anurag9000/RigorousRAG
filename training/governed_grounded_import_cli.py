"""Config-only CLI for governed grounded-training dataset conversion."""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.dataset_governance import DatasetCard, DatasetModality, DatasetTask, LicenseStatus
from training.advanced_path_authority import safe_advanced_path
from training.governed_grounded_import import (
    DeclarativeGroundedProfile,
    GroundedDatasetGovernanceSpec,
    GroundedSplitImportSpec,
    import_governed_grounded_dataset,
)
from training.governed_grounded_io import verify_governed_grounded_import
from training.grounded_generation import ReflectionAction

_MAX_CONFIG_BYTES = 16 * 1024 * 1024


def _read(path: str | Path) -> Mapping[str, Any]:
    source = safe_advanced_path(path, label="grounded import config", must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("grounded import config exceeds byte safety bound")
    try:
        payload = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc:
        raise ValueError("grounded import config is not strict JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("grounded import config must contain an object")
    return payload


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings")
    return tuple(value)


def _card(raw: Any) -> DatasetCard:
    if not isinstance(raw, Mapping):
        raise ValueError("governance.card must be an object")
    allowed = {field.name for field in fields(DatasetCard)}
    if set(raw) - allowed:
        raise ValueError(f"governance.card has unsupported fields: {sorted(set(raw)-allowed)}")
    return DatasetCard(
        summary=raw.get("summary", ""), intended_uses=_strings(raw.get("intended_uses", []), "card.intended_uses"),
        forbidden_uses=_strings(raw.get("forbidden_uses", []), "card.forbidden_uses"), populations_or_domains=_strings(raw.get("populations_or_domains", []), "card.populations_or_domains"),
        languages=_strings(raw.get("languages", []), "card.languages"), pii_notes=raw.get("pii_notes"), safety_notes=raw.get("safety_notes"),
        source_citation=raw.get("source_citation"), known_limitations=_strings(raw.get("known_limitations", []), "card.known_limitations"),
    )


def _profile(name: str, raw: Any) -> DeclarativeGroundedProfile:
    if not isinstance(raw, Mapping):
        raise ValueError(f"profile {name} must be an object")
    allowed = {field.name for field in fields(DeclarativeGroundedProfile)} - {"name"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"profile {name} has unsupported fields: {sorted(unknown)}")
    tuple_fields = {
        "id_paths", "prompt_paths", "answer_paths", "evidence_id_paths", "evidence_text_paths", "evidence_source_paths",
        "claim_start_paths", "claim_end_paths", "claim_text_paths", "claim_evidence_id_paths", "claim_supporting_id_paths", "claim_contradicting_id_paths",
        "claim_supported_paths", "claim_contradicted_paths", "unsupported_start_paths", "unsupported_end_paths", "unsupported_text_paths", "abstain_paths",
        "reflection_action_paths", "chosen_answer_paths", "rejected_answer_paths",
    }
    kwargs: dict[str, Any] = {"name": name}
    for field_name in tuple_fields:
        kwargs[field_name] = _strings(raw.get(field_name, []), f"profile {name}.{field_name}")
    for field_name in ("evidence_path", "claims_path", "unsupported_spans_path"):
        if field_name in raw:
            kwargs[field_name] = raw[field_name]
    for field_name in ("constant_abstain", "generate_missing_ids"):
        if field_name in raw:
            if not isinstance(raw[field_name], bool):
                raise ValueError(f"profile {name}.{field_name} must be boolean")
            kwargs[field_name] = raw[field_name]
    if "constant_reflection_action" in raw and raw["constant_reflection_action"] is not None:
        kwargs["constant_reflection_action"] = ReflectionAction(raw["constant_reflection_action"])
    metadata_paths = raw.get("metadata_paths", {})
    if not isinstance(metadata_paths, Mapping):
        raise ValueError(f"profile {name}.metadata_paths must be an object")
    kwargs["metadata_paths"] = {str(key): _strings(value, f"profile {name}.metadata_paths.{key}") for key, value in metadata_paths.items()}
    constants = raw.get("constant_metadata", {})
    if not isinstance(constants, Mapping):
        raise ValueError(f"profile {name}.constant_metadata must be an object")
    kwargs["constant_metadata"] = {str(key): str(value) for key, value in constants.items()}
    return DeclarativeGroundedProfile(**kwargs)


def run_import_config(path: str | Path) -> Mapping[str, Any]:
    raw = _read(path)
    if set(raw) != {"schema", "output_dir", "governance", "profiles", "splits"} or raw.get("schema") != "rigorousrag-governed-grounded-import-config/v1":
        raise ValueError("config must be rigorousrag-governed-grounded-import-config/v1")
    governance_raw = raw["governance"]
    if not isinstance(governance_raw, Mapping):
        raise ValueError("governance must be an object")
    allowed_governance = {"dataset_id", "exact_version", "source_locator", "license_identifier", "license_status", "license_evidence", "tasks", "modalities", "card", "metadata", "require_promotable"}
    unknown = set(governance_raw) - allowed_governance
    if unknown:
        raise ValueError(f"governance has unsupported fields: {sorted(unknown)}")
    tasks_raw, modalities_raw = governance_raw.get("tasks"), governance_raw.get("modalities")
    if not isinstance(tasks_raw, list) or not tasks_raw or not isinstance(modalities_raw, list) or not modalities_raw:
        raise ValueError("governance tasks/modalities must be non-empty arrays")
    metadata = governance_raw.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("governance.metadata must be an object")
    governance = GroundedDatasetGovernanceSpec(
        dataset_id=governance_raw.get("dataset_id"), exact_version=governance_raw.get("exact_version"), source_locator=governance_raw.get("source_locator"),
        license_identifier=governance_raw.get("license_identifier"), license_status=LicenseStatus(governance_raw.get("license_status")), license_evidence=governance_raw.get("license_evidence"),
        tasks=tuple(DatasetTask(item) for item in tasks_raw), modalities=tuple(DatasetModality(item) for item in modalities_raw), card=_card(governance_raw.get("card")),
        metadata={str(key): str(value) for key, value in metadata.items()}, require_promotable=bool(governance_raw.get("require_promotable", False)),
    )
    profiles_raw = raw["profiles"]
    if not isinstance(profiles_raw, Mapping) or not profiles_raw:
        raise ValueError("profiles must be a non-empty object")
    profiles = {str(name): _profile(str(name), value) for name, value in profiles_raw.items()}
    splits_raw = raw["splits"]
    if not isinstance(splits_raw, list) or not splits_raw:
        raise ValueError("splits must be a non-empty array")
    splits = []
    allowed_split = {"name", "source_path", "source_sha256", "profile", "input_format", "expected_record_count"}
    for index, item in enumerate(splits_raw):
        if not isinstance(item, Mapping) or set(item) - allowed_split:
            raise ValueError(f"split {index} has invalid fields")
        profile_name = str(item.get("profile", ""))
        if profile_name not in profiles:
            raise ValueError(f"split {index} references unknown profile {profile_name!r}")
        splits.append(GroundedSplitImportSpec(
            name=item.get("name"), source_path=item.get("source_path"), source_sha256=item.get("source_sha256"), profile=profiles[profile_name],
            input_format=item.get("input_format", "jsonl"), expected_record_count=item.get("expected_record_count"),
        ))
    manifest, receipt = import_governed_grounded_dataset(governance, tuple(splits), output_dir=raw["output_dir"])
    verified = verify_governed_grounded_import(Path(raw["output_dir"]) / "import_receipt.json", require_promotable=governance.require_promotable)
    if verified.manifest.manifest_digest != manifest.manifest_digest or verified.receipt.receipt_sha256 != receipt.receipt_sha256:
        raise RuntimeError("grounded import read-side verification returned a different identity")
    return {
        "dataset_id": manifest.dataset_id,
        "dataset_manifest_sha256": manifest.manifest_digest,
        "source_set_sha256": receipt.source_set_sha256,
        "transformation_sha256": receipt.transformation_sha256,
        "receipt_sha256": receipt.receipt_sha256,
        "splits": {item.name: {"path": item.output_path, "sha256": item.output_sha256, "record_count": item.record_count} for item in receipt.splits},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert exact local annotated data into governed grounded-generator training JSONL")
    parser.add_argument("config", help="rigorousrag-governed-grounded-import-config/v1 JSON file")
    result = run_import_config(parser.parse_args(argv).config)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["run_import_config"]
