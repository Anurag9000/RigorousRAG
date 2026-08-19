"""Strict config-only command for publishing dynamic-RAG trajectory datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.dataset_governance import DatasetCard, LicenseStatus
from training.advanced_path_authority import safe_advanced_path
from training.dynamic_dataset_io import verify_dynamic_dataset_publication
from training.dynamic_dataset_publication import DynamicDatasetGovernance, DynamicTrajectorySource, EpisodeSplitPolicy, publish_dynamic_training_dataset

_MAX_BYTES = 16 * 1024 * 1024


def _read(path: str | Path) -> Mapping[str, Any]:
    source = safe_advanced_path(path, label="dynamic dataset publication config", must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES: raise ValueError("dynamic publication config exceeds byte safety bound")
    try: value = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc: raise ValueError("dynamic publication config is not strict JSON") from exc
    if not isinstance(value, Mapping): raise ValueError("dynamic publication config must contain an object")
    return value


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value): raise ValueError(f"{label} must be an array of strings")
    return tuple(value)


def _card(raw: Any) -> DatasetCard:
    if not isinstance(raw, Mapping): raise ValueError("governance.card must be an object")
    allowed = {"summary", "intended_uses", "forbidden_uses", "populations_or_domains", "languages", "pii_notes", "safety_notes", "source_citation", "known_limitations"}
    if set(raw) - allowed: raise ValueError(f"governance.card contains unsupported fields: {sorted(set(raw)-allowed)}")
    return DatasetCard(summary=raw.get("summary", ""), intended_uses=_strings(raw.get("intended_uses", []), "intended_uses"), forbidden_uses=_strings(raw.get("forbidden_uses", []), "forbidden_uses"), populations_or_domains=_strings(raw.get("populations_or_domains", []), "populations_or_domains"), languages=_strings(raw.get("languages", []), "languages"), pii_notes=raw.get("pii_notes"), safety_notes=raw.get("safety_notes"), source_citation=raw.get("source_citation"), known_limitations=_strings(raw.get("known_limitations", []), "known_limitations"))


def run_dynamic_publication_config(path: str | Path) -> Mapping[str, Any]:
    raw = _read(path)
    required = {"schema", "output_dir", "governance", "split_policy", "sources"}
    if set(raw) != required or raw.get("schema") != "rigorousrag-dynamic-dataset-publication-config/v1": raise ValueError("config must be rigorousrag-dynamic-dataset-publication-config/v1")
    governance_raw = raw["governance"]
    if not isinstance(governance_raw, Mapping): raise ValueError("governance must be an object")
    allowed_governance = {"dataset_id", "exact_version", "source_locator", "license_identifier", "license_status", "license_evidence", "card", "metadata", "require_promotable"}
    if set(governance_raw) - allowed_governance: raise ValueError(f"governance contains unsupported fields: {sorted(set(governance_raw)-allowed_governance)}")
    metadata = governance_raw.get("metadata", {})
    if not isinstance(metadata, Mapping): raise ValueError("governance.metadata must be an object")
    governance = DynamicDatasetGovernance(dataset_id=governance_raw.get("dataset_id"), exact_version=governance_raw.get("exact_version"), source_locator=governance_raw.get("source_locator"), license_identifier=governance_raw.get("license_identifier"), license_status=LicenseStatus(governance_raw.get("license_status")), license_evidence=governance_raw.get("license_evidence"), card=_card(governance_raw.get("card")), metadata={str(k): str(v) for k, v in metadata.items()}, require_promotable=bool(governance_raw.get("require_promotable", False)))
    policy_raw = raw["split_policy"]
    if not isinstance(policy_raw, Mapping) or set(policy_raw) != {"seed", "weights"} or not isinstance(policy_raw["weights"], Mapping): raise ValueError("split_policy must contain seed and weights")
    policy = EpisodeSplitPolicy(seed=policy_raw["seed"], weights={str(k): v for k, v in policy_raw["weights"].items()})
    source_raw = raw["sources"]
    if not isinstance(source_raw, list) or not source_raw: raise ValueError("sources must be a non-empty array")
    sources = []
    for index, item in enumerate(source_raw):
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "lineage_receipt_sha256"}: raise ValueError(f"source {index} fields are invalid")
        sources.append(DynamicTrajectorySource(item["path"], item["sha256"], item["lineage_receipt_sha256"]))
    manifest, receipt = publish_dynamic_training_dataset(tuple(sources), governance=governance, split_policy=policy, output_dir=raw["output_dir"])
    verified = verify_dynamic_dataset_publication(Path(raw["output_dir"]) / "publication_receipt.json", sources=tuple(sources), require_promotable=governance.require_promotable)
    if verified.manifest.manifest_digest != manifest.manifest_digest or verified.receipt.receipt_sha256 != receipt.receipt_sha256: raise RuntimeError("dynamic publication verification returned a different identity")
    return {"dataset_id": manifest.dataset_id, "dataset_manifest_sha256": manifest.manifest_digest, "source_set_sha256": receipt.source_set_sha256, "split_policy_sha256": receipt.split_policy_sha256, "receipt_sha256": receipt.receipt_sha256, "splits": {item.name: {"path": item.path, "sha256": item.sha256, "record_count": item.record_count} for item in receipt.splits}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish governed dynamic-RAG train/validation datasets from exact trajectory JSONL")
    parser.add_argument("config", help="rigorousrag-dynamic-dataset-publication-config/v1 JSON file")
    print(json.dumps(run_dynamic_publication_config(parser.parse_args(argv).config), sort_keys=True, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())


__all__ = ["run_dynamic_publication_config"]
