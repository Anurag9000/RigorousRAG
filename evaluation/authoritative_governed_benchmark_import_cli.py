"""Config-only CLI for authoritative v2 governed benchmark publication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.authoritative_governed_benchmark_import import (
    import_authoritative_governed_benchmark,
)
from evaluation.authoritative_governed_benchmark_io import (
    verify_authoritative_governed_benchmark_import,
)
from evaluation.dataset_governance import DatasetModality, DatasetTask, LicenseStatus
from evaluation.governed_benchmark_import import (
    BenchmarkGovernanceSpec,
    BenchmarkSplitImportSpec,
    _card_from_json,
    _profile_from_json,
)
from training.advanced_path_authority import safe_advanced_path

_MAX_CONFIG_BYTES = 16 * 1024 * 1024


def _read(path: str | Path) -> Mapping[str, Any]:
    source = safe_advanced_path(
        path,
        label="benchmark import config",
        must_exist=True,
        require_file=True,
    )
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("benchmark import config exceeds byte safety bound")
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError("benchmark import config is not strict JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("benchmark import config must contain an object")
    return raw


def _strict_profile(raw: Any, label: str):
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    if "generate_missing_ids" in raw and not isinstance(raw["generate_missing_ids"], bool):
        raise ValueError(f"{label}.generate_missing_ids must be boolean")
    return _profile_from_json(raw, label)


def run_import_config(path: str | Path) -> Mapping[str, Any]:
    raw = _read(path)
    if (
        set(raw) != {"schema", "output_dir", "governance", "splits"}
        or raw.get("schema") != "rigorousrag-governed-benchmark-import-config/v1"
    ):
        raise ValueError(
            "benchmark import config must be rigorousrag-governed-benchmark-import-config/v1"
        )
    governance_raw = raw.get("governance")
    if not isinstance(governance_raw, Mapping):
        raise ValueError("governance must be an object")
    allowed_governance = {
        "dataset_id",
        "exact_version",
        "source_locator",
        "license_identifier",
        "license_status",
        "license_evidence",
        "tasks",
        "modalities",
        "card",
        "metadata",
        "require_promotable",
    }
    unknown = set(governance_raw) - allowed_governance
    if unknown:
        raise ValueError(f"governance contains unsupported fields: {sorted(unknown)}")
    require_promotable = governance_raw.get("require_promotable", False)
    if not isinstance(require_promotable, bool):
        raise ValueError("governance.require_promotable must be boolean")
    tasks = governance_raw.get("tasks")
    modalities = governance_raw.get("modalities")
    if (
        not isinstance(tasks, list)
        or not tasks
        or not isinstance(modalities, list)
        or not modalities
    ):
        raise ValueError("governance.tasks and governance.modalities must be non-empty arrays")
    metadata = governance_raw.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("governance.metadata must be an object")
    governance = BenchmarkGovernanceSpec(
        dataset_id=governance_raw.get("dataset_id"),
        exact_version=governance_raw.get("exact_version"),
        source_locator=governance_raw.get("source_locator"),
        license_identifier=governance_raw.get("license_identifier"),
        license_status=LicenseStatus(governance_raw.get("license_status")),
        license_evidence=governance_raw.get("license_evidence"),
        tasks=tuple(DatasetTask(item) for item in tasks),
        modalities=tuple(DatasetModality(item) for item in modalities),
        card=_card_from_json(governance_raw.get("card")),
        metadata={str(key): str(value) for key, value in metadata.items()},
        require_promotable=require_promotable,
    )

    split_raw = raw.get("splits")
    if not isinstance(split_raw, list) or not split_raw:
        raise ValueError("splits must be a non-empty array")
    splits: list[BenchmarkSplitImportSpec] = []
    allowed_split = {
        "name",
        "source_path",
        "source_sha256",
        "input_format",
        "adapter_name",
        "profile",
        "expected_record_count",
    }
    for index, item in enumerate(split_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"split {index} must be an object")
        unknown_split = set(item) - allowed_split
        if unknown_split:
            raise ValueError(
                f"split {index} contains unsupported fields: {sorted(unknown_split)}"
            )
        profile = (
            _strict_profile(item["profile"], f"split[{index}].profile")
            if item.get("profile") is not None
            else None
        )
        splits.append(
            BenchmarkSplitImportSpec(
                name=item.get("name"),
                source_path=item.get("source_path"),
                source_sha256=item.get("source_sha256"),
                input_format=item.get("input_format", "jsonl"),
                adapter_name=item.get("adapter_name"),
                profile=profile,
                expected_record_count=item.get("expected_record_count"),
            )
        )

    manifest, receipt = import_authoritative_governed_benchmark(
        governance,
        tuple(splits),
        output_dir=raw.get("output_dir"),
    )
    verified = verify_authoritative_governed_benchmark_import(
        Path(raw["output_dir"]) / "import_receipt.json",
        require_promotable=governance.require_promotable,
    )
    if (
        verified.manifest.manifest_digest != manifest.manifest_digest
        or verified.receipt.receipt_sha256 != receipt.receipt_sha256
    ):
        raise RuntimeError("authoritative benchmark read-side identity differs after publication")
    return {
        "dataset_id": manifest.dataset_id,
        "dataset_manifest_sha256": manifest.manifest_digest,
        "artifact_sha256": manifest.artifact_sha256,
        "record_count": sum(split.record_count for split in manifest.splits),
        "receipt_sha256": receipt.receipt_sha256,
        "manifest_path": receipt.manifest_path,
        "publication_authority": "authoritative_governed_benchmark_import/v2",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import exact local benchmark bytes through the authoritative v2 publisher"
    )
    parser.add_argument(
        "config",
        help="rigorousrag-governed-benchmark-import-config/v1 JSON file",
    )
    result = run_import_config(parser.parse_args(argv).config)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "run_import_config"]
