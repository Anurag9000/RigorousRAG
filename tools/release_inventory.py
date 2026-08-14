"""Deterministic SPDX/CycloneDX inventories and release-attestation serialization."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from tools.release_supply_chain import ReleaseProvenance

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def normalize_components(records: Iterable[Mapping[str, object]]) -> tuple[dict[str, str], ...]:
    """Normalize installed-package records such as ``pip list --format=json`` output."""

    components: dict[str, dict[str, str]] = {}
    for record in records:
        name = str(record.get("name", "")).strip()
        version = str(record.get("version", "")).strip()
        if not _NAME.fullmatch(name) or not version or any(ord(ch) < 32 for ch in version):
            raise ValueError("component name/version is invalid.")
        key = re.sub(r"[-_.]+", "-", name).lower()
        normalized = {"name": name, "version": version}
        if key in components and components[key] != normalized:
            raise ValueError(f"conflicting component versions for {name!r}.")
        components[key] = normalized
    return tuple(components[key] for key in sorted(components))


def build_cyclonedx_sbom(records: Iterable[Mapping[str, object]]) -> Mapping[str, object]:
    components = normalize_components(records)
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "components": [
            {
                "type": "library",
                "name": item["name"],
                "version": item["version"],
                "bom-ref": f"pkg:pypi/{item['name']}@{item['version']}",
                "purl": f"pkg:pypi/{item['name']}@{item['version']}",
            }
            for item in components
        ],
    }


def reproducible_timestamp(source_date_epoch: int | str) -> str:
    try:
        epoch = int(source_date_epoch)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("SOURCE_DATE_EPOCH must be an integer timestamp.") from exc
    if epoch < 0:
        raise ValueError("SOURCE_DATE_EPOCH must be non-negative.")
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_spdx_sbom(
    records: Iterable[Mapping[str, object]], *, namespace: str, created: str
) -> Mapping[str, object]:
    selected_namespace = str(namespace).strip()
    if not selected_namespace.startswith(("https://", "urn:")):
        raise ValueError("SPDX document namespace must be an https URL or URN.")
    selected_created = str(created).strip()
    if not _TIMESTAMP.fullmatch(selected_created):
        raise ValueError("SPDX creation timestamp must use YYYY-MM-DDTHH:MM:SSZ.")
    components = normalize_components(records)
    packages = []
    relationships = []
    for index, item in enumerate(components, 1):
        spdx_id = f"SPDXRef-Package-{index}"
        packages.append(
            {
                "SPDXID": spdx_id,
                "name": item["name"],
                "versionInfo": item["version"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:pypi/{item['name']}@{item['version']}",
                    }
                ],
            }
        )
        relationships.append(
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": spdx_id,
            }
        )
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "RigorousRAG release inventory",
        "documentNamespace": selected_namespace,
        "creationInfo": {
            "created": selected_created,
            "creators": ["Tool: RigorousRAG-release-inventory"],
        },
        "packages": packages,
        "relationships": relationships,
    }


def write_canonical_json(path: str | Path, value: object) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(value)
    destination.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_provenance(path: str | Path, provenance: ReleaseProvenance) -> str:
    return write_canonical_json(path, asdict(provenance))


def load_pip_list(path: str | Path) -> tuple[dict[str, str], ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError("pip-list input must be a JSON array of package records.")
    return normalize_components(raw)


__all__ = [
    "build_cyclonedx_sbom",
    "build_spdx_sbom",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_pip_list",
    "normalize_components",
    "reproducible_timestamp",
    "write_canonical_json",
    "write_provenance",
]
