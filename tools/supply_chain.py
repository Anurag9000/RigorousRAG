"""Deterministic artifact integrity, SBOM, provenance and vulnerability policy controls."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Protocol


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True, order=True)
class ArtifactDigest:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ArtifactManifest:
    artifacts: tuple[ArtifactDigest, ...]
    manifest_sha256: str


def build_manifest(files: Mapping[str, bytes]) -> ArtifactManifest:
    artifacts = tuple(
        sorted(
            (
                ArtifactDigest(str(path), sha256_bytes(bytes(content)), len(content))
                for path, content in files.items()
            ),
            key=lambda item: item.path,
        )
    )
    digest = sha256_bytes(_canonical_json([asdict(item) for item in artifacts]))
    return ArtifactManifest(artifacts, digest)


def verify_manifest(manifest: ArtifactManifest, files: Mapping[str, bytes]) -> bool:
    return build_manifest(files) == manifest


@dataclass(frozen=True, order=True)
class Component:
    name: str
    version: str
    purl: str = ""
    license: str = ""


@dataclass(frozen=True)
class SoftwareBillOfMaterials:
    components: tuple[Component, ...]
    document_sha256: str


def build_sbom(components: Iterable[Component]) -> SoftwareBillOfMaterials:
    normalized = tuple(sorted(set(components)))
    digest = sha256_bytes(_canonical_json([asdict(item) for item in normalized]))
    return SoftwareBillOfMaterials(normalized, digest)


@dataclass(frozen=True)
class BuildProvenance:
    repository: str
    revision: str
    builder_id: str
    build_config_sha256: str
    input_manifest_sha256: str
    output_sha256: str

    def canonical_bytes(self) -> bytes:
        return _canonical_json(asdict(self))


def verify_provenance_output(provenance: BuildProvenance, output: bytes) -> bool:
    return hmac.compare_digest(provenance.output_sha256, sha256_bytes(output))


class Severity(IntEnum):
    UNKNOWN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def parse(cls, value: object) -> "Severity":
        name = str(value or "UNKNOWN").strip().upper()
        return cls.__members__.get(name, cls.UNKNOWN)


@dataclass(frozen=True, order=True)
class Vulnerability:
    package: str
    vulnerability_id: str
    severity: Severity
    installed_version: str = ""
    fixed_version: str = ""


@dataclass(frozen=True)
class VulnerabilityPolicy:
    max_allowed_severity: Severity = Severity.MEDIUM
    fail_on_unknown: bool = True


@dataclass(frozen=True)
class VulnerabilityDecision:
    allowed: bool
    blocking: tuple[Vulnerability, ...]


def evaluate_vulnerabilities(
    vulnerabilities: Iterable[Vulnerability],
    policy: VulnerabilityPolicy | None = None,
) -> VulnerabilityDecision:
    selected = policy or VulnerabilityPolicy()
    blocking = tuple(
        sorted(
            (
                item
                for item in vulnerabilities
                if item.severity > selected.max_allowed_severity
                or (selected.fail_on_unknown and item.severity == Severity.UNKNOWN)
            ),
            key=lambda item: (int(item.severity), item.package, item.vulnerability_id),
            reverse=True,
        )
    )
    return VulnerabilityDecision(not blocking, blocking)


def parse_pip_audit(payload: Mapping[str, object]) -> tuple[Vulnerability, ...]:
    records: list[Vulnerability] = []
    dependencies = payload.get("dependencies", ())
    if not isinstance(dependencies, list):
        return ()
    for dependency in dependencies:
        if not isinstance(dependency, Mapping):
            continue
        package = str(dependency.get("name", ""))
        installed = str(dependency.get("version", ""))
        vulns = dependency.get("vulns", ())
        if not isinstance(vulns, list):
            continue
        for vuln in vulns:
            if not isinstance(vuln, Mapping):
                continue
            severity_value: object = vuln.get("severity", "UNKNOWN")
            fixes = vuln.get("fix_versions", ())
            fixed = str(fixes[0]) if isinstance(fixes, list) and fixes else ""
            records.append(
                Vulnerability(
                    package=package,
                    vulnerability_id=str(vuln.get("id", "UNKNOWN")),
                    severity=Severity.parse(severity_value),
                    installed_version=installed,
                    fixed_version=fixed,
                )
            )
    return tuple(records)


def parse_trivy(payload: Mapping[str, object]) -> tuple[Vulnerability, ...]:
    records: list[Vulnerability] = []
    results = payload.get("Results", ())
    if not isinstance(results, list):
        return ()
    for result in results:
        if not isinstance(result, Mapping):
            continue
        vulnerabilities = result.get("Vulnerabilities", ())
        if not isinstance(vulnerabilities, list):
            continue
        for vuln in vulnerabilities:
            if not isinstance(vuln, Mapping):
                continue
            records.append(
                Vulnerability(
                    package=str(vuln.get("PkgName", "")),
                    vulnerability_id=str(vuln.get("VulnerabilityID", "UNKNOWN")),
                    severity=Severity.parse(vuln.get("Severity")),
                    installed_version=str(vuln.get("InstalledVersion", "")),
                    fixed_version=str(vuln.get("FixedVersion", "")),
                )
            )
    return tuple(records)


class Signer(Protocol):
    def sign(self, payload: bytes) -> str: ...


class Verifier(Protocol):
    def verify(self, payload: bytes, signature: str) -> bool: ...


class HMACSigner:
    """Local integrity reference signer; not a substitute for Sigstore/KMS/HSM signing."""

    def __init__(self, key: bytes) -> None:
        if not key:
            raise ValueError("key must not be empty")
        self._key = bytes(key)

    def sign(self, payload: bytes) -> str:
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def verify(self, payload: bytes, signature: str) -> bool:
        return hmac.compare_digest(self.sign(payload), str(signature))


__all__ = [
    "ArtifactDigest",
    "ArtifactManifest",
    "BuildProvenance",
    "Component",
    "HMACSigner",
    "Severity",
    "Signer",
    "SoftwareBillOfMaterials",
    "Verifier",
    "Vulnerability",
    "VulnerabilityDecision",
    "VulnerabilityPolicy",
    "build_manifest",
    "build_sbom",
    "evaluate_vulnerabilities",
    "parse_pip_audit",
    "parse_trivy",
    "sha256_bytes",
    "verify_manifest",
    "verify_provenance_output",
]
