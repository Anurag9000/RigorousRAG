"""Deterministic release provenance and software-supply-chain policy gates."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: str, label: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return text


@dataclass(frozen=True)
class ReleaseProvenance:
    commit_sha: str
    dependency_lock_sha256: str
    sbom_sha256: str
    artifact_sha256: str
    image_digest: str | None
    workflow: str
    run_id: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40,64}", self.commit_sha.lower()):
            raise ValueError("commit_sha is invalid.")
        _digest(self.dependency_lock_sha256, "dependency_lock_sha256")
        _digest(self.sbom_sha256, "sbom_sha256")
        _digest(self.artifact_sha256, "artifact_sha256")
        if self.image_digest is not None:
            value = self.image_digest.removeprefix("sha256:")
            _digest(value, "image_digest")
        if not self.workflow.strip() or not self.run_id.strip():
            raise ValueError("workflow and run_id are required.")


@dataclass(frozen=True)
class VulnerabilitySummary:
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0

    def __post_init__(self) -> None:
        for name in ("critical", "high", "medium", "low"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")


@dataclass(frozen=True)
class SupplyChainPolicy:
    require_signature: bool = True
    max_critical: int = 0
    max_high: int = 0
    require_sbom: bool = True
    require_hashed_lock: bool = True


@dataclass(frozen=True)
class SupplyChainDecision:
    eligible: bool
    reason_codes: tuple[str, ...]
    provenance_sha256: str


def provenance_sha256(provenance: ReleaseProvenance) -> str:
    payload = json.dumps(asdict(provenance), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_supply_chain(
    *,
    provenance: ReleaseProvenance,
    vulnerabilities: VulnerabilitySummary,
    signature_verified: bool,
    sbom_present: bool,
    hashed_lock_verified: bool,
    policy: SupplyChainPolicy | None = None,
) -> SupplyChainDecision:
    selected = policy or SupplyChainPolicy()
    reasons: list[str] = []
    if selected.require_signature and not signature_verified:
        reasons.append("signature_not_verified")
    if selected.require_sbom and not sbom_present:
        reasons.append("sbom_missing")
    if selected.require_hashed_lock and not hashed_lock_verified:
        reasons.append("hashed_lock_not_verified")
    if vulnerabilities.critical > selected.max_critical:
        reasons.append("critical_vulnerability_budget_exceeded")
    if vulnerabilities.high > selected.max_high:
        reasons.append("high_vulnerability_budget_exceeded")
    return SupplyChainDecision(
        eligible=not reasons,
        reason_codes=tuple(reasons),
        provenance_sha256=provenance_sha256(provenance),
    )


def vulnerability_summary_from_records(records: Iterable[Mapping[str, object]]) -> VulnerabilitySummary:
    """Normalize scanner findings into the release policy severity buckets."""

    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for record in records:
        severity = str(record.get("severity", "")).strip().lower()
        if severity not in counts:
            raise ValueError("vulnerability severity is unsupported.")
        counts[severity] += 1
    return VulnerabilitySummary(**counts)


def verify_ed25519_signature(*, public_key: bytes, payload: bytes, signature: bytes) -> bool:
    """Verify a detached Ed25519 signature for provenance or SBOM bytes."""

    try:
        key = Ed25519PublicKey.from_public_bytes(public_key)
        key.verify(signature, payload)
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


def build_minimal_sbom(*, components: Iterable[Mapping[str, str]]) -> Mapping[str, object]:
    """Build a deterministic dependency inventory suitable for hashing/attestation.

    This intentionally does not claim CycloneDX/SPDX conformance; release tooling may translate
    the canonical component inventory into either standard without changing its identity.
    """

    normalized = []
    for component in components:
        name = str(component.get("name", "")).strip()
        version = str(component.get("version", "")).strip()
        source = str(component.get("source", "")).strip()
        if not name or not version:
            raise ValueError("component name and version are required.")
        normalized.append({"name": name, "version": version, "source": source})
    normalized.sort(key=lambda item: (item["name"], item["version"], item["source"]))
    return {"schema": "rigorousrag-component-inventory/v1", "components": normalized}


__all__ = [
    "ReleaseProvenance",
    "SupplyChainDecision",
    "SupplyChainPolicy",
    "VulnerabilitySummary",
    "build_minimal_sbom",
    "evaluate_supply_chain",
    "provenance_sha256",
    "sha256_file",
    "verify_ed25519_signature",
    "vulnerability_summary_from_records",
]
