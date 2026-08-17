"""Portable, content-addressed research bundles with private-reference separation.

Public/publishable artifacts may be embedded. Private evidence is represented only by
content digests and authorization-scoped reference identifiers; this module refuses to
embed entries classified as private. Bundle manifests can be attested using the existing
provider-neutral manifest-signing contracts.
"""
from __future__ import annotations

import hashlib
import io
import json
import posixpath
import zipfile
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.manifest_attestation import ManifestAttestation, ManifestSigner, attest_manifest, canonical_manifest_bytes

_MAX_ENTRIES = 100_000
_MAX_EMBEDDED_BYTES = 512 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_RESERVED_PATHS = frozenset({"manifest.json", "attestation.json"})


def _text(value: Any, label: str, maximum: int = 2000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if (not selected and not allow_empty) or len(selected) > maximum or "\x00" in selected:
        raise ValueError(f"{label} is invalid")
    return selected


def _sha(value: str, label: str, *, allow_empty: bool = False) -> str:
    selected = _text(value, label, 64, allow_empty=allow_empty).lower()
    if selected and (len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected)):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _safe_archive_path(value: str) -> str:
    selected = _text(value, "archive_path", 1000).replace("\\", "/")
    normalized = posixpath.normpath(selected)
    if normalized.startswith("/") or normalized == ".." or normalized.startswith("../") or ":" in normalized.split("/")[0]:
        raise ValueError("archive_path must be a relative safe path")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError("archive_path contains an invalid segment")
    return normalized


@dataclass(frozen=True)
class PortableBundleEntry:
    entry_id: str
    kind: str
    media_type: str
    content_sha256: str
    size_bytes: int
    access_class: str
    disposition: str
    archive_path: str = ""
    reference_id: str = ""
    authorization_policy_fingerprint: str = ""
    source_identity_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_id", _text(self.entry_id, "entry_id", 500))
        object.__setattr__(self, "kind", _text(self.kind, "kind", 128).lower())
        object.__setattr__(self, "media_type", _text(self.media_type, "media_type", 256).lower())
        object.__setattr__(self, "content_sha256", _sha(self.content_sha256, "content_sha256"))
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or not 0 <= self.size_bytes <= 2**63 - 1:
            raise ValueError("size_bytes is invalid")
        access = _text(self.access_class, "access_class", 64).lower()
        if access not in {"public", "shared", "private"}:
            raise ValueError("access_class must be public, shared, or private")
        object.__setattr__(self, "access_class", access)
        disposition = _text(self.disposition, "disposition", 64).lower()
        if disposition not in {"embedded", "reference"}:
            raise ValueError("disposition must be embedded or reference")
        if access == "private" and disposition == "embedded":
            raise ValueError("private bundle entries may not be embedded")
        object.__setattr__(self, "disposition", disposition)
        archive_path = self.archive_path.strip()
        reference_id = self.reference_id.strip()
        if disposition == "embedded":
            if not archive_path or reference_id:
                raise ValueError("embedded entry requires archive_path and no reference_id")
            safe_path = _safe_archive_path(archive_path)
            if safe_path in _RESERVED_PATHS:
                raise ValueError("embedded entry may not use a reserved archive path")
            object.__setattr__(self, "archive_path", safe_path)
            object.__setattr__(self, "reference_id", "")
        else:
            if not reference_id or archive_path:
                raise ValueError("reference entry requires reference_id and no archive_path")
            object.__setattr__(self, "reference_id", _text(reference_id, "reference_id", 2000))
            object.__setattr__(self, "archive_path", "")
        object.__setattr__(
            self,
            "authorization_policy_fingerprint",
            _sha(self.authorization_policy_fingerprint, "authorization_policy_fingerprint", allow_empty=True),
        )
        object.__setattr__(self, "source_identity_sha256", _sha(self.source_identity_sha256, "source_identity_sha256", allow_empty=True))


@dataclass(frozen=True)
class PortableResearchBundleManifest:
    bundle_id: str
    project_id: str
    capsule_fingerprint: str
    disclosure_policy_fingerprint: str
    code_revision: str
    entries: tuple[PortableBundleEntry, ...]
    report_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    schema: str = "rigorousrag.portable-research-bundle/v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_id", _text(self.bundle_id, "bundle_id", 500))
        object.__setattr__(self, "project_id", _text(self.project_id, "project_id", 500))
        object.__setattr__(self, "capsule_fingerprint", _sha(self.capsule_fingerprint, "capsule_fingerprint"))
        object.__setattr__(self, "disclosure_policy_fingerprint", _sha(self.disclosure_policy_fingerprint, "disclosure_policy_fingerprint"))
        object.__setattr__(self, "code_revision", _text(self.code_revision, "code_revision", 256))
        if len(self.entries) > _MAX_ENTRIES or any(not isinstance(item, PortableBundleEntry) for item in self.entries):
            raise ValueError("entries are invalid")
        entry_ids = [item.entry_id for item in self.entries]
        if len(set(entry_ids)) != len(entry_ids):
            raise ValueError("entry_id values must be unique")
        embedded_paths = [item.archive_path for item in self.entries if item.disposition == "embedded"]
        if len(set(embedded_paths)) != len(embedded_paths):
            raise ValueError("archive_path values must be unique")
        object.__setattr__(self, "entries", tuple(sorted(self.entries, key=lambda item: item.entry_id)))
        object.__setattr__(self, "report_ids", tuple(sorted(set(_text(item, "report_id", 500) for item in self.report_ids))))
        object.__setattr__(self, "notes", tuple(dict.fromkeys(_text(item, "note", 4000) for item in self.notes)))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_manifest_bytes(asdict(self))).hexdigest()


@dataclass(frozen=True)
class PortableResearchBundle:
    manifest: PortableResearchBundleManifest
    archive_sha256: str
    archive_size_bytes: int
    attestation: ManifestAttestation | None = None


@dataclass(frozen=True)
class PortableBundleVerification:
    manifest_matches: bool
    embedded_entries: Mapping[str, bool]
    unexpected_paths: tuple[str, ...]
    unsafe_paths: tuple[str, ...]
    duplicate_paths: tuple[str, ...]
    archive_sha256: str
    archive_size_bytes: int

    @property
    def valid(self) -> bool:
        return (
            self.manifest_matches
            and all(self.embedded_entries.values())
            and not self.unexpected_paths
            and not self.unsafe_paths
            and not self.duplicate_paths
        )


class PortableBundleBuilder:
    def __init__(
        self,
        *,
        bundle_id: str,
        project_id: str,
        capsule_fingerprint: str,
        disclosure_policy_fingerprint: str,
        code_revision: str,
        report_ids: Sequence[str] = (),
    ) -> None:
        self._bundle_id = _text(bundle_id, "bundle_id", 500)
        self._project_id = _text(project_id, "project_id", 500)
        self._capsule_fingerprint = _sha(capsule_fingerprint, "capsule_fingerprint")
        self._disclosure_policy_fingerprint = _sha(disclosure_policy_fingerprint, "disclosure_policy_fingerprint")
        self._code_revision = _text(code_revision, "code_revision", 256)
        self._report_ids = tuple(report_ids)
        self._entries: dict[str, PortableBundleEntry] = {}
        self._embedded: dict[str, bytes] = {}
        self._embedded_bytes = 0

    def add_embedded(
        self,
        *,
        entry_id: str,
        kind: str,
        media_type: str,
        data: bytes,
        archive_path: str,
        access_class: str = "public",
        source_identity_sha256: str = "",
    ) -> PortableBundleEntry:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if len(self._entries) >= _MAX_ENTRIES:
            raise ValueError("bundle entry limit exceeded")
        if self._embedded_bytes + len(data) > _MAX_EMBEDDED_BYTES:
            raise ValueError("embedded bundle byte limit exceeded")
        selected_id = _text(entry_id, "entry_id", 500)
        if selected_id in self._entries:
            raise ValueError("duplicate entry_id")
        selected_path = _safe_archive_path(archive_path)
        if selected_path in self._embedded or selected_path in _RESERVED_PATHS:
            raise ValueError("duplicate or reserved archive_path")
        entry = PortableBundleEntry(
            entry_id=selected_id,
            kind=kind,
            media_type=media_type,
            content_sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            access_class=access_class,
            disposition="embedded",
            archive_path=selected_path,
            source_identity_sha256=source_identity_sha256,
        )
        self._entries[selected_id] = entry
        self._embedded[selected_path] = bytes(data)
        self._embedded_bytes += len(data)
        return entry

    def add_reference(
        self,
        *,
        entry_id: str,
        kind: str,
        media_type: str,
        content_sha256: str,
        size_bytes: int,
        reference_id: str,
        access_class: str = "private",
        authorization_policy_fingerprint: str,
        source_identity_sha256: str = "",
    ) -> PortableBundleEntry:
        if len(self._entries) >= _MAX_ENTRIES:
            raise ValueError("bundle entry limit exceeded")
        selected_id = _text(entry_id, "entry_id", 500)
        if selected_id in self._entries:
            raise ValueError("duplicate entry_id")
        entry = PortableBundleEntry(
            entry_id=selected_id,
            kind=kind,
            media_type=media_type,
            content_sha256=content_sha256,
            size_bytes=size_bytes,
            access_class=access_class,
            disposition="reference",
            reference_id=reference_id,
            authorization_policy_fingerprint=authorization_policy_fingerprint,
            source_identity_sha256=source_identity_sha256,
        )
        self._entries[selected_id] = entry
        return entry

    def manifest(self, *, notes: Sequence[str] = ()) -> PortableResearchBundleManifest:
        return PortableResearchBundleManifest(
            bundle_id=self._bundle_id,
            project_id=self._project_id,
            capsule_fingerprint=self._capsule_fingerprint,
            disclosure_policy_fingerprint=self._disclosure_policy_fingerprint,
            code_revision=self._code_revision,
            entries=tuple(self._entries.values()),
            report_ids=self._report_ids,
            notes=tuple(notes),
        )

    def build_zip(
        self,
        *,
        notes: Sequence[str] = (),
        signer: ManifestSigner | None = None,
    ) -> tuple[PortableResearchBundle, bytes]:
        manifest = self.manifest(notes=notes)
        manifest_payload = asdict(manifest)
        manifest_bytes = canonical_manifest_bytes(manifest_payload)
        attestation = None if signer is None else attest_manifest(f"portable-bundle:{manifest.bundle_id}", manifest_payload, signer)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(self._embedded):
                info = zipfile.ZipInfo(path, _FIXED_ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, self._embedded[path])
            manifest_info = zipfile.ZipInfo("manifest.json", _FIXED_ZIP_TIME)
            manifest_info.compress_type = zipfile.ZIP_DEFLATED
            manifest_info.external_attr = 0o600 << 16
            archive.writestr(manifest_info, manifest_bytes)
            if attestation is not None:
                attestation_info = zipfile.ZipInfo("attestation.json", _FIXED_ZIP_TIME)
                attestation_info.compress_type = zipfile.ZIP_DEFLATED
                attestation_info.external_attr = 0o600 << 16
                archive.writestr(attestation_info, _canonical(asdict(attestation)))
        data = buffer.getvalue()
        if len(data) > _MAX_ARCHIVE_BYTES:
            raise ValueError("portable bundle archive exceeds the byte limit")
        result = PortableResearchBundle(
            manifest=manifest,
            archive_sha256=hashlib.sha256(data).hexdigest(),
            archive_size_bytes=len(data),
            attestation=attestation,
        )
        return result, data


def verify_bundle_archive(manifest: PortableResearchBundleManifest, archive_bytes: bytes) -> PortableBundleVerification:
    if not isinstance(manifest, PortableResearchBundleManifest) or not isinstance(archive_bytes, bytes):
        raise TypeError("manifest/archive types are invalid")
    if len(archive_bytes) > _MAX_ARCHIVE_BYTES:
        raise ValueError("portable bundle archive exceeds the byte limit")
    embedded = {item.archive_path: item for item in manifest.entries if item.disposition == "embedded"}
    output: dict[str, bool] = {item.entry_id: False for item in embedded.values()}
    expected_paths = set(embedded) | {"manifest.json", "attestation.json"}
    unexpected: list[str] = []
    unsafe: list[str] = []
    duplicates: list[str] = []
    manifest_matches = False
    with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
        infos = archive.infolist()
        if len(infos) > _MAX_ENTRIES + len(_RESERVED_PATHS):
            raise ValueError("portable bundle contains too many archive entries")
        seen: set[str] = set()
        for info in infos:
            raw_name = info.filename
            try:
                safe_name = _safe_archive_path(raw_name)
            except ValueError:
                unsafe.append(raw_name)
                continue
            if safe_name in seen:
                duplicates.append(safe_name)
                continue
            seen.add(safe_name)
            if safe_name not in expected_paths:
                unexpected.append(safe_name)
            if info.file_size > _MAX_EMBEDDED_BYTES:
                continue
            if safe_name == "manifest.json":
                payload = archive.read(info)
                manifest_matches = payload == canonical_manifest_bytes(asdict(manifest))
                continue
            entry = embedded.get(safe_name)
            if entry is None:
                continue
            if info.file_size != entry.size_bytes:
                continue
            data = archive.read(info)
            output[entry.entry_id] = hashlib.sha256(data).hexdigest() == entry.content_sha256
    return PortableBundleVerification(
        manifest_matches=manifest_matches,
        embedded_entries=output,
        unexpected_paths=tuple(sorted(set(unexpected))),
        unsafe_paths=tuple(sorted(set(unsafe))),
        duplicate_paths=tuple(sorted(set(duplicates))),
        archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        archive_size_bytes=len(archive_bytes),
    )


def verify_embedded_entries(manifest: PortableResearchBundleManifest, archive_bytes: bytes) -> Mapping[str, bool]:
    """Compatibility projection for callers that only need per-entry content checks."""
    return verify_bundle_archive(manifest, archive_bytes).embedded_entries


__all__ = [
    "PortableBundleBuilder",
    "PortableBundleEntry",
    "PortableBundleVerification",
    "PortableResearchBundle",
    "PortableResearchBundleManifest",
    "verify_bundle_archive",
    "verify_embedded_entries",
]
