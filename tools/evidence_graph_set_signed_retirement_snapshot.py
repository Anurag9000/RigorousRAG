"""Deterministic text-free snapshot export for signed retirement journals."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.evidence_graph_set_signed_retirement_contracts import (
    SignedPublicationRetirementAttempt,
    _integer,
    _timestamp,
)
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_MAX_PATH = 4096
_MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str], *, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{label} must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError(f"{label} is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"{label} could not be validated.") from exc
        if _redirecting(info):
            raise ValueError(f"{label} may not contain redirects.")
    return absolute


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class SignedRetirementSnapshot:
    owner_id: str
    generated_at: float
    record_count: int
    records: tuple[SignedPublicationRetirementAttempt, ...]
    snapshot_digest: str
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        generated = _timestamp(self.generated_at, "generated_at")
        count = _integer(self.record_count, "record_count", 0, 10_000)
        if not isinstance(self.records, tuple) or any(
            not isinstance(value, SignedPublicationRetirementAttempt)
            for value in self.records
        ):
            raise ValueError("records must be retirement attempts.")
        ordered = tuple(sorted(self.records, key=lambda value: value.retirement_id))
        if ordered != self.records:
            raise ValueError("snapshot records must be ordered by retirement ID.")
        if len(ordered) != count:
            raise ValueError("record_count differs from snapshot records.")
        if len({value.retirement_id for value in ordered}) != len(ordered):
            raise ValueError("snapshot contains duplicate retirement IDs.")
        if any(value.owner_id != owner for value in ordered):
            raise ValueError("snapshot record escaped owner scope.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("snapshot schema is unsupported.")
        stable = {
            "scope": "rigorousrag-signed-retirement-snapshot-v1",
            "owner_id": owner,
            "generated_at": generated,
            "record_count": count,
            "records": [asdict(value) for value in ordered],
            "schema_version": self.schema_version,
        }
        digest = self.snapshot_digest.strip().lower() if isinstance(
            self.snapshot_digest, str
        ) else ""
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or digest != _sha256(stable)
        ):
            raise ValueError("snapshot_digest differs from snapshot content.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "record_count", count)
        object.__setattr__(self, "records", ordered)
        object.__setattr__(self, "snapshot_digest", digest)

    def public_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "owner_id": self.owner_id,
            "generated_at": self.generated_at,
            "record_count": self.record_count,
            "records": [asdict(value) for value in self.records],
            "snapshot_digest": self.snapshot_digest,
            "contains_source_text": False,
            "contains_assertion_secrets": False,
            "journal_mutation_performed": False,
        }


def build_signed_retirement_snapshot(
    *,
    owner_id: str,
    journal: Any,
    now: float | None = None,
    limit: int = 10_000,
) -> SignedRetirementSnapshot:
    owner = normalize_owner_id(owner_id)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    count = _integer(limit, "limit", 1, 10_000)
    if not callable(getattr(journal, "list", None)):
        raise ValueError("journal lacks the required read boundary.")
    records = tuple(journal.list(owner_id=owner, limit=count))
    if len(records) >= count:
        raise RuntimeError("snapshot reached the bounded result limit.")
    ordered = tuple(sorted(records, key=lambda value: value.retirement_id))
    stable = {
        "scope": "rigorousrag-signed-retirement-snapshot-v1",
        "owner_id": owner,
        "generated_at": timestamp,
        "record_count": len(ordered),
        "records": [asdict(value) for value in ordered],
        "schema_version": _SCHEMA_VERSION,
    }
    return SignedRetirementSnapshot(
        owner_id=owner,
        generated_at=timestamp,
        record_count=len(ordered),
        records=ordered,
        snapshot_digest=_sha256(stable),
    )


def _atomic_create(path: Path, payload: bytes) -> None:
    if len(payload) > _MAX_SNAPSHOT_BYTES:
        raise ValueError("snapshot exceeds the byte limit.")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    parent_info = parent.lstat()
    if _redirecting(parent_info) or not stat.S_ISDIR(parent_info.st_mode):
        raise ValueError("snapshot parent must be a non-redirecting directory.")
    if path.exists():
        raise FileExistsError(path)
    temporary = parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("snapshot write made no progress.")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(temporary, path)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def export_signed_retirement_snapshot(
    *,
    owner_id: str,
    journal: Any,
    output_path: str | os.PathLike[str],
    now: float | None = None,
    limit: int = 10_000,
) -> SignedRetirementSnapshot:
    snapshot = build_signed_retirement_snapshot(
        owner_id=owner_id,
        journal=journal,
        now=now,
        limit=limit,
    )
    path = _path(output_path, label="output_path")
    payload = _canonical_bytes(snapshot.public_payload()) + b"\n"
    _atomic_create(path, payload)
    return snapshot


def _attempt(raw: Any) -> SignedPublicationRetirementAttempt:
    if not isinstance(raw, dict):
        raise ValueError("snapshot retirement record must be an object.")
    return SignedPublicationRetirementAttempt(**raw)


def verify_signed_retirement_snapshot(
    path: str | os.PathLike[str],
) -> SignedRetirementSnapshot:
    selected = _path(path, label="snapshot_path")
    info = selected.lstat()
    if _redirecting(info) or not stat.S_ISREG(info.st_mode):
        raise ValueError("snapshot must be a regular non-redirecting file.")
    if info.st_size <= 0 or info.st_size > _MAX_SNAPSHOT_BYTES:
        raise ValueError("snapshot size is invalid.")
    payload = selected.read_bytes()
    if len(payload) != info.st_size:
        raise RuntimeError("snapshot changed while being read.")
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("snapshot JSON is invalid.") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "owner_id",
        "generated_at",
        "record_count",
        "records",
        "snapshot_digest",
        "contains_source_text",
        "contains_assertion_secrets",
        "journal_mutation_performed",
    }:
        raise ValueError("snapshot schema is invalid.")
    if (
        raw["contains_source_text"] is not False
        or raw["contains_assertion_secrets"] is not False
        or raw["journal_mutation_performed"] is not False
    ):
        raise ValueError("snapshot safety flags are invalid.")
    if not isinstance(raw["records"], list):
        raise ValueError("snapshot records must be a list.")
    records = tuple(_attempt(value) for value in raw["records"])
    return SignedRetirementSnapshot(
        owner_id=raw["owner_id"],
        generated_at=raw["generated_at"],
        record_count=raw["record_count"],
        records=records,
        snapshot_digest=raw["snapshot_digest"],
        schema_version=raw["schema_version"],
    )


__all__ = [
    "SignedRetirementSnapshot",
    "build_signed_retirement_snapshot",
    "export_signed_retirement_snapshot",
    "verify_signed_retirement_snapshot",
]
