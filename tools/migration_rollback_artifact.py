"""Validated encrypted rollback payload and manifest contracts."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from tools.migration_cutover_preflight import (
    CutoverPreflight,
    _field_identity,
    _sha256,
    _sparse_identity,
    _vector_identity,
)
from tools.migration_types import digest, exact_integer, identifier, timestamp
from tools.security import normalize_owner_id

_ALGORITHM = "AES-256-GCM"
_SCHEMA_VERSION = 1
_MAX_PAYLOAD_BYTES = 1_000_000_000


def _strict_object_pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def strict_json_loads(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise RuntimeError(f"{label} is invalid JSON.") from exc


def canonical_json_bytes(value: Any, *, maximum: int = _MAX_PAYLOAD_BYTES) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if not payload or len(payload) > maximum:
        raise ValueError("rollback JSON exceeds the byte limit.")
    return payload


def _generation_payload(generation: Any) -> dict[str, Any]:
    metadata = getattr(generation, "metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("generation metadata must be a mapping.")
    return {
        "owner_id": normalize_owner_id(getattr(generation, "owner_id", None)),
        "doc_id": identifier(getattr(generation, "doc_id", None), "doc_id", 200),
        "sequence": exact_integer(
            getattr(generation, "sequence", None), "sequence", 1, 2**63 - 1
        ),
        "state": identifier(getattr(generation, "state", None), "state", 20),
        "content_sha256": digest(
            getattr(generation, "content_sha256", None), "content_sha256"
        ),
        "profile_fingerprint": digest(
            getattr(generation, "profile_fingerprint", None),
            "profile_fingerprint",
        ),
        "vector_rows": exact_integer(
            getattr(generation, "vector_rows", None),
            "vector_rows",
            1,
            100_000_000,
        ),
        "sparse_generation": exact_integer(
            getattr(generation, "sparse_generation", None),
            "sparse_generation",
            1,
            2**63 - 1,
        ),
        "committed_at": timestamp(
            getattr(generation, "committed_at", 0.0), "committed_at"
        ),
        "metadata": dict(metadata),
    }


def _vector_rows_payload(vector: Any, owner_id: str, doc_id: str) -> list[dict[str, Any]]:
    ids = tuple(getattr(vector, "ids"))
    documents = tuple(getattr(vector, "documents"))
    metadatas = tuple(getattr(vector, "metadatas"))
    if not len(ids) == len(documents) == len(metadatas):
        raise RuntimeError("vector rollback snapshot arrays are inconsistent.")
    rows: list[dict[str, Any]] = []
    for raw_id, document, metadata in zip(ids, documents, metadatas, strict=True):
        row_id = identifier(raw_id, "vector row id", 500)
        if not isinstance(document, str) or "\x00" in document:
            raise ValueError("vector rollback text is invalid.")
        if not isinstance(metadata, Mapping):
            raise ValueError("vector rollback metadata must be a mapping.")
        if metadata.get("owner_id") != owner_id or metadata.get("doc_id") != doc_id:
            raise RuntimeError("vector rollback metadata escaped task scope.")
        rows.append({"id": row_id, "document": document, "metadata": dict(metadata)})
    return rows


def _sparse_payload(sparse: Any) -> dict[str, Any]:
    metadata = getattr(sparse, "metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("sparse rollback metadata must be a mapping.")
    return {
        "owner_id": normalize_owner_id(getattr(sparse, "owner_id", None)),
        "doc_id": identifier(getattr(sparse, "doc_id", None), "doc_id", 200),
        "generation": exact_integer(
            getattr(sparse, "generation", None), "generation", 1, 2**63 - 1
        ),
        "profile_fingerprint": digest(
            getattr(sparse, "profile_fingerprint", None),
            "profile_fingerprint",
        ),
        "metadata": dict(metadata),
        "fields": [_field_identity(field) for field in tuple(getattr(sparse, "fields"))],
        "schema_version": exact_integer(
            getattr(sparse, "schema_version", 1), "schema_version", 1, 1_000_000
        ),
    }


def _rollback_identity(
    preflight: CutoverPreflight,
    vector_digest: str,
    sparse_digest: str,
    vector_rows: int,
    sparse_fields: int,
) -> str:
    return _sha256(
        {
            "owner_id": preflight.owner_id,
            "doc_id": preflight.doc_id,
            "source_sequence": preflight.source_sequence,
            "source_profile_fingerprint": preflight.source_profile_fingerprint,
            "source_content_sha256": preflight.source_content_sha256,
            "vector_snapshot_digest": vector_digest,
            "sparse_snapshot_digest": sparse_digest,
            "vector_rows": vector_rows,
            "sparse_generation": preflight.source_sparse_generation,
            "sparse_fields": sparse_fields,
        }
    )


def capture_rollback_payload(
    preflight: CutoverPreflight,
    authoritative_snapshot: Any,
) -> dict[str, Any]:
    if not isinstance(preflight, CutoverPreflight):
        raise ValueError("preflight must be CutoverPreflight.")
    if (
        getattr(authoritative_snapshot, "owner_id", None) != preflight.owner_id
        or getattr(authoritative_snapshot, "doc_id", None) != preflight.doc_id
    ):
        raise RuntimeError("authoritative rollback snapshot escaped preflight scope.")
    generation = getattr(authoritative_snapshot, "generation", None)
    generation_payload = _generation_payload(generation)
    if (
        generation_payload["sequence"] != preflight.source_sequence
        or generation_payload["state"] not in {"active", "restored"}
        or generation_payload["content_sha256"] != preflight.source_content_sha256
        or generation_payload["profile_fingerprint"]
        != preflight.source_profile_fingerprint
        or generation_payload["vector_rows"] != preflight.source_vector_rows
        or generation_payload["sparse_generation"]
        != preflight.source_sparse_generation
    ):
        raise RuntimeError("authoritative generation changed after preflight.")
    stores = getattr(authoritative_snapshot, "stores", None)
    vector = getattr(stores, "vector", None)
    sparse = getattr(stores, "sparse", None)
    vector_digest, vector_count = _vector_identity(
        vector, preflight.owner_id, preflight.doc_id
    )
    sparse_digest, sparse_count = _sparse_identity(
        sparse,
        preflight.owner_id,
        preflight.doc_id,
        preflight.source_profile_fingerprint,
        preflight.source_sparse_generation,
    )
    if (
        vector_digest != preflight.vector_snapshot_digest
        or sparse_digest != preflight.sparse_snapshot_digest
        or vector_count != preflight.source_vector_rows
        or sparse_count != preflight.source_sparse_fields
    ):
        raise RuntimeError("authoritative rollback snapshot changed after preflight.")
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "task_id": preflight.task_id,
        "preflight_digest": preflight.preflight_digest,
        "rollback_identity_digest": preflight.rollback_identity_digest,
        "owner_id": preflight.owner_id,
        "doc_id": preflight.doc_id,
        "generation": generation_payload,
        "vector_rows": _vector_rows_payload(
            vector, preflight.owner_id, preflight.doc_id
        ),
        "sparse_snapshot": _sparse_payload(sparse),
    }
    return validate_rollback_payload(preflight, payload)


def validate_rollback_payload(
    preflight: CutoverPreflight,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(preflight, CutoverPreflight):
        raise ValueError("preflight must be CutoverPreflight.")
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "task_id",
        "preflight_digest",
        "rollback_identity_digest",
        "owner_id",
        "doc_id",
        "generation",
        "vector_rows",
        "sparse_snapshot",
    }:
        raise RuntimeError("rollback payload schema is invalid.")
    if exact_integer(payload["schema_version"], "schema_version", 1, 1) != 1:
        raise RuntimeError("rollback payload schema is unsupported.")
    expected = {
        "task_id": preflight.task_id,
        "preflight_digest": preflight.preflight_digest,
        "rollback_identity_digest": preflight.rollback_identity_digest,
        "owner_id": preflight.owner_id,
        "doc_id": preflight.doc_id,
    }
    if any(payload[name] != value for name, value in expected.items()):
        raise RuntimeError("rollback payload identity does not match preflight.")

    generation = payload["generation"]
    if not isinstance(generation, Mapping) or set(generation) != {
        "owner_id",
        "doc_id",
        "sequence",
        "state",
        "content_sha256",
        "profile_fingerprint",
        "vector_rows",
        "sparse_generation",
        "committed_at",
        "metadata",
    }:
        raise RuntimeError("rollback generation payload is invalid.")
    if (
        generation["owner_id"] != preflight.owner_id
        or generation["doc_id"] != preflight.doc_id
        or generation["sequence"] != preflight.source_sequence
        or generation["state"] not in {"active", "restored"}
        or generation["content_sha256"] != preflight.source_content_sha256
        or generation["profile_fingerprint"]
        != preflight.source_profile_fingerprint
        or generation["vector_rows"] != preflight.source_vector_rows
        or generation["sparse_generation"] != preflight.source_sparse_generation
        or not isinstance(generation["metadata"], Mapping)
    ):
        raise RuntimeError("rollback generation payload does not match preflight.")

    raw_vector = payload["vector_rows"]
    if isinstance(raw_vector, (str, bytes, bytearray)) or not isinstance(
        raw_vector, Sequence
    ):
        raise RuntimeError("rollback vector rows are invalid.")
    vector_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw_vector:
        if not isinstance(row, Mapping) or set(row) != {"id", "document", "metadata"}:
            raise RuntimeError("rollback vector row schema is invalid.")
        row_id = identifier(row["id"], "vector row id", 500)
        if row_id in seen:
            raise RuntimeError("rollback vector rows contain duplicate IDs.")
        seen.add(row_id)
        document = row["document"]
        metadata = row["metadata"]
        if not isinstance(document, str) or "\x00" in document:
            raise RuntimeError("rollback vector text is invalid.")
        if not isinstance(metadata, Mapping) or (
            metadata.get("owner_id") != preflight.owner_id
            or metadata.get("doc_id") != preflight.doc_id
        ):
            raise RuntimeError("rollback vector metadata escaped preflight scope.")
        vector_rows.append(
            {"id": row_id, "document": document, "metadata": dict(metadata)}
        )
    if len(vector_rows) != preflight.source_vector_rows:
        raise RuntimeError("rollback vector row count changed.")

    sparse = payload["sparse_snapshot"]
    if not isinstance(sparse, Mapping) or set(sparse) != {
        "owner_id",
        "doc_id",
        "generation",
        "profile_fingerprint",
        "metadata",
        "fields",
        "schema_version",
    }:
        raise RuntimeError("rollback sparse snapshot schema is invalid.")
    if (
        sparse["owner_id"] != preflight.owner_id
        or sparse["doc_id"] != preflight.doc_id
        or sparse["generation"] != preflight.source_sparse_generation
        or sparse["profile_fingerprint"] != preflight.source_profile_fingerprint
        or not isinstance(sparse["metadata"], Mapping)
        or sparse["schema_version"] != 1
    ):
        raise RuntimeError("rollback sparse snapshot does not match preflight.")
    raw_fields = sparse["fields"]
    if isinstance(raw_fields, (str, bytes, bytearray)) or not isinstance(
        raw_fields, Sequence
    ):
        raise RuntimeError("rollback sparse fields are invalid.")
    fields = [dict(field) if isinstance(field, Mapping) else field for field in raw_fields]
    if len(fields) != preflight.source_sparse_fields:
        raise RuntimeError("rollback sparse field count changed.")

    vector_digest = _sha256(vector_rows)
    sparse_identity_payload = {
        "generation": sparse["generation"],
        "profile_fingerprint": sparse["profile_fingerprint"],
        "metadata": dict(sparse["metadata"]),
        "fields": fields,
    }
    sparse_digest = _sha256(sparse_identity_payload)
    if (
        vector_digest != preflight.vector_snapshot_digest
        or sparse_digest != preflight.sparse_snapshot_digest
    ):
        raise RuntimeError("rollback payload snapshot digest changed.")
    if (
        _rollback_identity(
            preflight,
            vector_digest,
            sparse_digest,
            len(vector_rows),
            len(fields),
        )
        != preflight.rollback_identity_digest
    ):
        raise RuntimeError("rollback payload identity digest changed.")
    return {
        "schema_version": 1,
        "task_id": preflight.task_id,
        "preflight_digest": preflight.preflight_digest,
        "rollback_identity_digest": preflight.rollback_identity_digest,
        "owner_id": preflight.owner_id,
        "doc_id": preflight.doc_id,
        "generation": dict(generation),
        "vector_rows": vector_rows,
        "sparse_snapshot": {
            "owner_id": sparse["owner_id"],
            "doc_id": sparse["doc_id"],
            "generation": sparse["generation"],
            "profile_fingerprint": sparse["profile_fingerprint"],
            "metadata": dict(sparse["metadata"]),
            "fields": fields,
            "schema_version": 1,
        },
    }


@dataclass(frozen=True)
class RollbackEncryptionKey:
    key_id: str
    key: bytes = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_id", identifier(self.key_id, "key_id", 128))
        if not isinstance(self.key, bytes) or len(self.key) != 32:
            raise ValueError("rollback encryption key must contain exactly 32 bytes.")


def rollback_key_from_environment() -> RollbackEncryptionKey:
    raw_key = os.getenv("MIGRATION_ROLLBACK_KEY_B64")
    raw_id = os.getenv("MIGRATION_ROLLBACK_KEY_ID")
    if raw_key is None or raw_id is None:
        raise RuntimeError("rollback encryption key configuration is unavailable.")
    if raw_key != raw_key.strip() or raw_id != raw_id.strip():
        raise RuntimeError("rollback encryption key configuration must be canonical.")
    try:
        key = base64.b64decode(raw_key, validate=True)
    except Exception as exc:
        raise RuntimeError("rollback encryption key encoding is invalid.") from exc
    try:
        return RollbackEncryptionKey(raw_id, key)
    except ValueError as exc:
        raise RuntimeError("rollback encryption key configuration is invalid.") from exc


@dataclass(frozen=True)
class EncryptedRollbackManifest:
    task_id: str
    owner_id: str
    doc_id: str
    preflight_digest: str
    rollback_identity_digest: str
    source_sequence: int
    source_profile_fingerprint: str
    source_content_sha256: str
    vector_snapshot_digest: str
    sparse_snapshot_digest: str
    plaintext_sha256: str
    ciphertext_sha256: str
    plaintext_bytes: int
    ciphertext_bytes: int
    algorithm: str
    key_id: str
    nonce_b64: str
    aad_sha256: str
    created_at: float
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", identifier(self.task_id, "task_id", 64))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", identifier(self.doc_id, "doc_id", 200))
        object.__setattr__(
            self,
            "source_sequence",
            exact_integer(self.source_sequence, "source_sequence", 1, 2**63 - 1),
        )
        for name in (
            "preflight_digest",
            "rollback_identity_digest",
            "source_profile_fingerprint",
            "source_content_sha256",
            "vector_snapshot_digest",
            "sparse_snapshot_digest",
            "plaintext_sha256",
            "ciphertext_sha256",
            "aad_sha256",
        ):
            object.__setattr__(self, name, digest(getattr(self, name), name))
        for name in ("plaintext_bytes", "ciphertext_bytes"):
            object.__setattr__(
                self,
                name,
                exact_integer(getattr(self, name), name, 1, _MAX_PAYLOAD_BYTES + 1024),
            )
        if self.algorithm != _ALGORITHM:
            raise ValueError("rollback encryption algorithm is unsupported.")
        object.__setattr__(self, "key_id", identifier(self.key_id, "key_id", 128))
        if not isinstance(self.nonce_b64, str):
            raise ValueError("rollback nonce must be text.")
        try:
            nonce = base64.b64decode(self.nonce_b64, validate=True)
        except Exception as exc:
            raise ValueError("rollback nonce encoding is invalid.") from exc
        if len(nonce) != 12:
            raise ValueError("rollback nonce must contain 12 bytes.")
        object.__setattr__(self, "created_at", timestamp(self.created_at, "created_at"))
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("rollback artifact schema is unsupported.")

    @property
    def artifact_digest(self) -> str:
        stable = asdict(self)
        stable.pop("created_at", None)
        return hashlib.sha256(canonical_json_bytes(stable)).hexdigest()


__all__ = [
    "EncryptedRollbackManifest",
    "RollbackEncryptionKey",
    "canonical_json_bytes",
    "capture_rollback_payload",
    "rollback_key_from_environment",
    "strict_json_loads",
    "validate_rollback_payload",
]
