"""Non-mutating cutover preflight identities for validated migration shadows."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from tools.migration_promotion import PromotionReport
from tools.migration_types import digest, exact_integer, identifier, timestamp
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_MAX_ITEMS = 1_000_000
_MAX_DEPTH = 16
_MAX_STRING = 5_000_000


def _canonical(value: Any, *, depth: int, counter: list[int]) -> Any:
    if depth > _MAX_DEPTH:
        raise ValueError("snapshot identity exceeds the nesting limit.")
    counter[0] += 1
    if counter[0] > _MAX_ITEMS:
        raise ValueError("snapshot identity exceeds the item limit.")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("snapshot identity may not contain non-finite numbers.")
        return value
    if isinstance(value, str):
        if len(value) > _MAX_STRING or "\x00" in value:
            raise ValueError("snapshot identity contains invalid text.")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        try:
            items = value.items()
        except Exception as exc:
            raise ValueError("snapshot identity mapping is unreadable.") from exc
        for raw_key, item in items:
            key = identifier(raw_key, "snapshot key", 500)
            if key in result:
                raise ValueError("snapshot identity contains a duplicate key.")
            result[key] = _canonical(item, depth=depth + 1, counter=counter)
        return result
    if isinstance(value, (bytes, bytearray)):
        raise ValueError("snapshot identity bytes are unsupported.")
    if isinstance(value, Sequence):
        return [
            _canonical(item, depth=depth + 1, counter=counter)
            for item in value
        ]
    try:
        iterator = iter(value)
    except Exception as exc:
        raise ValueError("snapshot identity contains an unsupported value.") from exc
    return [
        _canonical(item, depth=depth + 1, counter=counter)
        for item in iterator
    ]


def _sha256(value: Any) -> str:
    normalized = _canonical(value, depth=0, counter=[0])
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _attr(value: Any, name: str) -> Any:
    try:
        return getattr(value, name)
    except Exception as exc:
        raise ValueError(f"snapshot {name} is unavailable.") from exc


def _vector_identity(vector: Any, owner_id: str, doc_id: str) -> tuple[str, int]:
    if _attr(vector, "owner_id") != owner_id or _attr(vector, "doc_id") != doc_id:
        raise RuntimeError("vector rollback snapshot escaped task scope.")
    ids = tuple(_attr(vector, "ids"))
    documents = tuple(_attr(vector, "documents"))
    metadatas = tuple(_attr(vector, "metadatas"))
    if not len(ids) == len(documents) == len(metadatas):
        raise RuntimeError("vector rollback snapshot arrays are inconsistent.")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_id, text, metadata in zip(ids, documents, metadatas, strict=True):
        row_id = identifier(raw_id, "vector row id", 500)
        if row_id in seen:
            raise RuntimeError("vector rollback snapshot contains duplicate row IDs.")
        seen.add(row_id)
        if not isinstance(text, str) or "\x00" in text:
            raise ValueError("vector rollback text is invalid.")
        if not isinstance(metadata, Mapping):
            raise ValueError("vector rollback metadata must be a mapping.")
        if metadata.get("owner_id") != owner_id or metadata.get("doc_id") != doc_id:
            raise RuntimeError("vector rollback metadata escaped task scope.")
        rows.append({"id": row_id, "document": text, "metadata": dict(metadata)})
    return _sha256(rows), len(rows)


def _field_identity(field: Any) -> dict[str, Any]:
    return {
        "field_id": identifier(_attr(field, "field_id"), "field_id", 500),
        "field_type": identifier(_attr(field, "field_type"), "field_type", 100),
        "text": _attr(field, "text"),
        "position": _attr(field, "position"),
        "token_count": _attr(field, "token_count"),
        "page_number": _attr(field, "page_number"),
        "section": _attr(field, "section"),
        "metadata": dict(_attr(field, "metadata")),
    }


def _sparse_identity(
    sparse: Any,
    owner_id: str,
    doc_id: str,
    source_profile_fingerprint: str,
    source_sparse_generation: int,
) -> tuple[str, int]:
    if sparse is None:
        raise RuntimeError("sparse rollback snapshot is unavailable.")
    if _attr(sparse, "owner_id") != owner_id or _attr(sparse, "doc_id") != doc_id:
        raise RuntimeError("sparse rollback snapshot escaped task scope.")
    if _attr(sparse, "generation") != source_sparse_generation:
        raise RuntimeError("sparse rollback generation changed.")
    if _attr(sparse, "profile_fingerprint") != source_profile_fingerprint:
        raise RuntimeError("sparse rollback profile changed.")
    fields = tuple(_attr(sparse, "fields"))
    rows = [_field_identity(field) for field in fields]
    return _sha256(
        {
            "generation": source_sparse_generation,
            "profile_fingerprint": source_profile_fingerprint,
            "metadata": dict(_attr(sparse, "metadata")),
            "fields": rows,
        }
    ), len(rows)


@dataclass(frozen=True)
class CutoverPreflight:
    task_id: str
    owner_id: str
    doc_id: str
    source_sequence: int
    source_profile_fingerprint: str
    target_profile_fingerprint: str
    source_content_sha256: str
    validation_digest: str
    promotion_report_digest: str
    benchmark_fingerprint: str
    vector_snapshot_digest: str
    sparse_snapshot_digest: str
    rollback_identity_digest: str
    target_artifact_digest: str
    source_vector_rows: int
    source_sparse_generation: int
    source_sparse_fields: int
    target_vector_rows: int
    target_sparse_rows: int
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
            "source_profile_fingerprint",
            "target_profile_fingerprint",
            "source_content_sha256",
            "validation_digest",
            "promotion_report_digest",
            "benchmark_fingerprint",
            "vector_snapshot_digest",
            "sparse_snapshot_digest",
            "rollback_identity_digest",
            "target_artifact_digest",
        ):
            object.__setattr__(self, name, digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "source_vector_rows",
            exact_integer(self.source_vector_rows, "source_vector_rows", 1, 100_000_000),
        )
        object.__setattr__(
            self,
            "source_sparse_generation",
            exact_integer(
                self.source_sparse_generation,
                "source_sparse_generation",
                1,
                2**63 - 1,
            ),
        )
        object.__setattr__(
            self,
            "source_sparse_fields",
            exact_integer(self.source_sparse_fields, "source_sparse_fields", 1, 100_000_000),
        )
        object.__setattr__(
            self,
            "target_vector_rows",
            exact_integer(self.target_vector_rows, "target_vector_rows", 1, 100_000_000),
        )
        object.__setattr__(
            self,
            "target_sparse_rows",
            exact_integer(self.target_sparse_rows, "target_sparse_rows", 1, 100_000_000),
        )
        object.__setattr__(self, "created_at", timestamp(self.created_at, "created_at"))
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("cutover preflight schema is unsupported.")

    @property
    def preflight_digest(self) -> str:
        stable = asdict(self)
        stable.pop("created_at", None)
        return _sha256(stable)


def build_cutover_preflight(
    *,
    task: Any,
    shadow_manifest: Any,
    promotion_report: PromotionReport,
    authoritative_snapshot: Any,
    now: float | None = None,
) -> CutoverPreflight:
    if getattr(task, "state", None) != "validated":
        raise ValueError("cutover preflight requires a validated migration task.")
    if not isinstance(promotion_report, PromotionReport):
        raise ValueError("promotion_report must be PromotionReport.")
    if promotion_report.decision != "eligible":
        raise RuntimeError("cutover preflight requires an eligible promotion report.")
    if promotion_report.policy_id != "paired-promotion-v1":
        raise RuntimeError("cutover preflight requires paired statistical promotion.")

    task_id = identifier(getattr(task, "task_id", None), "task_id", 64)
    owner = normalize_owner_id(getattr(task, "owner_id", None))
    doc_id = identifier(getattr(task, "doc_id", None), "doc_id", 200)
    source_sequence = exact_integer(
        getattr(task, "source_sequence", None),
        "source_sequence",
        1,
        2**63 - 1,
    )
    source_profile = digest(
        getattr(task, "source_profile_fingerprint", None),
        "source_profile_fingerprint",
    )
    target_profile = digest(
        getattr(task, "target_profile_fingerprint", None),
        "target_profile_fingerprint",
    )
    validation = digest(
        getattr(shadow_manifest, "validation_digest", None),
        "validation_digest",
    )

    expected_pairs = (
        (getattr(shadow_manifest, "task_id", None), task_id),
        (getattr(shadow_manifest, "owner_id", None), owner),
        (getattr(shadow_manifest, "doc_id", None), doc_id),
        (getattr(shadow_manifest, "source_sequence", None), source_sequence),
        (getattr(shadow_manifest, "source_profile_fingerprint", None), source_profile),
        (getattr(shadow_manifest, "target_profile_fingerprint", None), target_profile),
        (getattr(task, "validation_digest", None), validation),
        (promotion_report.task_id, task_id),
        (promotion_report.owner_id, owner),
        (promotion_report.doc_id, doc_id),
        (promotion_report.source_sequence, source_sequence),
        (promotion_report.source_profile_fingerprint, source_profile),
        (promotion_report.target_profile_fingerprint, target_profile),
        (promotion_report.validation_digest, validation),
    )
    if any(actual != expected for actual, expected in expected_pairs):
        raise RuntimeError("task, shadow and promotion identities are inconsistent.")

    if getattr(authoritative_snapshot, "owner_id", None) != owner or getattr(
        authoritative_snapshot, "doc_id", None
    ) != doc_id:
        raise RuntimeError("authoritative rollback snapshot escaped task scope.")
    generation = getattr(authoritative_snapshot, "generation", None)
    if generation is None or getattr(generation, "state", None) not in {
        "active",
        "restored",
    }:
        raise RuntimeError("authoritative source generation is unavailable.")
    if (
        getattr(generation, "sequence", None) != source_sequence
        or getattr(generation, "profile_fingerprint", None) != source_profile
    ):
        raise RuntimeError("authoritative source generation changed.")
    source_content = digest(
        getattr(generation, "content_sha256", None),
        "source_content_sha256",
    )
    if getattr(shadow_manifest, "content_sha256", None) != source_content:
        raise RuntimeError("shadow content does not match the authoritative source.")
    if getattr(generation, "vector_rows", None) <= 0 or getattr(
        generation, "sparse_generation", None
    ) <= 0:
        raise RuntimeError("authoritative source generation is incomplete.")

    stores = getattr(authoritative_snapshot, "stores", None)
    if stores is None:
        raise ValueError("authoritative stores snapshot is unavailable.")
    vector_digest, vector_rows = _vector_identity(
        getattr(stores, "vector", None), owner, doc_id
    )
    if vector_rows != getattr(generation, "vector_rows", None):
        raise RuntimeError("vector rollback row count differs from the generation record.")
    sparse_digest, sparse_fields = _sparse_identity(
        getattr(stores, "sparse", None),
        owner,
        doc_id,
        source_profile,
        getattr(generation, "sparse_generation", None),
    )

    target_vector_rows = exact_integer(
        getattr(shadow_manifest, "vector_count", None),
        "target_vector_rows",
        1,
        100_000_000,
    )
    target_sparse_rows = exact_integer(
        getattr(shadow_manifest, "sparse_count", None),
        "target_sparse_rows",
        1,
        100_000_000,
    )
    rollback_identity = _sha256(
        {
            "owner_id": owner,
            "doc_id": doc_id,
            "source_sequence": source_sequence,
            "source_profile_fingerprint": source_profile,
            "source_content_sha256": source_content,
            "vector_snapshot_digest": vector_digest,
            "sparse_snapshot_digest": sparse_digest,
            "vector_rows": vector_rows,
            "sparse_generation": getattr(generation, "sparse_generation", None),
            "sparse_fields": sparse_fields,
        }
    )
    target_identity = _sha256(
        {
            "validation_digest": validation,
            "target_profile_fingerprint": target_profile,
            "content_sha256": source_content,
            "vector_sha256": digest(
                getattr(shadow_manifest, "vector_sha256", None),
                "vector_sha256",
            ),
            "sparse_sha256": digest(
                getattr(shadow_manifest, "sparse_sha256", None),
                "sparse_sha256",
            ),
            "vector_count": target_vector_rows,
            "sparse_count": target_sparse_rows,
        }
    )
    return CutoverPreflight(
        task_id=task_id,
        owner_id=owner,
        doc_id=doc_id,
        source_sequence=source_sequence,
        source_profile_fingerprint=source_profile,
        target_profile_fingerprint=target_profile,
        source_content_sha256=source_content,
        validation_digest=validation,
        promotion_report_digest=promotion_report.report_digest,
        benchmark_fingerprint=promotion_report.benchmark_fingerprint,
        vector_snapshot_digest=vector_digest,
        sparse_snapshot_digest=sparse_digest,
        rollback_identity_digest=rollback_identity,
        target_artifact_digest=target_identity,
        source_vector_rows=vector_rows,
        source_sparse_generation=getattr(generation, "sparse_generation", None),
        source_sparse_fields=sparse_fields,
        target_vector_rows=target_vector_rows,
        target_sparse_rows=target_sparse_rows,
        created_at=time.time() if now is None else timestamp(now, "created_at"),
    )


__all__ = ["CutoverPreflight", "build_cutover_preflight"]
