"""Build isolated target-profile shadow rows from retained source bytes."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any

from tools.embedding_adapters import EmbeddingEncoder, create_embedding_encoder
from tools.embedding_models import EmbeddingProfile
from tools.embedding_registry import resolve_embedding_profile
from tools.migration_shadow_store import ShadowBuild
from tools.migration_types import MigrationTask
from tools.security import normalize_owner_id
from tools.sparse_fields import build_sparse_fields
from tools.upload_storage import validated_owner_file_path

_BUILDER_CONTRACT = "rigorousrag-migration-shadow-builder-v1"
_MAX_VECTOR_DIMENSIONS = 1_000_000


def _parse_retained_source(source_path: str, owner_id: str) -> Any:
    """Use the current parser and complete final-index privacy boundary."""

    from tools.document_service import _enforce_index_redaction, _verify_source_identity
    from tools.ingestion import ingest_file

    result = ingest_file(source_path, owner_id=owner_id)
    if not bool(getattr(result, "success", False)) or getattr(result, "document", None) is None:
        raise ValueError("retained source could not be ingested for migration.")
    document = result.document
    _verify_source_identity(document, owner_id)
    _enforce_index_redaction(document)
    return document


def _parser_fingerprint(document: Any) -> str:
    raw_metadata = getattr(document, "metadata", {})
    metadata: dict[str, Any] = {}
    if isinstance(raw_metadata, Mapping):
        for key in (
            "parser",
            "parser_version",
            "ocr_used",
            "ocr_pages",
            "redaction",
            "document_identity",
        ):
            value = raw_metadata.get(key)
            if value is None or isinstance(value, (bool, int, str)):
                metadata[key] = value
    sections = getattr(document, "sections", ())
    section_count = len(sections) if hasattr(sections, "__len__") else None
    payload = json.dumps(
        {
            "contract": _BUILDER_CONTRACT,
            "metadata": metadata,
            "section_count": section_count,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _field_row(field: Any) -> dict[str, Any]:
    try:
        row = asdict(field)
    except (TypeError, ValueError) as exc:
        raise ValueError("sparse field could not be serialized.") from exc
    if not isinstance(row, dict):
        raise ValueError("sparse field serialization is invalid.")
    return row


def _validated_vector(
    value: Any,
    *,
    dimensions: int | None,
) -> list[float]:
    if isinstance(value, (str, bytes, bytearray)):
        raise RuntimeError("migration encoder returned an invalid vector.")
    try:
        raw = list(
            itertools.islice(
                iter(value),
                _MAX_VECTOR_DIMENSIONS + 1,
            )
        )
    except Exception as exc:
        raise RuntimeError("migration encoder returned an invalid vector.") from exc
    if not raw or len(raw) > _MAX_VECTOR_DIMENSIONS:
        raise RuntimeError("migration encoder returned an invalid vector dimension.")
    if dimensions is not None and len(raw) != dimensions:
        raise RuntimeError("migration encoder vector dimensions do not match the profile.")
    result: list[float] = []
    for item in raw:
        if isinstance(item, bool):
            raise RuntimeError("migration encoder returned a non-finite vector.")
        try:
            numeric = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("migration encoder returned a non-finite vector.") from exc
        if not math.isfinite(numeric):
            raise RuntimeError("migration encoder returned a non-finite vector.")
        result.append(numeric)
    return result


class MigrationShadowBuilder:
    """Reparse, redact, field, and encode one retained migration source."""

    def __init__(
        self,
        *,
        registry: Any = None,
        parser: Callable[[str, str], Any] | None = None,
        encoder_factory: Callable[[EmbeddingProfile], EmbeddingEncoder] | None = None,
    ) -> None:
        if registry is None:
            from tools.document_store import get_document_store

            registry = get_document_store()
        if not callable(getattr(registry, "get", None)):
            raise ValueError("registry must expose get().")
        selected_parser = parser or _parse_retained_source
        selected_encoder = encoder_factory or create_embedding_encoder
        if not callable(selected_parser):
            raise ValueError("parser must be callable.")
        if not callable(selected_encoder):
            raise ValueError("encoder_factory must be callable.")
        self._registry = registry
        self._parser = selected_parser
        self._encoder_factory = selected_encoder

    def _profile(self, task: MigrationTask) -> EmbeddingProfile:
        profile = resolve_embedding_profile(
            task.target_profile_name,
            allow_compatibility=False,
        )
        if profile.fingerprint != task.target_profile_fingerprint:
            raise RuntimeError("target embedding profile fingerprint changed.")
        return profile

    def _source_path(self, task: MigrationTask) -> str:
        record = self._registry.get(
            owner_id=task.owner_id,
            doc_id=task.doc_id,
        )
        if not isinstance(record, Mapping) or not record.get("source_retained"):
            raise RuntimeError("migration retained source is unavailable.")
        raw_path = record.get("source_path")
        candidate = validated_owner_file_path(
            self._registry.upload_root,
            raw_path,
        )
        if candidate is None:
            raise RuntimeError("migration retained source path is invalid.")
        return str(candidate)

    def __call__(self, task: MigrationTask) -> ShadowBuild:
        if not isinstance(task, MigrationTask):
            raise ValueError("task must be a MigrationTask.")
        owner = normalize_owner_id(task.owner_id)
        profile = self._profile(task)
        source_path = self._source_path(task)
        document = self._parser(source_path, owner)
        if getattr(document, "id", None) != task.doc_id:
            raise RuntimeError("retained source document identity changed.")
        text = getattr(document, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("privacy-finalized migration document is empty.")
        content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        fields = build_sparse_fields(document, doc_id=task.doc_id)
        if not fields:
            raise RuntimeError("migration document produced no sparse fields.")
        encoder = self._encoder_factory(profile)
        if getattr(encoder, "profile", None) != profile:
            raise RuntimeError("migration encoder profile is incompatible.")
        vectors = encoder.encode_passages(tuple(field.text for field in fields))
        if len(vectors) != len(fields):
            raise RuntimeError("migration encoder returned the wrong row count.")

        vector_rows: list[dict[str, Any]] = []
        sparse_rows: list[dict[str, Any]] = []
        inferred_dimensions: int | None = profile.dimensions
        for field, raw_vector in zip(fields, vectors, strict=True):
            vector = _validated_vector(
                raw_vector,
                dimensions=inferred_dimensions,
            )
            if inferred_dimensions is None:
                inferred_dimensions = len(vector)
            sparse = _field_row(field)
            sparse_rows.append(sparse)
            vector_rows.append(
                {
                    "row_id": field.field_id,
                    "text": field.text,
                    "embedding": vector,
                    "metadata": {
                        "owner_id": owner,
                        "doc_id": task.doc_id,
                        "source_sequence": task.source_sequence,
                        "target_profile_name": profile.alias,
                        "target_profile_fingerprint": profile.fingerprint,
                        "content_sha256": content_sha256,
                        "field_id": field.field_id,
                        "field_type": field.field_type,
                        "field_position": field.position,
                        "page_number": field.page_number,
                        "section": field.section,
                    },
                }
            )
        return ShadowBuild(
            content_sha256=content_sha256,
            parser_fingerprint=_parser_fingerprint(document),
            vector_rows=tuple(vector_rows),
            sparse_rows=tuple(sparse_rows),
        )


__all__ = ["MigrationShadowBuilder"]
