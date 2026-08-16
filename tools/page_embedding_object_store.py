"""Lossless object-store persistence for page-native late-interaction artifacts.

Vectors are encoded as IEEE-754 float64 so decoding preserves the exact Python float
values used by ``PageEmbeddingArtifact.artifact_sha256``. The control-plane manifest binds
both the intrinsic page artifact hash and the provider object-content hash.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.document_evidence_bundle import DocumentEvidenceBundle
from tools.page_late_interaction import PageEmbeddingArtifact, PatchBBox
from tools.production_runtime import ObjectRecord, ObjectStore
from tools.security import normalize_owner_id

_MAGIC = b"RRPEMB01"
_HEADER_LIMIT = 8 * 1024 * 1024
_MAX_VECTOR_VALUES = 200_000_000
_MAX_OBJECT_BYTES = 2_000_000_000


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


@dataclass(frozen=True)
class PageEmbeddingObjectRef:
    page_artifact_sha256: str
    object_id: str
    object_content_sha256: str
    object_version: int
    size_bytes: int
    page_number: int
    model_id: str


@dataclass(frozen=True)
class PageEmbeddingStorageManifest:
    owner_id: str
    bundle_fingerprint: str
    objects: tuple[PageEmbeddingObjectRef, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "bundle_fingerprint", _sha(self.bundle_fingerprint, "bundle_fingerprint"))
        object.__setattr__(self, "fingerprint", _sha(self.fingerprint, "fingerprint"))
        if len({item.page_artifact_sha256 for item in self.objects}) != len(self.objects):
            raise ValueError("storage manifest contains duplicate page artifacts")
        if len({item.object_id for item in self.objects}) != len(self.objects):
            raise ValueError("storage manifest contains duplicate object IDs")


def encode_page_embedding_artifact(artifact: PageEmbeddingArtifact) -> bytes:
    if not isinstance(artifact, PageEmbeddingArtifact):
        raise TypeError("artifact must be PageEmbeddingArtifact")
    patch_count = len(artifact.patch_vectors)
    values = patch_count * artifact.dimension
    if values > _MAX_VECTOR_VALUES:
        raise ValueError("page embedding contains too many vector values")
    header = {
        "contract": "rigorousrag-page-embedding-object-v1",
        "owner_id": artifact.owner_id,
        "doc_id": artifact.doc_id,
        "source_sha256": artifact.source_sha256,
        "page_number": artifact.page_number,
        "rendered_page_sha256": artifact.rendered_page_sha256,
        "model_id": artifact.model_id,
        "dimension": artifact.dimension,
        "patch_count": patch_count,
        "patch_boxes": [asdict(item) for item in artifact.patch_boxes],
        "patch_region_ids": list(artifact.patch_region_ids),
        "artifact_sha256": artifact.artifact_sha256,
        "float_encoding": "ieee754-f64-le",
    }
    header_bytes = _canonical(header)
    if len(header_bytes) > _HEADER_LIMIT:
        raise ValueError("page embedding header exceeds the size limit")
    body = bytearray(values * 8)
    offset = 0
    for row in artifact.patch_vectors:
        for value in row:
            struct.pack_into("<d", body, offset, float(value))
            offset += 8
    payload = _MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + bytes(body)
    if len(payload) > _MAX_OBJECT_BYTES:
        raise ValueError("page embedding object exceeds the object size limit")
    return payload


def decode_page_embedding_artifact(payload: bytes) -> PageEmbeddingArtifact:
    if not isinstance(payload, bytes) or len(payload) < len(_MAGIC) + 4 or len(payload) > _MAX_OBJECT_BYTES:
        raise ValueError("page embedding object payload is invalid")
    if not payload.startswith(_MAGIC):
        raise ValueError("page embedding object magic/version is unsupported")
    header_length = struct.unpack_from("<I", payload, len(_MAGIC))[0]
    if not 1 <= header_length <= _HEADER_LIMIT:
        raise ValueError("page embedding object header length is invalid")
    header_start = len(_MAGIC) + 4
    header_end = header_start + header_length
    if header_end > len(payload):
        raise ValueError("page embedding object is truncated")
    try:
        header = json.loads(payload[header_start:header_end].decode("utf-8"))
    except Exception as exc:
        raise ValueError("page embedding object header is invalid JSON") from exc
    if not isinstance(header, Mapping) or header.get("contract") != "rigorousrag-page-embedding-object-v1":
        raise ValueError("page embedding object contract is invalid")
    dimension = int(header["dimension"])
    patch_count = int(header["patch_count"])
    if not 1 <= dimension <= 8192 or not 1 <= patch_count <= 1_000_000:
        raise ValueError("page embedding object dimensions are invalid")
    value_count = dimension * patch_count
    if value_count > _MAX_VECTOR_VALUES or len(payload) - header_end != value_count * 8:
        raise ValueError("page embedding object vector payload length is invalid")
    vectors: list[tuple[float, ...]] = []
    cursor = header_end
    for _ in range(patch_count):
        row = struct.unpack_from(f"<{dimension}d", payload, cursor)
        cursor += dimension * 8
        vectors.append(tuple(row))
    boxes_raw = header.get("patch_boxes", [])
    if not isinstance(boxes_raw, list) or len(boxes_raw) != patch_count:
        raise ValueError("page embedding patch boxes are invalid")
    boxes = tuple(PatchBBox(float(item["x0"]), float(item["y0"]), float(item["x1"]), float(item["y1"])) for item in boxes_raw)
    region_ids = header.get("patch_region_ids", [])
    if not isinstance(region_ids, list) or len(region_ids) != patch_count:
        raise ValueError("page embedding patch region IDs are invalid")
    artifact = PageEmbeddingArtifact(
        owner_id=str(header["owner_id"]),
        doc_id=str(header["doc_id"]),
        source_sha256=str(header["source_sha256"]),
        page_number=int(header["page_number"]),
        rendered_page_sha256=str(header["rendered_page_sha256"]),
        model_id=str(header["model_id"]),
        patch_vectors=tuple(vectors),
        patch_boxes=boxes,
        patch_region_ids=tuple(str(item) for item in region_ids),
    )
    expected = _sha(str(header["artifact_sha256"]), "artifact_sha256")
    if artifact.artifact_sha256 != expected:
        raise RuntimeError("decoded page embedding artifact failed intrinsic SHA-256 verification")
    return artifact


class PageEmbeddingObjectRepository:
    def __init__(self, object_store: ObjectStore) -> None:
        if object_store is None:
            raise ValueError("object_store is required")
        self.object_store = object_store

    @staticmethod
    def object_id(artifact_sha256: str) -> str:
        return f"multimodal/page-embeddings/{_sha(artifact_sha256, 'artifact_sha256')}.rrpemb"

    def put(self, artifact: PageEmbeddingArtifact) -> PageEmbeddingObjectRef:
        payload = encode_page_embedding_artifact(artifact)
        object_id = self.object_id(artifact.artifact_sha256)
        content_sha = hashlib.sha256(payload).hexdigest()
        try:
            existing = self.object_store.head(artifact.owner_id, object_id)
        except (KeyError, FileNotFoundError):
            existing = None
        if existing is not None:
            if existing.deleted or existing.content_sha256 != content_sha or existing.size_bytes != len(payload):
                raise RuntimeError("content-addressed page embedding object identity collision")
            record = existing
        else:
            record = self.object_store.put(
                artifact.owner_id,
                object_id,
                payload,
                content_type="application/vnd.rigorousrag.page-embedding+binary",
                metadata={
                    "page-artifact-sha256": artifact.artifact_sha256,
                    "doc-id": artifact.doc_id,
                    "page-number": str(artifact.page_number),
                    "model-id": artifact.model_id,
                },
            )
            if record.content_sha256 != content_sha:
                raise RuntimeError("object store returned a mismatched content SHA-256")
        return PageEmbeddingObjectRef(
            artifact.artifact_sha256,
            object_id,
            record.content_sha256,
            record.version,
            record.size_bytes,
            artifact.page_number,
            artifact.model_id,
        )

    def get(self, owner_id: str, reference: PageEmbeddingObjectRef) -> PageEmbeddingArtifact:
        owner = normalize_owner_id(owner_id)
        if not isinstance(reference, PageEmbeddingObjectRef):
            raise TypeError("reference must be PageEmbeddingObjectRef")
        payload = self.object_store.get(owner, reference.object_id)
        if hashlib.sha256(payload).hexdigest() != reference.object_content_sha256:
            raise RuntimeError("page embedding object content SHA-256 mismatch")
        artifact = decode_page_embedding_artifact(payload)
        if artifact.artifact_sha256 != reference.page_artifact_sha256:
            raise RuntimeError("page embedding object does not match its storage manifest")
        if artifact.page_number != reference.page_number or artifact.model_id != reference.model_id:
            raise RuntimeError("page embedding object metadata differs from its storage manifest")
        return artifact

    def persist_bundle_pages(
        self,
        bundle: DocumentEvidenceBundle,
        artifacts: Sequence[PageEmbeddingArtifact],
    ) -> PageEmbeddingStorageManifest:
        expected = {item.page_artifact_sha256 for item in bundle.page_embeddings}
        supplied = {item.artifact_sha256 for item in artifacts}
        if expected != supplied:
            raise ValueError("page embedding artifacts do not match the document evidence bundle")
        references = tuple(sorted((self.put(item) for item in artifacts), key=lambda item: (item.page_number, item.page_artifact_sha256)))
        payload = {
            "contract": "rigorousrag-page-embedding-storage-manifest-v1",
            "owner_id": bundle.owner_id,
            "bundle_fingerprint": bundle.fingerprint,
            "objects": [asdict(item) for item in references],
        }
        fingerprint = hashlib.sha256(_canonical(payload)).hexdigest()
        return PageEmbeddingStorageManifest(bundle.owner_id, bundle.fingerprint, references, fingerprint)

    def load_bundle_pages(
        self,
        bundle: DocumentEvidenceBundle,
        manifest: PageEmbeddingStorageManifest,
    ) -> tuple[PageEmbeddingArtifact, ...]:
        if manifest.owner_id != bundle.owner_id or manifest.bundle_fingerprint != bundle.fingerprint:
            raise ValueError("page embedding storage manifest does not belong to the document evidence bundle")
        expected = {item.page_artifact_sha256 for item in bundle.page_embeddings}
        actual = {item.page_artifact_sha256 for item in manifest.objects}
        if expected != actual:
            raise RuntimeError("page embedding storage manifest is incomplete or contains unexpected artifacts")
        return tuple(self.get(bundle.owner_id, item) for item in manifest.objects)


__all__ = [
    "PageEmbeddingObjectRef",
    "PageEmbeddingObjectRepository",
    "PageEmbeddingStorageManifest",
    "decode_page_embedding_artifact",
    "encode_page_embedding_artifact",
]
