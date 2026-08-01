"""Manifest-last isolated vector/sparse artifacts for profile migrations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.migration_types import digest, exact_integer, identifier, timestamp
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_ROWS = 100_000
_MAX_FILE_BYTES = 512 * 1024 * 1024
_MAX_DEPTH = 12
_MAX_ITEMS = 1_000_000
_MAX_STRING = 2_000_000


def _redirecting(value: os.stat_result) -> bool:
    return stat.S_ISLNK(value.st_mode) or bool(
        int(getattr(value, "st_file_attributes", 0)) & _REPARSE
    )


def _root(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("shadow root must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("shadow root is invalid.")
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
            raise ValueError("shadow root could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("shadow root may not contain redirects.")
    return absolute


def _canonical(value: Any, *, depth: int, counter: list[int]) -> Any:
    if depth > _MAX_DEPTH:
        raise ValueError("shadow JSON exceeds the nesting limit.")
    counter[0] += 1
    if counter[0] > _MAX_ITEMS:
        raise ValueError("shadow JSON exceeds the item limit.")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("shadow JSON may not contain non-finite numbers.")
        return value
    if isinstance(value, str):
        if len(value) > _MAX_STRING or "\x00" in value:
            raise ValueError("shadow JSON string is invalid or too long.")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        try:
            items = value.items()
        except Exception as exc:
            raise ValueError("shadow JSON mapping is unreadable.") from exc
        for raw_key, item in items:
            key = identifier(raw_key, "shadow JSON key", 500)
            if key in result:
                raise ValueError("shadow JSON contains a duplicate key.")
            result[key] = _canonical(
                item,
                depth=depth + 1,
                counter=counter,
            )
        return result
    if isinstance(value, (bytes, bytearray)):
        raise ValueError("shadow JSON bytes are unsupported.")
    try:
        iterator = iter(value)
    except Exception as exc:
        raise ValueError("shadow JSON value is unsupported.") from exc
    result_list: list[Any] = []
    try:
        for item in iterator:
            result_list.append(
                _canonical(
                    item,
                    depth=depth + 1,
                    counter=counter,
                )
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("shadow JSON iterable is unreadable.") from exc
    return result_list


def _rows(
    values: Iterable[Mapping[str, Any]],
    label: str,
) -> tuple[dict[str, Any], ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an iterable of mappings.")
    rows: list[dict[str, Any]] = []
    counter = [0]
    try:
        for value in values:
            if len(rows) >= _MAX_ROWS:
                raise ValueError(f"{label} exceeds the row limit.")
            if not isinstance(value, Mapping):
                raise ValueError(f"every {label} row must be a mapping.")
            normalized = _canonical(value, depth=0, counter=counter)
            if not isinstance(normalized, dict):
                raise ValueError(f"every {label} row must normalize to an object.")
            rows.append(normalized)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{label} is not safely iterable.") from exc
    if not rows:
        raise ValueError(f"{label} must contain at least one row.")
    return tuple(rows)


def _encoded(values: tuple[dict[str, Any], ...]) -> bytes:
    payload = json.dumps(
        list(values),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(payload) > _MAX_FILE_BYTES:
        raise ValueError("shadow artifact exceeds the file-size limit.")
    return payload


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _strict_loads(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise RuntimeError(f"{label} is invalid JSON.") from exc


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            if os.name != "nt":
                raise
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ShadowBuild:
    content_sha256: str
    parser_fingerprint: str
    vector_rows: tuple[dict[str, Any], ...]
    sparse_rows: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "content_sha256",
            digest(self.content_sha256, "content_sha256"),
        )
        object.__setattr__(
            self,
            "parser_fingerprint",
            digest(self.parser_fingerprint, "parser_fingerprint"),
        )
        object.__setattr__(
            self,
            "vector_rows",
            _rows(self.vector_rows, "vector_rows"),
        )
        object.__setattr__(
            self,
            "sparse_rows",
            _rows(self.sparse_rows, "sparse_rows"),
        )


@dataclass(frozen=True)
class ShadowArtifactManifest:
    task_id: str
    owner_id: str
    doc_id: str
    source_sequence: int
    source_profile_fingerprint: str
    target_profile_name: str
    target_profile_fingerprint: str
    content_sha256: str
    parser_fingerprint: str
    vector_count: int
    sparse_count: int
    vector_sha256: str
    sparse_sha256: str
    vector_bytes: int
    sparse_bytes: int
    created_at: float
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", identifier(self.task_id, "task_id", 64))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", identifier(self.doc_id, "doc_id"))
        object.__setattr__(
            self,
            "source_sequence",
            exact_integer(
                self.source_sequence,
                "source_sequence",
                1,
                2**63 - 1,
            ),
        )
        object.__setattr__(
            self,
            "source_profile_fingerprint",
            digest(
                self.source_profile_fingerprint,
                "source_profile_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "target_profile_name",
            identifier(self.target_profile_name, "target_profile_name"),
        )
        object.__setattr__(
            self,
            "target_profile_fingerprint",
            digest(
                self.target_profile_fingerprint,
                "target_profile_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "content_sha256",
            digest(self.content_sha256, "content_sha256"),
        )
        object.__setattr__(
            self,
            "parser_fingerprint",
            digest(self.parser_fingerprint, "parser_fingerprint"),
        )
        object.__setattr__(
            self,
            "vector_count",
            exact_integer(self.vector_count, "vector_count", 1, _MAX_ROWS),
        )
        object.__setattr__(
            self,
            "sparse_count",
            exact_integer(self.sparse_count, "sparse_count", 1, _MAX_ROWS),
        )
        object.__setattr__(
            self,
            "vector_sha256",
            digest(self.vector_sha256, "vector_sha256"),
        )
        object.__setattr__(
            self,
            "sparse_sha256",
            digest(self.sparse_sha256, "sparse_sha256"),
        )
        object.__setattr__(
            self,
            "vector_bytes",
            exact_integer(
                self.vector_bytes,
                "vector_bytes",
                1,
                _MAX_FILE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "sparse_bytes",
            exact_integer(
                self.sparse_bytes,
                "sparse_bytes",
                1,
                _MAX_FILE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "created_at",
            timestamp(self.created_at, "created_at"),
        )
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("shadow artifact schema is unsupported.")

    @property
    def validation_digest(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return _sha256(payload)


def _artifact_identity(manifest: ShadowArtifactManifest) -> tuple[Any, ...]:
    return (
        manifest.task_id,
        manifest.owner_id,
        manifest.doc_id,
        manifest.source_sequence,
        manifest.source_profile_fingerprint,
        manifest.target_profile_name,
        manifest.target_profile_fingerprint,
        manifest.content_sha256,
        manifest.parser_fingerprint,
        manifest.vector_count,
        manifest.sparse_count,
        manifest.vector_sha256,
        manifest.sparse_sha256,
        manifest.vector_bytes,
        manifest.sparse_bytes,
        manifest.schema_version,
    )


class MigrationShadowStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = _root(root)
        self.root.mkdir(parents=True, exist_ok=True)
        info = self.root.lstat()
        if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("shadow root must be a regular directory.")
        self._root_identity = (int(info.st_dev), int(info.st_ino))
        self._lock = threading.RLock()

    def _verify_root(self) -> None:
        info = self.root.lstat()
        if (
            _redirecting(info)
            or not stat.S_ISDIR(info.st_mode)
            or (int(info.st_dev), int(info.st_ino)) != self._root_identity
        ):
            raise RuntimeError("shadow root identity changed.")

    def _task_directory(self, task_id: str) -> Path:
        return self.root / identifier(task_id, "task_id", 64)

    @staticmethod
    def _manifest_payload(manifest: ShadowArtifactManifest) -> bytes:
        return json.dumps(
            asdict(manifest),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @staticmethod
    def _read_bounded(path: Path) -> bytes:
        info = path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("shadow artifact member is not a regular file.")
        if info.st_size <= 0 or info.st_size > _MAX_FILE_BYTES:
            raise RuntimeError("shadow artifact member exceeds its size limit.")
        with path.open("rb") as handle:
            payload = handle.read(_MAX_FILE_BYTES + 1)
        if len(payload) > _MAX_FILE_BYTES:
            raise RuntimeError("shadow artifact member exceeds its size limit.")
        return payload

    def write(
        self,
        *,
        task: Any,
        build: ShadowBuild,
        now: float | None = None,
    ) -> ShadowArtifactManifest:
        if not isinstance(build, ShadowBuild):
            raise ValueError("build must be a ShadowBuild.")
        vector_payload = _encoded(build.vector_rows)
        sparse_payload = _encoded(build.sparse_rows)
        manifest = ShadowArtifactManifest(
            task_id=task.task_id,
            owner_id=task.owner_id,
            doc_id=task.doc_id,
            source_sequence=task.source_sequence,
            source_profile_fingerprint=task.source_profile_fingerprint,
            target_profile_name=task.target_profile_name,
            target_profile_fingerprint=task.target_profile_fingerprint,
            content_sha256=build.content_sha256,
            parser_fingerprint=build.parser_fingerprint,
            vector_count=len(build.vector_rows),
            sparse_count=len(build.sparse_rows),
            vector_sha256=_sha256(vector_payload),
            sparse_sha256=_sha256(sparse_payload),
            vector_bytes=len(vector_payload),
            sparse_bytes=len(sparse_payload),
            created_at=time.time() if now is None else timestamp(now),
        )
        destination = self._task_directory(manifest.task_id)
        with self._lock:
            self._verify_root()
            if destination.exists():
                existing = self.validate(manifest.task_id)
                if _artifact_identity(existing) != _artifact_identity(manifest):
                    raise RuntimeError(
                        "shadow task already has different artifacts."
                    )
                return existing
            staging = Path(
                tempfile.mkdtemp(
                    prefix=f".{manifest.task_id}.",
                    dir=self.root,
                )
            )
            try:
                vector_path = staging / "vectors.json"
                sparse_path = staging / "sparse.json"
                manifest_path = staging / "manifest.json"
                for path, payload in (
                    (vector_path, vector_payload),
                    (sparse_path, sparse_payload),
                    (manifest_path, self._manifest_payload(manifest)),
                ):
                    with path.open("xb") as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                _fsync_directory(staging)
                os.replace(staging, destination)
                _fsync_directory(self.root)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        return self.validate(manifest.task_id)

    def validate(self, task_id: str) -> ShadowArtifactManifest:
        with self._lock:
            self._verify_root()
            directory = self._task_directory(task_id)
            info = directory.lstat()
            if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("shadow task directory is invalid.")
            raw = _strict_loads(
                self._read_bounded(directory / "manifest.json"),
                "shadow manifest",
            )
            if not isinstance(raw, dict):
                raise RuntimeError("shadow manifest must be an object.")
            try:
                manifest = ShadowArtifactManifest(**raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("shadow manifest is invalid.") from exc
            if manifest.task_id != identifier(task_id, "task_id", 64):
                raise RuntimeError(
                    "shadow manifest task identity does not match its directory."
                )
            vector_payload = self._read_bounded(directory / "vectors.json")
            sparse_payload = self._read_bounded(directory / "sparse.json")
            if (
                len(vector_payload) != manifest.vector_bytes
                or _sha256(vector_payload) != manifest.vector_sha256
            ):
                raise RuntimeError("shadow vector artifact digest does not match.")
            if (
                len(sparse_payload) != manifest.sparse_bytes
                or _sha256(sparse_payload) != manifest.sparse_sha256
            ):
                raise RuntimeError("shadow sparse artifact digest does not match.")
            vectors = _strict_loads(vector_payload, "shadow vector artifact")
            sparse = _strict_loads(sparse_payload, "shadow sparse artifact")
            if not isinstance(vectors, list) or len(vectors) != manifest.vector_count:
                raise RuntimeError("shadow vector count does not match.")
            if not isinstance(sparse, list) or len(sparse) != manifest.sparse_count:
                raise RuntimeError("shadow sparse count does not match.")
            return manifest

    def remove(self, task_id: str) -> bool:
        with self._lock:
            self._verify_root()
            directory = self._task_directory(task_id)
            try:
                info = directory.lstat()
            except FileNotFoundError:
                return False
            if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("shadow task directory is invalid.")
            shutil.rmtree(directory)
            _fsync_directory(self.root)
            return True


__all__ = [
    "MigrationShadowStore",
    "ShadowArtifactManifest",
    "ShadowBuild",
]
