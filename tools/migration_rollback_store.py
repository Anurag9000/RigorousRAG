"""AES-GCM encrypted durable rollback artifact store."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import stat
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from tools.config import bounded_int_env
from tools.migration_cutover_preflight import CutoverPreflight
from tools.migration_rollback_artifact import (
    EncryptedRollbackManifest,
    RollbackEncryptionKey,
    canonical_json_bytes,
    strict_json_loads,
    validate_rollback_payload,
)
from tools.migration_types import digest, identifier

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_MANIFEST_BYTES = 2_000_000
_MAX_PLAINTEXT_BYTES = bounded_int_env(
    "MIGRATION_ROLLBACK_MAX_BYTES",
    536_870_912,
    minimum=1_000_000,
    maximum=1_000_000_000,
)


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _root(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("rollback artifact root must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("rollback artifact root is invalid.")
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
            raise ValueError("rollback artifact root could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("rollback artifact root may not contain redirects.")
    return absolute


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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _aad(preflight: CutoverPreflight, key_id: str) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": 1,
            "algorithm": "AES-256-GCM",
            "key_id": identifier(key_id, "key_id", 128),
            "task_id": preflight.task_id,
            "owner_id": preflight.owner_id,
            "doc_id": preflight.doc_id,
            "preflight_digest": preflight.preflight_digest,
            "rollback_identity_digest": preflight.rollback_identity_digest,
            "source_sequence": preflight.source_sequence,
            "source_profile_fingerprint": preflight.source_profile_fingerprint,
            "source_content_sha256": preflight.source_content_sha256,
            "vector_snapshot_digest": preflight.vector_snapshot_digest,
            "sparse_snapshot_digest": preflight.sparse_snapshot_digest,
        },
        maximum=_MAX_MANIFEST_BYTES,
    )


class MigrationRollbackStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = _root(root)
        self.root.mkdir(parents=True, exist_ok=True)
        info = self.root.lstat()
        if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("rollback artifact root must be a regular directory.")
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass
        self._identity = (int(info.st_dev), int(info.st_ino))
        self._lock = threading.RLock()

    def _verify(self) -> None:
        info = self.root.lstat()
        if (
            _redirecting(info)
            or not stat.S_ISDIR(info.st_mode)
            or (int(info.st_dev), int(info.st_ino)) != self._identity
        ):
            raise RuntimeError("rollback artifact root identity changed.")

    def _directory(self, task_id: str, preflight_digest: str) -> Path:
        return (
            self.root
            / identifier(task_id, "task_id", 64)
            / digest(preflight_digest, "preflight_digest")
        )

    @staticmethod
    def _read(path: Path, maximum: int) -> bytes:
        info = path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("rollback artifact member is not a regular file.")
        if info.st_size <= 0 or info.st_size > maximum:
            raise RuntimeError("rollback artifact member exceeds its size limit.")
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                raise RuntimeError("rollback artifact member changed before reading.")
            payload = handle.read(maximum + 1)
        if len(payload) > maximum:
            raise RuntimeError("rollback artifact member exceeds its size limit.")
        after = path.lstat()
        if (
            _redirecting(after)
            or (after.st_dev, after.st_ino) != (info.st_dev, info.st_ino)
            or after.st_size != info.st_size
        ):
            raise RuntimeError("rollback artifact member changed during reading.")
        return payload

    @staticmethod
    def _manifest(payload: bytes) -> EncryptedRollbackManifest:
        raw = strict_json_loads(payload, "rollback manifest")
        if not isinstance(raw, dict):
            raise RuntimeError("rollback manifest must be an object.")
        try:
            return EncryptedRollbackManifest(**raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("rollback manifest schema is invalid.") from exc

    @staticmethod
    def _write_member(path: Path, payload: bytes) -> None:
        with path.open("xb") as handle:
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def _read_manifest_path(self, directory: Path) -> EncryptedRollbackManifest:
        return self._manifest(
            self._read(directory / "manifest.json", _MAX_MANIFEST_BYTES)
        )

    def write(
        self,
        *,
        preflight: CutoverPreflight,
        payload: Mapping[str, Any],
        key: RollbackEncryptionKey,
        now: float | None = None,
    ) -> EncryptedRollbackManifest:
        if not isinstance(preflight, CutoverPreflight):
            raise ValueError("preflight must be CutoverPreflight.")
        if not isinstance(key, RollbackEncryptionKey):
            raise ValueError("key must be RollbackEncryptionKey.")
        normalized = validate_rollback_payload(preflight, payload)
        plaintext = canonical_json_bytes(
            normalized,
            maximum=_MAX_PLAINTEXT_BYTES,
        )
        destination = self._directory(preflight.task_id, preflight.preflight_digest)
        with self._lock:
            self._verify()
            if destination.exists():
                _payload, existing = self.load(
                    preflight=preflight,
                    key=key,
                )
                return existing
            nonce = os.urandom(12)
            aad = _aad(preflight, key.key_id)
            ciphertext = AESGCM(key.key).encrypt(nonce, plaintext, aad)
            manifest = EncryptedRollbackManifest(
                task_id=preflight.task_id,
                owner_id=preflight.owner_id,
                doc_id=preflight.doc_id,
                preflight_digest=preflight.preflight_digest,
                rollback_identity_digest=preflight.rollback_identity_digest,
                source_sequence=preflight.source_sequence,
                source_profile_fingerprint=preflight.source_profile_fingerprint,
                source_content_sha256=preflight.source_content_sha256,
                vector_snapshot_digest=preflight.vector_snapshot_digest,
                sparse_snapshot_digest=preflight.sparse_snapshot_digest,
                plaintext_sha256=_sha256(plaintext),
                ciphertext_sha256=_sha256(ciphertext),
                plaintext_bytes=len(plaintext),
                ciphertext_bytes=len(ciphertext),
                algorithm="AES-256-GCM",
                key_id=key.key_id,
                nonce_b64=base64.b64encode(nonce).decode("ascii"),
                aad_sha256=_sha256(aad),
                created_at=time.time() if now is None else now,
            )
            task_dir = destination.parent
            if task_dir.exists():
                info = task_dir.lstat()
                if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
                    raise RuntimeError("rollback task directory is invalid.")
            else:
                task_dir.mkdir(mode=0o700)
                _fsync_directory(self.root)
            staging = Path(
                tempfile.mkdtemp(prefix=f".{preflight.preflight_digest}.", dir=task_dir)
            )
            try:
                try:
                    os.chmod(staging, 0o700)
                except OSError:
                    pass
                self._write_member(staging / "ciphertext.bin", ciphertext)
                self._write_member(
                    staging / "manifest.json",
                    canonical_json_bytes(
                        asdict(manifest), maximum=_MAX_MANIFEST_BYTES
                    ),
                )
                _fsync_directory(staging)
                os.replace(staging, destination)
                _fsync_directory(task_dir)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        _payload, persisted = self.load(preflight=preflight, key=key)
        return persisted

    def read_manifest(
        self,
        task_id: str,
        preflight_digest: str,
    ) -> EncryptedRollbackManifest:
        with self._lock:
            self._verify()
            directory = self._directory(task_id, preflight_digest)
            info = directory.lstat()
            if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("rollback artifact directory is invalid.")
            manifest = self._read_manifest_path(directory)
            if (
                manifest.task_id != identifier(task_id, "task_id", 64)
                or manifest.preflight_digest
                != digest(preflight_digest, "preflight_digest")
            ):
                raise RuntimeError("rollback artifact manifest escaped requested scope.")
            return manifest

    def load(
        self,
        *,
        preflight: CutoverPreflight,
        key: RollbackEncryptionKey,
    ) -> tuple[dict[str, Any], EncryptedRollbackManifest]:
        if not isinstance(preflight, CutoverPreflight):
            raise ValueError("preflight must be CutoverPreflight.")
        if not isinstance(key, RollbackEncryptionKey):
            raise ValueError("key must be RollbackEncryptionKey.")
        with self._lock:
            self._verify()
            directory = self._directory(preflight.task_id, preflight.preflight_digest)
            manifest = self._read_manifest_path(directory)
            if (
                manifest.task_id != preflight.task_id
                or manifest.owner_id != preflight.owner_id
                or manifest.doc_id != preflight.doc_id
                or manifest.preflight_digest != preflight.preflight_digest
                or manifest.rollback_identity_digest
                != preflight.rollback_identity_digest
                or manifest.source_sequence != preflight.source_sequence
                or manifest.source_profile_fingerprint
                != preflight.source_profile_fingerprint
                or manifest.source_content_sha256 != preflight.source_content_sha256
                or manifest.vector_snapshot_digest
                != preflight.vector_snapshot_digest
                or manifest.sparse_snapshot_digest
                != preflight.sparse_snapshot_digest
            ):
                raise RuntimeError("rollback artifact manifest does not match preflight.")
            if manifest.key_id != key.key_id:
                raise RuntimeError("rollback artifact key ID does not match configured key.")
            ciphertext = self._read(
                directory / "ciphertext.bin",
                _MAX_PLAINTEXT_BYTES + 1024,
            )
            if (
                len(ciphertext) != manifest.ciphertext_bytes
                or _sha256(ciphertext) != manifest.ciphertext_sha256
            ):
                raise RuntimeError("rollback ciphertext digest changed.")
            aad = _aad(preflight, key.key_id)
            if _sha256(aad) != manifest.aad_sha256:
                raise RuntimeError("rollback authenticated metadata digest changed.")
            nonce = base64.b64decode(manifest.nonce_b64, validate=True)
            try:
                plaintext = AESGCM(key.key).decrypt(nonce, ciphertext, aad)
            except InvalidTag as exc:
                raise RuntimeError("rollback artifact authentication failed.") from exc
            if (
                len(plaintext) != manifest.plaintext_bytes
                or _sha256(plaintext) != manifest.plaintext_sha256
            ):
                raise RuntimeError("rollback plaintext digest changed.")
            raw = strict_json_loads(plaintext, "rollback payload")
            if not isinstance(raw, Mapping):
                raise RuntimeError("rollback payload must be an object.")
            normalized = validate_rollback_payload(preflight, raw)
            return normalized, manifest

    def remove(
        self,
        task_id: str,
        preflight_digest: str,
    ) -> bool:
        with self._lock:
            self._verify()
            directory = self._directory(task_id, preflight_digest)
            try:
                info = directory.lstat()
            except FileNotFoundError:
                return False
            if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("rollback artifact directory is invalid.")
            shutil.rmtree(directory)
            task_dir = directory.parent
            try:
                if not any(task_dir.iterdir()):
                    task_dir.rmdir()
            except OSError:
                pass
            _fsync_directory(self.root)
            return True


__all__ = ["MigrationRollbackStore"]
