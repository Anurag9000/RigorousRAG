"""Corpus-scale, disk-backed sealed safetensor supervision-cache authority.

The older authoritative cache keeps the entire sealed key map in Python memory. This variant is
for canonical large-corpus training data: entry identity lives in a read-only SQLite index,
contract computation streams sorted rows, writes hash tensors without reading them wholesale,
and every requested read rechecks the exact manifest/tensor bytes against the sealed index.
A canonical parent receipt should additionally bind ``contract_sha256`` and the two authority
file SHAs exposed by ``authority_file_sha256s``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Mapping

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_supervision import SafetensorSupervisionCache, SupervisionCacheIdentity

_MAX_ENTRY_BYTES = 512 * 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_AUTHORITY_BYTES = 16 * 1024 * 1024
_AUTHORITY_JSON = "authority.json"
_AUTHORITY_DB = "authority.sqlite"
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _atomic(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _bounded_regular(path: Path, label: str, maximum: int) -> int:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > maximum:
        raise ValueError(f"{label} exceeds byte safety bound")
    return size


def _strict_entry_manifest(path: Path, identity_sha256: str) -> Mapping[str, Any]:
    _bounded_regular(path, "supervision cache manifest", _MAX_MANIFEST_BYTES)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError("supervision cache manifest is not strict JSON") from exc
    required = {"schema", "key", "identity_sha256", "tensor_sha256", "tensor_names"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("supervision cache manifest has unexpected fields")
    if value.get("schema") != "rigorousrag-supervision-cache-entry/v1":
        raise ValueError("unsupported supervision cache entry schema")
    if value.get("identity_sha256") != identity_sha256:
        raise ValueError("supervision cache manifest identity mismatch")
    key = value.get("key")
    if (
        not isinstance(key, str)
        or not key.strip()
        or len(key) > 1000
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in key)
    ):
        raise ValueError("supervision cache manifest key is invalid")
    tensor_sha = _sha(value.get("tensor_sha256"), "tensor_sha256")
    names = value.get("tensor_names")
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) or not name for name in names)
        or names != sorted(set(names))
    ):
        raise ValueError("supervision cache tensor_names must be sorted and unique")
    return {
        "schema": value["schema"],
        "key": key,
        "identity_sha256": identity_sha256,
        "tensor_sha256": tensor_sha,
        "tensor_names": names,
    }


class DiskBackedAuthoritativeSafetensorCache(SafetensorSupervisionCache):
    """Safetensor cache with a disk-backed immutable sealed key authority."""

    def __init__(self, root: str | Path, identity: SupervisionCacheIdentity) -> None:
        if not isinstance(identity, SupervisionCacheIdentity):
            raise ValueError("identity must be SupervisionCacheIdentity")
        selected = safe_advanced_path(root, label="disk-backed supervision cache", must_exist=False)
        if selected.exists() and (selected.is_symlink() or not selected.is_dir()):
            raise ValueError("disk-backed supervision cache root must be a non-symlink directory")
        super().__init__(selected, identity)
        self._authority: Mapping[str, Any] | None = None
        authority_json = self.root / _AUTHORITY_JSON
        authority_db = self.root / _AUTHORITY_DB
        if authority_json.exists() or authority_db.exists():
            if not authority_json.exists() or not authority_db.exists():
                raise ValueError("sealed supervision cache has incomplete authority files")
            self._authority = self._load_authority()

    @property
    def is_sealed(self) -> bool:
        return self._authority is not None

    def _authority_paths(self) -> tuple[Path, Path]:
        return self.root / _AUTHORITY_JSON, self.root / _AUTHORITY_DB

    def _load_authority(self) -> Mapping[str, Any]:
        authority_json, authority_db = self._authority_paths()
        _bounded_regular(authority_json, "supervision cache authority JSON", _MAX_AUTHORITY_BYTES)
        _bounded_regular(authority_db, "supervision cache authority SQLite", 1 << 50)
        try:
            raw = json.loads(
                authority_json.read_text(encoding="utf-8", errors="strict"),
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
        except Exception as exc:
            raise ValueError("supervision cache authority JSON is invalid") from exc
        required = {
            "schema",
            "identity_sha256",
            "entry_count",
            "entry_digest_sha256",
            "contract_sha256",
            "index_sha256",
        }
        if (
            not isinstance(raw, Mapping)
            or set(raw) != required
            or raw.get("schema") != "rigorousrag-disk-backed-supervision-cache-authority/v2"
        ):
            raise ValueError("unsupported disk-backed supervision-cache authority schema")
        if raw.get("identity_sha256") != self.identity.digest:
            raise ValueError("supervision cache authority identity mismatch")
        count = raw.get("entry_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("supervision cache authority entry_count is invalid")
        entry_digest = _sha(raw.get("entry_digest_sha256"), "entry_digest_sha256")
        contract = _sha(raw.get("contract_sha256"), "contract_sha256")
        index_sha = _sha(raw.get("index_sha256"), "index_sha256")
        if _stream_sha(authority_db) != index_sha:
            raise ValueError("supervision cache authority SQLite bytes differ from authority JSON")
        expected_contract = _digest(
            {
                "schema": "rigorousrag-disk-backed-supervision-cache-contract/v2",
                "identity_sha256": self.identity.digest,
                "entry_count": count,
                "entry_digest_sha256": entry_digest,
            }
        )
        if expected_contract != contract:
            raise ValueError("supervision cache authority contract digest mismatch")
        return {
            "schema": raw["schema"],
            "identity_sha256": self.identity.digest,
            "entry_count": count,
            "entry_digest_sha256": entry_digest,
            "contract_sha256": contract,
            "index_sha256": index_sha,
        }

    def _read_manifest_for_key(self, key: str) -> Mapping[str, Any] | None:
        tensor_path, manifest_path = self._paths(key)
        tensor_exists = tensor_path.exists()
        manifest_exists = manifest_path.exists()
        if not tensor_exists and not manifest_exists:
            return None
        if tensor_exists != manifest_exists:
            raise ValueError(f"supervision cache key {key!r} has an orphan tensor/manifest")
        manifest = _strict_entry_manifest(manifest_path, self.identity.digest)
        if manifest["key"] != key:
            raise ValueError("supervision cache manifest key mismatch")
        expected_stem = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if tensor_path.stem != expected_stem or manifest_path.stem != expected_stem:
            raise ValueError("supervision cache entry filename does not match key")
        _bounded_regular(tensor_path, "supervision cache tensor", _MAX_ENTRY_BYTES)
        actual = _stream_sha(tensor_path)
        if actual != manifest["tensor_sha256"]:
            raise ValueError("supervision cache tensor digest mismatch")
        return manifest

    def put(self, key: str, tensors: Mapping[str, Any]) -> str:
        if self.is_sealed:
            raise ValueError("sealed disk-backed supervision cache is read-only")
        try:
            import torch
            from safetensors.torch import save_file
        except Exception as exc:
            raise RuntimeError("PyTorch and safetensors are required for supervision cache writes") from exc
        if not tensors:
            raise ValueError("cache tensor mapping may not be empty")
        normalized: dict[str, Any] = {}
        for raw_name, tensor in tensors.items():
            name = str(raw_name).strip()
            if (
                not name
                or len(name) > 300
                or any(ord(ch) < 32 or ord(ch) == 127 for ch in name)
            ):
                raise ValueError("tensor name is invalid")
            if not torch.is_tensor(tensor):
                raise ValueError("supervision cache values must be tensors")
            normalized[name] = tensor.detach().cpu().contiguous()
        tensor_path, manifest_path = self._paths(key)
        if tensor_path.exists() or manifest_path.exists():
            raise ValueError(f"supervision cache key {key!r} already exists")
        with tempfile.NamedTemporaryFile(
            prefix=".rag-cache-",
            suffix=".safetensors",
            dir=self.root,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            save_file(normalized, str(temporary))
            tensor_sha = _stream_sha(temporary)
            os.replace(temporary, tensor_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        manifest = {
            "schema": "rigorousrag-supervision-cache-entry/v1",
            "key": key,
            "identity_sha256": self.identity.digest,
            "tensor_sha256": tensor_sha,
            "tensor_names": sorted(normalized),
        }
        try:
            _atomic(manifest_path, _canonical(manifest) + b"\n")
        except Exception:
            tensor_path.unlink(missing_ok=True)
            raise
        return tensor_sha

    def _open_index(self) -> sqlite3.Connection:
        if not self.is_sealed:
            raise ValueError("supervision cache is not sealed")
        _, authority_db = self._authority_paths()
        connection = sqlite3.connect(f"file:{authority_db}?mode=ro", uri=True, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _indexed_row(self, key: str) -> sqlite3.Row | None:
        with self._open_index() as connection:
            return connection.execute(
                "SELECT key,key_sha256,tensor_sha256,tensor_names_json FROM entries WHERE key=?",
                (key,),
            ).fetchone()

    def contains(self, key: str) -> bool:
        if self.is_sealed:
            row = self._indexed_row(key)
            if row is None:
                tensor_path, manifest_path = self._paths(key)
                if tensor_path.exists() or manifest_path.exists():
                    raise ValueError(f"cache key {key!r} exists outside the sealed authority")
                return False
            manifest = self._read_manifest_for_key(key)
            if manifest is None:
                raise ValueError(f"sealed cache lost required key {key!r}")
            if (
                row["key_sha256"] != hashlib.sha256(key.encode("utf-8")).hexdigest()
                or row["tensor_sha256"] != manifest["tensor_sha256"]
                or json.loads(row["tensor_names_json"]) != manifest["tensor_names"]
            ):
                raise ValueError(f"sealed cache authority differs for key {key!r}")
            return True
        return self._read_manifest_for_key(key) is not None

    def get(self, key: str) -> Mapping[str, Any]:
        if self.is_sealed and not self.contains(key):
            raise KeyError(f"supervision cache lacks key {key!r}")
        manifest = self._read_manifest_for_key(key)
        if manifest is None:
            raise KeyError(f"supervision cache lacks key {key!r}")
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("safetensors is required for supervision cache reads") from exc
        tensor_path, _ = self._paths(key)
        tensors = load_file(str(tensor_path), device="cpu")
        if sorted(tensors) != list(manifest["tensor_names"]):
            raise ValueError("supervision cache tensor names differ from manifest")
        return tensors

    def _build_index(self) -> Mapping[str, Any]:
        authority_json, authority_db = self._authority_paths()
        if authority_json.exists() or authority_db.exists():
            raise ValueError("supervision cache authority files already exist")
        descriptor, temp_name = tempfile.mkstemp(
            prefix=".rag-cache-index-",
            suffix=".sqlite",
            dir=self.root.parent,
        )
        os.close(descriptor)
        temp_db = Path(temp_name)
        temp_db.unlink(missing_ok=True)
        connection = sqlite3.connect(str(temp_db), timeout=30.0)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """CREATE TABLE files (
                    stem TEXT PRIMARY KEY,
                    has_manifest INTEGER NOT NULL DEFAULT 0,
                    has_tensor INTEGER NOT NULL DEFAULT 0
                ) WITHOUT ROWID"""
            )
            connection.execute(
                """CREATE TABLE entries (
                    key TEXT PRIMARY KEY,
                    key_sha256 TEXT NOT NULL UNIQUE,
                    tensor_sha256 TEXT NOT NULL,
                    tensor_names_json TEXT NOT NULL
                ) WITHOUT ROWID"""
            )
            for path in self.root.iterdir():
                if path.name in {_AUTHORITY_JSON, _AUTHORITY_DB}:
                    raise ValueError("unexpected pre-existing cache authority file")
                if path.is_symlink() or not path.is_file():
                    raise ValueError("supervision cache root may contain only regular files")
                if path.suffix not in {".json", ".safetensors"}:
                    raise ValueError(f"supervision cache contains unknown file {path.name!r}")
                stem = path.stem.lower()
                if len(stem) != 64 or any(ch not in _HEX for ch in stem):
                    raise ValueError("supervision cache entry filename is not a key SHA-256")
                connection.execute(
                    "INSERT OR IGNORE INTO files(stem) VALUES(?)",
                    (stem,),
                )
                column = "has_manifest" if path.suffix == ".json" else "has_tensor"
                previous = connection.execute(
                    f"SELECT {column} FROM files WHERE stem=?",
                    (stem,),
                ).fetchone()
                if previous is None or int(previous[0]) != 0:
                    raise ValueError("supervision cache contains duplicate entry filenames")
                connection.execute(
                    f"UPDATE files SET {column}=1 WHERE stem=?",
                    (stem,),
                )
            orphan = connection.execute(
                "SELECT stem FROM files WHERE has_manifest!=1 OR has_tensor!=1 LIMIT 1"
            ).fetchone()
            if orphan is not None:
                raise ValueError(f"supervision cache contains orphan entry {orphan[0]!r}")

            stems = connection.execute("SELECT stem FROM files ORDER BY stem")
            for row in stems:
                stem = str(row[0])
                manifest_path = self.root / f"{stem}.json"
                tensor_path = self.root / f"{stem}.safetensors"
                manifest = _strict_entry_manifest(manifest_path, self.identity.digest)
                key = str(manifest["key"])
                if hashlib.sha256(key.encode("utf-8")).hexdigest() != stem:
                    raise ValueError("supervision cache filename does not match logical key")
                _bounded_regular(tensor_path, "supervision cache tensor", _MAX_ENTRY_BYTES)
                actual_sha = _stream_sha(tensor_path)
                if actual_sha != manifest["tensor_sha256"]:
                    raise ValueError("supervision cache tensor digest mismatch while sealing")
                connection.execute(
                    "INSERT INTO entries(key,key_sha256,tensor_sha256,tensor_names_json) VALUES(?,?,?,?)",
                    (
                        key,
                        stem,
                        actual_sha,
                        json.dumps(manifest["tensor_names"], separators=(",", ":")),
                    ),
                )
            connection.execute("DROP TABLE files")
            connection.commit()

            entry_digest = hashlib.sha256()
            count = 0
            rows = connection.execute(
                "SELECT key,key_sha256,tensor_sha256,tensor_names_json FROM entries ORDER BY key"
            )
            for row in rows:
                names = json.loads(row[3])
                entry_digest.update(
                    _canonical(
                        {
                            "key": row[0],
                            "key_sha256": row[1],
                            "tensor_sha256": row[2],
                            "tensor_names": names,
                        }
                    )
                    + b"\n"
                )
                count += 1
            digest = entry_digest.hexdigest()
            contract = _digest(
                {
                    "schema": "rigorousrag-disk-backed-supervision-cache-contract/v2",
                    "identity_sha256": self.identity.digest,
                    "entry_count": count,
                    "entry_digest_sha256": digest,
                }
            )
        finally:
            connection.close()

        os.replace(temp_db, authority_db)
        index_sha = _stream_sha(authority_db)
        authority = {
            "schema": "rigorousrag-disk-backed-supervision-cache-authority/v2",
            "identity_sha256": self.identity.digest,
            "entry_count": count,
            "entry_digest_sha256": digest,
            "contract_sha256": contract,
            "index_sha256": index_sha,
        }
        try:
            _atomic(authority_json, _canonical(authority) + b"\n")
        except Exception:
            authority_db.unlink(missing_ok=True)
            raise
        return authority

    def seal(self) -> str:
        if self._authority is None:
            self._authority = self._build_index()
        self.assert_sealed_integrity()
        return str(self._authority["contract_sha256"])

    def assert_sealed_integrity(self) -> str:
        if self._authority is None:
            raise ValueError("supervision cache is not sealed")
        current = self._load_authority()
        if dict(current) != dict(self._authority):
            raise ValueError("supervision cache authority files changed after sealing")
        expected_count = int(current["entry_count"])
        file_count = 0
        for path in self.root.iterdir():
            if path.name in {_AUTHORITY_JSON, _AUTHORITY_DB}:
                continue
            if path.is_symlink() or not path.is_file() or path.suffix not in {".json", ".safetensors"}:
                raise ValueError("sealed supervision cache contains an unexpected filesystem entry")
            file_count += 1
        if file_count != expected_count * 2:
            raise ValueError("sealed supervision cache entry-file count changed")

        digest = hashlib.sha256()
        count = 0
        with self._open_index() as connection:
            rows = connection.execute(
                "SELECT key,key_sha256,tensor_sha256,tensor_names_json FROM entries ORDER BY key"
            )
            for row in rows:
                key = str(row[0])
                manifest = self._read_manifest_for_key(key)
                if manifest is None:
                    raise ValueError(f"sealed supervision cache lost key {key!r}")
                names = json.loads(row[3])
                if row[1] != hashlib.sha256(key.encode("utf-8")).hexdigest():
                    raise ValueError("sealed supervision cache key digest changed")
                if row[2] != manifest["tensor_sha256"] or names != manifest["tensor_names"]:
                    raise ValueError("sealed supervision cache indexed entry changed")
                digest.update(
                    _canonical(
                        {
                            "key": key,
                            "key_sha256": row[1],
                            "tensor_sha256": row[2],
                            "tensor_names": names,
                        }
                    )
                    + b"\n"
                )
                count += 1
        if count != expected_count or digest.hexdigest() != current["entry_digest_sha256"]:
            raise ValueError("sealed supervision cache content digest changed")
        return str(current["contract_sha256"])

    @property
    def contract_sha256(self) -> str:
        if self._authority is None:
            raise ValueError("disk-backed supervision cache must be sealed before authority use")
        return str(self._authority["contract_sha256"])

    @property
    def entry_count(self) -> int:
        if self._authority is None:
            raise ValueError("disk-backed supervision cache must be sealed before authority use")
        return int(self._authority["entry_count"])

    @property
    def authority_file_sha256s(self) -> tuple[str, str]:
        if self._authority is None:
            raise ValueError("disk-backed supervision cache must be sealed before authority use")
        authority_json, authority_db = self._authority_paths()
        return _stream_sha(authority_json), _stream_sha(authority_db)


__all__ = ["DiskBackedAuthoritativeSafetensorCache"]
