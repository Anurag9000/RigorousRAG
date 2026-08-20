"""Fail-closed safetensor supervision cache for authoritative advanced RAG workflows.

The reader supports two immutable authority generations. Historical v1 caches are sealed by
scanning their complete key map into memory, preserving compatibility. Canonical v2 caches carry
``authority.json`` + ``authority.sqlite`` and are verified/read through that disk-backed index,
so training configuration parsing does not materialize corpus-sized key dictionaries.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_supervision import SafetensorSupervisionCache, SupervisionCacheIdentity

_MAX_ENTRY_BYTES = 512 * 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_AUTHORITY_BYTES = 16 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


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


class AuthoritativeSafetensorSupervisionCache(SafetensorSupervisionCache):
    """Writable v1 cache or read-only disk-backed v2 cache authority."""

    def __init__(self, root: str | Path, identity: SupervisionCacheIdentity) -> None:
        safe = safe_advanced_path(root, label="supervision cache root", must_exist=False)
        if safe.exists() and (safe.is_symlink() or not safe.is_dir()):
            raise ValueError("supervision cache root must be a non-symlink directory when it exists")
        self._sealed_contract_sha256: str | None = None
        self._sealed_entries: dict[str, tuple[str, tuple[str, ...]]] | None = None
        self._disk_authority: Mapping[str, Any] | None = None
        super().__init__(safe, identity)
        authority_json = self.root / "authority.json"
        authority_db = self.root / "authority.sqlite"
        if authority_json.exists() or authority_db.exists():
            if not authority_json.exists() or not authority_db.exists():
                raise ValueError("disk-backed supervision cache has incomplete authority files")
            self._disk_authority = self._load_disk_authority()
            self.assert_sealed_integrity()
        else:
            self._scan_contract()

    @staticmethod
    def _bounded_regular(path: Path, label: str, maximum: int) -> int:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} must be a regular non-symlink file")
        size = path.stat().st_size
        if size <= 0 or size > maximum:
            raise ValueError(f"{label} exceeds byte safety bound")
        return size

    def _strict_manifest(self, manifest_path: Path) -> Mapping[str, Any]:
        self._bounded_regular(manifest_path, "supervision manifest entry", _MAX_MANIFEST_BYTES)
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8", errors="strict"),
                parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)),
            )
        except Exception as exc:
            raise ValueError("supervision cache manifest is not strict JSON") from exc
        required = {"schema", "key", "identity_sha256", "tensor_sha256", "tensor_names"}
        if not isinstance(manifest, Mapping) or set(manifest) != required:
            raise ValueError("supervision cache manifest has unexpected fields")
        if manifest.get("schema") != "rigorousrag-supervision-cache-entry/v1":
            raise ValueError("unsupported supervision cache entry schema")
        if manifest.get("identity_sha256") != self.identity.digest:
            raise ValueError("supervision cache manifest identity mismatch")
        key = manifest.get("key")
        if not isinstance(key, str) or not key.strip() or len(key) > 1000 or any(ord(ch) < 32 or ord(ch) == 127 for ch in key):
            raise ValueError("supervision cache manifest key is invalid")
        tensor_sha = _sha(manifest.get("tensor_sha256"), "tensor_sha256")
        names = manifest.get("tensor_names")
        if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names) or names != sorted(set(names)):
            raise ValueError("supervision cache tensor_names must be a sorted unique string list")
        return {"schema": manifest["schema"], "key": key, "identity_sha256": self.identity.digest, "tensor_sha256": tensor_sha, "tensor_names": names}

    def _load_disk_authority(self) -> Mapping[str, Any]:
        authority_json = self.root / "authority.json"; authority_db = self.root / "authority.sqlite"
        self._bounded_regular(authority_json, "supervision cache authority JSON", _MAX_AUTHORITY_BYTES)
        self._bounded_regular(authority_db, "supervision cache authority SQLite", 1 << 50)
        try:
            raw = json.loads(authority_json.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        except Exception as exc:
            raise ValueError("supervision cache authority JSON is invalid") from exc
        required = {"schema", "identity_sha256", "entry_count", "entry_digest_sha256", "contract_sha256", "index_sha256"}
        if not isinstance(raw, Mapping) or set(raw) != required or raw.get("schema") != "rigorousrag-disk-backed-supervision-cache-authority/v2":
            raise ValueError("unsupported disk-backed supervision-cache authority schema")
        if raw.get("identity_sha256") != self.identity.digest:
            raise ValueError("disk-backed supervision cache identity mismatch")
        count = raw.get("entry_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("disk-backed supervision cache entry_count is invalid")
        entry_digest = _sha(raw.get("entry_digest_sha256"), "entry_digest_sha256")
        contract = _sha(raw.get("contract_sha256"), "contract_sha256")
        index_sha = _sha(raw.get("index_sha256"), "index_sha256")
        if _stream_sha(authority_db) != index_sha:
            raise ValueError("disk-backed supervision cache SQLite bytes differ from authority")
        expected_contract = hashlib.sha256(_canonical({"schema": "rigorousrag-disk-backed-supervision-cache-contract/v2", "identity_sha256": self.identity.digest, "entry_count": count, "entry_digest_sha256": entry_digest})).hexdigest()
        if expected_contract != contract:
            raise ValueError("disk-backed supervision cache contract digest mismatch")
        return {"entry_count": count, "entry_digest_sha256": entry_digest, "contract_sha256": contract, "index_sha256": index_sha}

    def _open_disk_index(self) -> sqlite3.Connection:
        if self._disk_authority is None:
            raise ValueError("supervision cache has no disk-backed authority")
        database = self.root / "authority.sqlite"
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _disk_row(self, key: str) -> sqlite3.Row | None:
        with self._open_disk_index() as connection:
            return connection.execute("SELECT key,key_sha256,tensor_sha256,tensor_names_json FROM entries WHERE key=?", (key,)).fetchone()

    def _verify_disk_entry(self, key: str, row: sqlite3.Row | None = None) -> Mapping[str, Any] | None:
        if row is None:
            row = self._disk_row(key)
        tensor_path, manifest_path = self._paths(key)
        if row is None:
            if tensor_path.exists() or manifest_path.exists():
                raise ValueError(f"supervision cache key {key!r} exists outside sealed disk authority")
            return None
        if row["key_sha256"] != hashlib.sha256(key.encode("utf-8")).hexdigest():
            raise ValueError("disk-backed supervision cache key digest mismatch")
        if not tensor_path.exists() or not manifest_path.exists():
            raise ValueError(f"disk-backed supervision cache lost required key {key!r}")
        manifest = self._strict_manifest(manifest_path)
        if manifest["key"] != key:
            raise ValueError("disk-backed supervision cache manifest key mismatch")
        self._bounded_regular(tensor_path, "supervision tensor entry", _MAX_ENTRY_BYTES)
        actual_sha = _stream_sha(tensor_path)
        names = json.loads(row["tensor_names_json"])
        if actual_sha != row["tensor_sha256"] or actual_sha != manifest["tensor_sha256"] or names != manifest["tensor_names"]:
            raise ValueError(f"disk-backed supervision cache entry {key!r} differs from sealed authority")
        return manifest

    def _scan_contract(self) -> tuple[str, dict[str, tuple[str, tuple[str, ...]]]]:
        manifest_paths: dict[str, Path] = {}; tensor_paths: dict[str, Path] = {}
        for path in self.root.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError("supervision cache root may contain only regular non-symlink files")
            suffix = path.suffix; stem = path.stem.lower()
            if suffix not in {".json", ".safetensors"} or len(stem) != 64 or any(ch not in _HEX for ch in stem):
                raise ValueError(f"supervision cache contains unknown file {path.name!r}")
            target = manifest_paths if suffix == ".json" else tensor_paths
            if stem in target: raise ValueError("supervision cache contains duplicate entry filenames")
            target[stem] = path
        if set(manifest_paths) != set(tensor_paths):
            raise ValueError("supervision cache contains orphan tensor/manifest entries")
        entries = []; descriptors: dict[str, tuple[str, tuple[str, ...]]] = {}
        for stem in sorted(manifest_paths):
            manifest = self._strict_manifest(manifest_paths[stem]); key = str(manifest["key"])
            if hashlib.sha256(key.encode("utf-8")).hexdigest() != stem: raise ValueError("supervision cache filename does not match manifest key digest")
            self._bounded_regular(tensor_paths[stem], "supervision tensor entry", _MAX_ENTRY_BYTES)
            actual_sha = _stream_sha(tensor_paths[stem])
            if actual_sha != manifest["tensor_sha256"]: raise ValueError("supervision cache tensor digest mismatch during sealing")
            names = tuple(manifest["tensor_names"])
            if key in descriptors: raise ValueError("supervision cache contains duplicate logical keys")
            descriptors[key] = (actual_sha, names); entries.append({"key": key, "key_sha256": stem, "tensor_sha256": actual_sha, "tensor_names": list(names)})
        payload = {"schema": "rigorousrag-authoritative-supervision-cache-contract/v1", "identity_sha256": self.identity.digest, "entry_count": len(entries), "entries": entries}
        return hashlib.sha256(_canonical(payload)).hexdigest(), descriptors

    @property
    def is_sealed(self) -> bool:
        return self._disk_authority is not None or self._sealed_contract_sha256 is not None

    def seal(self) -> str:
        if self._disk_authority is not None:
            return self.assert_sealed_integrity()
        contract, descriptors = self._scan_contract()
        if self._sealed_contract_sha256 is not None:
            if contract != self._sealed_contract_sha256 or descriptors != self._sealed_entries: raise ValueError("sealed supervision cache contents changed")
            return self._sealed_contract_sha256
        self._sealed_contract_sha256 = contract; self._sealed_entries = descriptors
        return contract

    def assert_sealed_integrity(self) -> str:
        if self._disk_authority is not None:
            authority = self._load_disk_authority()
            if dict(authority) != dict(self._disk_authority): raise ValueError("disk-backed supervision cache authority changed")
            expected_count = int(authority["entry_count"]); file_count = 0; digest = hashlib.sha256(); count = 0
            for path in self.root.iterdir():
                if path.name in {"authority.json", "authority.sqlite"}: continue
                if path.is_symlink() or not path.is_file() or path.suffix not in {".json", ".safetensors"}: raise ValueError("disk-backed supervision cache contains unexpected filesystem entry")
                file_count += 1
            if file_count != expected_count * 2: raise ValueError("disk-backed supervision cache entry-file count changed")
            with self._open_disk_index() as connection:
                for row in connection.execute("SELECT key,key_sha256,tensor_sha256,tensor_names_json FROM entries ORDER BY key"):
                    key = str(row["key"]); manifest = self._verify_disk_entry(key, row); names = json.loads(row["tensor_names_json"])
                    digest.update(_canonical({"key": key, "key_sha256": row["key_sha256"], "tensor_sha256": row["tensor_sha256"], "tensor_names": names}) + b"\n"); count += 1
            if count != expected_count or digest.hexdigest() != authority["entry_digest_sha256"]: raise ValueError("disk-backed supervision cache content digest changed")
            return str(authority["contract_sha256"])
        if self._sealed_contract_sha256 is None or self._sealed_entries is None: raise ValueError("supervision cache is not sealed")
        contract, descriptors = self._scan_contract()
        if contract != self._sealed_contract_sha256 or descriptors != self._sealed_entries: raise ValueError("sealed supervision cache content contract changed")
        return contract

    def put(self, key: str, tensors: Mapping[str, Any]) -> str:
        if self.is_sealed: raise ValueError("sealed supervision cache is read-only")
        return super().put(key, tensors)

    def _verify_expected_entry(self, key: str) -> Mapping[str, Any] | None:
        if self._disk_authority is not None:
            return self._verify_disk_entry(key)
        tensor_path, manifest_path = self._paths(key); tensor_exists, manifest_exists = tensor_path.exists(), manifest_path.exists(); expected = None if self._sealed_entries is None else self._sealed_entries.get(key)
        if not tensor_exists and not manifest_exists:
            if expected is not None: raise ValueError(f"sealed supervision cache lost required key {key!r}")
            return None
        if tensor_exists != manifest_exists: raise ValueError(f"supervision cache key {key!r} has an orphan tensor/manifest entry")
        self._bounded_regular(tensor_path, "supervision tensor entry", _MAX_ENTRY_BYTES); manifest = self._strict_manifest(manifest_path)
        if manifest.get("key") != key: raise ValueError("supervision cache manifest key mismatch")
        actual_sha = _stream_sha(tensor_path)
        if actual_sha != manifest.get("tensor_sha256"): raise ValueError("supervision cache tensor digest mismatch")
        if self._sealed_entries is not None:
            if expected is None: raise ValueError(f"key {key!r} was added after supervision cache sealing")
            expected_sha, expected_names = expected
            if actual_sha != expected_sha or tuple(manifest["tensor_names"]) != expected_names: raise ValueError(f"sealed supervision cache key {key!r} changed after sealing")
        return manifest

    def contains(self, key: str) -> bool:
        return self._verify_expected_entry(key) is not None

    def get(self, key: str) -> Mapping[str, Any]:
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("safetensors is required for supervision cache reads") from exc
        manifest = self._verify_expected_entry(key)
        if manifest is None: raise KeyError(f"supervision cache lacks key {key!r}")
        tensor_path, _ = self._paths(key); tensors = load_file(str(tensor_path), device="cpu")
        if sorted(tensors) != list(manifest["tensor_names"]): raise ValueError("supervision cache tensor names differ from manifest")
        return tensors

    @property
    def contract_sha256(self) -> str:
        if self._disk_authority is not None: return str(self._disk_authority["contract_sha256"])
        if self._sealed_contract_sha256 is not None: return self._sealed_contract_sha256
        contract, _ = self._scan_contract(); return contract


__all__ = ["AuthoritativeSafetensorSupervisionCache"]
