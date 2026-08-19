"""Fail-closed safetensor supervision cache for authoritative advanced RAG workflows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_supervision import SafetensorSupervisionCache, SupervisionCacheIdentity

_MAX_ENTRY_BYTES = 512 * 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
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


class AuthoritativeSafetensorSupervisionCache(SafetensorSupervisionCache):
    """Writable materialization cache that can transition to a frozen read authority.

    Before ``seal()`` the cache may be populated by explicit canonical materialization code.
    ``seal()`` snapshots the exact key -> tensor SHA/name mapping and immutable cache contract.
    After sealing, writes are rejected and every membership/read verifies the requested entry
    against that frozen snapshot. This prevents an internally consistent file replacement after
    the training input identity was computed from silently changing supervision consumed by the
    run.
    """

    def __init__(self, root: str | Path, identity: SupervisionCacheIdentity) -> None:
        safe = safe_advanced_path(root, label="supervision cache root", must_exist=False)
        if safe.exists() and not safe.is_dir():
            raise ValueError("supervision cache root must be a directory when it exists")
        self._sealed_contract_sha256: str | None = None
        self._sealed_entries: dict[str, tuple[str, tuple[str, ...]]] | None = None
        super().__init__(safe, identity)
        # Existing malformed/orphan caches fail at construction. Empty fresh caches remain
        # valid for the explicit materialization phase.
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
        if not isinstance(key, str) or not key.strip() or any(ord(ch) < 32 or ord(ch) == 127 for ch in key):
            raise ValueError("supervision cache manifest key is invalid")
        tensor_sha = str(manifest.get("tensor_sha256", "")).strip().lower()
        if len(tensor_sha) != 64 or any(ch not in _HEX for ch in tensor_sha):
            raise ValueError("supervision cache manifest tensor_sha256 is invalid")
        names = manifest.get("tensor_names")
        if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names) or names != sorted(set(names)):
            raise ValueError("supervision cache tensor_names must be a sorted unique string list")
        return manifest

    def _scan_contract(self) -> tuple[str, dict[str, tuple[str, tuple[str, ...]]]]:
        manifest_paths: dict[str, Path] = {}
        tensor_paths: dict[str, Path] = {}
        for path in self.root.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError("supervision cache root may contain only regular non-symlink files")
            suffix = path.suffix
            stem = path.stem.lower()
            if suffix not in {".json", ".safetensors"} or len(stem) != 64 or any(ch not in _HEX for ch in stem):
                raise ValueError(f"supervision cache contains unknown file {path.name!r}")
            target = manifest_paths if suffix == ".json" else tensor_paths
            if stem in target:
                raise ValueError("supervision cache contains duplicate entry filenames")
            target[stem] = path
        if set(manifest_paths) != set(tensor_paths):
            missing_tensor = sorted(set(manifest_paths) - set(tensor_paths))[:20]
            missing_manifest = sorted(set(tensor_paths) - set(manifest_paths))[:20]
            raise ValueError(
                f"supervision cache contains orphan entries; missing_tensor={missing_tensor}, missing_manifest={missing_manifest}"
            )
        entries = []
        descriptors: dict[str, tuple[str, tuple[str, ...]]] = {}
        for stem in sorted(manifest_paths):
            manifest_path = manifest_paths[stem]
            tensor_path = tensor_paths[stem]
            manifest = self._strict_manifest(manifest_path)
            key = str(manifest["key"])
            expected_stem = hashlib.sha256(key.encode("utf-8")).hexdigest()
            if expected_stem != stem:
                raise ValueError("supervision cache filename does not match manifest key digest")
            self._bounded_regular(tensor_path, "supervision tensor entry", _MAX_ENTRY_BYTES)
            actual_tensor_sha = _stream_sha(tensor_path)
            if actual_tensor_sha != manifest["tensor_sha256"]:
                raise ValueError("supervision cache tensor digest mismatch during sealing")
            names = tuple(manifest["tensor_names"])
            if key in descriptors:
                raise ValueError("supervision cache contains duplicate logical keys")
            descriptors[key] = (actual_tensor_sha, names)
            entries.append(
                {
                    "key": key,
                    "key_sha256": stem,
                    "tensor_sha256": actual_tensor_sha,
                    "tensor_names": list(names),
                }
            )
        payload = {
            "schema": "rigorousrag-authoritative-supervision-cache-contract/v1",
            "identity_sha256": self.identity.digest,
            "entry_count": len(entries),
            "entries": entries,
        }
        return hashlib.sha256(_canonical(payload)).hexdigest(), descriptors

    @property
    def is_sealed(self) -> bool:
        return self._sealed_contract_sha256 is not None

    def seal(self) -> str:
        """Freeze exact current contents for read-only authoritative consumption."""
        contract, descriptors = self._scan_contract()
        if self._sealed_contract_sha256 is not None:
            if contract != self._sealed_contract_sha256 or descriptors != self._sealed_entries:
                raise ValueError("sealed supervision cache contents changed")
            return self._sealed_contract_sha256
        self._sealed_contract_sha256 = contract
        self._sealed_entries = descriptors
        return contract

    def assert_sealed_integrity(self) -> str:
        if self._sealed_contract_sha256 is None or self._sealed_entries is None:
            raise ValueError("supervision cache is not sealed")
        contract, descriptors = self._scan_contract()
        if contract != self._sealed_contract_sha256 or descriptors != self._sealed_entries:
            raise ValueError("sealed supervision cache content contract changed")
        return contract

    def put(self, key: str, tensors: Mapping[str, Any]) -> str:
        if self.is_sealed:
            raise ValueError("sealed supervision cache is read-only")
        return super().put(key, tensors)

    def _verify_expected_entry(self, key: str) -> Mapping[str, Any] | None:
        tensor_path, manifest_path = self._paths(key)
        tensor_exists = tensor_path.exists()
        manifest_exists = manifest_path.exists()
        expected = None if self._sealed_entries is None else self._sealed_entries.get(key)
        if not tensor_exists and not manifest_exists:
            if expected is not None:
                raise ValueError(f"sealed supervision cache lost required key {key!r}")
            return None
        if tensor_exists != manifest_exists:
            raise ValueError(f"supervision cache key {key!r} has an orphan tensor/manifest entry")
        self._bounded_regular(tensor_path, "supervision tensor entry", _MAX_ENTRY_BYTES)
        manifest = self._strict_manifest(manifest_path)
        if manifest.get("key") != key:
            raise ValueError("supervision cache manifest key mismatch")
        actual_sha = _stream_sha(tensor_path)
        if actual_sha != manifest.get("tensor_sha256"):
            raise ValueError("supervision cache tensor digest mismatch")
        if self._sealed_entries is not None:
            if expected is None:
                raise ValueError(f"key {key!r} was added after supervision cache sealing")
            expected_sha, expected_names = expected
            if actual_sha != expected_sha or tuple(manifest["tensor_names"]) != expected_names:
                raise ValueError(f"sealed supervision cache key {key!r} changed after sealing")
        return manifest

    def contains(self, key: str) -> bool:
        """Return membership only after proving exact current/frozen pair integrity."""
        return self._verify_expected_entry(key) is not None

    def get(self, key: str) -> Mapping[str, Any]:
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("safetensors is required for supervision cache reads") from exc
        manifest = self._verify_expected_entry(key)
        if manifest is None:
            raise KeyError(f"supervision cache lacks key {key!r}")
        tensor_path, _ = self._paths(key)
        tensors = load_file(str(tensor_path), device="cpu")
        if sorted(tensors) != list(manifest["tensor_names"]):
            raise ValueError("supervision cache tensor names differ from manifest")
        return tensors

    @property
    def contract_sha256(self) -> str:
        """Return frozen contract after sealing, otherwise the exact current contract."""
        if self._sealed_contract_sha256 is not None:
            return self._sealed_contract_sha256
        contract, _ = self._scan_contract()
        return contract


__all__ = ["AuthoritativeSafetensorSupervisionCache"]
