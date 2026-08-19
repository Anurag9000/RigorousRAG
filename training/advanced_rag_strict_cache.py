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
    """Base cache plus path authority, strict JSON, entry verification and exact sealing.

    ``contract_sha256`` is intentionally derived from the immutable cache identity and the
    complete set of exact tensor-entry digests. It is therefore safe to persist in promotion,
    restart and training-data receipts. Contract computation never trusts filenames alone:
    every entry must be a regular non-symlink manifest/tensor pair, the manifest key must hash
    back to its filename, the identity must match this cache, and the tensor bytes must match
    the manifest digest. Unknown/orphan files make sealing fail closed.

    Construction validates the current closed entry set immediately. This gives configuration
    preflight the same cache-integrity semantics as training/restart while still allowing a
    genuinely empty fresh cache to be created and populated by explicit materialization code.
    """

    def __init__(self, root: str | Path, identity: SupervisionCacheIdentity) -> None:
        safe = safe_advanced_path(root, label="supervision cache root", must_exist=False)
        if safe.exists() and not safe.is_dir():
            raise ValueError("supervision cache root must be a directory when it exists")
        super().__init__(safe, identity)
        # Force fail-closed validation now rather than deferring malformed/orphan discovery
        # until the first training batch or restart operation.
        _ = self.contract_sha256

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

    def get(self, key: str) -> Mapping[str, Any]:
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("safetensors is required for supervision cache reads") from exc
        tensor_path, manifest_path = self._paths(key)
        self._bounded_regular(tensor_path, "supervision tensor entry", _MAX_ENTRY_BYTES)
        manifest = self._strict_manifest(manifest_path)
        if manifest.get("key") != key:
            raise ValueError("supervision cache manifest key mismatch")
        if _stream_sha(tensor_path) != manifest.get("tensor_sha256"):
            raise ValueError("supervision cache tensor digest mismatch")
        tensors = load_file(str(tensor_path), device="cpu")
        if sorted(tensors) != list(manifest["tensor_names"]):
            raise ValueError("supervision cache tensor names differ from manifest")
        return tensors

    @property
    def contract_sha256(self) -> str:
        """Return a deterministic digest of the exact closed cache contents.

        The root is treated as a closed authority boundary: only paired ``<64hex>.json`` and
        ``<64hex>.safetensors`` files are permitted. This catches partial writes, stale files,
        symlink substitution and post-seal mutation before a cache is admitted into a recipe or
        restarted canonical bundle.
        """
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
            entries.append(
                {
                    "key": key,
                    "key_sha256": stem,
                    "tensor_sha256": actual_tensor_sha,
                    "tensor_names": list(manifest["tensor_names"]),
                }
            )
        payload = {
            "schema": "rigorousrag-authoritative-supervision-cache-contract/v1",
            "identity_sha256": self.identity.digest,
            "entry_count": len(entries),
            "entries": entries,
        }
        return hashlib.sha256(_canonical(payload)).hexdigest()


__all__ = ["AuthoritativeSafetensorSupervisionCache"]
