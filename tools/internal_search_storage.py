"""Cross-platform content signatures for classic search storage."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, List, Optional, Tuple

_FileIdentity = Tuple[str, int, int, str]
_StorageSignature = Tuple[str, Tuple[_FileIdentity, ...]]
_MAX_MANIFEST_BYTES = 1_000_000
_MAX_FILE_BYTES = 2_000_000_000
_MAX_PATH_CHARS = 4096
_REPARSE = 0x0400


def _controls(value: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


def _link(info: os.stat_result) -> bool:
    attrs = int(getattr(info, "st_file_attributes", 0) or 0)
    return stat.S_ISLNK(info.st_mode) or bool(attrs & _REPARSE)


def _absolute_without_resolving(path: str | os.PathLike[str]) -> Path:
    rendered = os.fspath(path)
    if not isinstance(rendered, str) or not rendered or len(rendered) > _MAX_PATH_CHARS or _controls(rendered):
        raise ValueError("CLASSIC_STORAGE_DIR is invalid or too long.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for component in (absolute, *absolute.parents):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("CLASSIC_STORAGE_DIR could not be inspected safely.") from exc
        if _link(info):
            raise ValueError("CLASSIC_STORAGE_DIR may not contain link/reparse components.")
    return absolute


def _regular_bytes(path: Path, maximum: int) -> bytes | None:
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if _link(info) or not stat.S_ISREG(info.st_mode) or info.st_size < 0 or info.st_size > maximum:
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        opened = os.fstat(fd)
        if _link(opened) or not stat.S_ISREG(opened.st_mode):
            return None
        data = bytearray()
        while True:
            chunk = os.read(fd, min(64 * 1024, maximum + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > maximum:
                return None
    except OSError:
        return None
    finally:
        os.close(fd)
    try:
        after = os.lstat(path)
    except OSError:
        return None
    if _link(after) or not stat.S_ISREG(after.st_mode):
        return None
    return bytes(data)


def _file_identity(path: Path) -> _FileIdentity:
    try:
        info = os.lstat(path)
    except OSError:
        return path.name, -1, -1, "missing"
    if _link(info) or not stat.S_ISREG(info.st_mode):
        return path.name, -2, -2, "invalid"
    if info.st_size > _MAX_FILE_BYTES:
        return path.name, int(info.st_size), int(info.st_mtime_ns), "oversize"
    payload = _regular_bytes(path, _MAX_FILE_BYTES)
    if payload is None:
        return path.name, int(info.st_size), int(info.st_mtime_ns), "unstable"
    return path.name, len(payload), int(info.st_mtime_ns), hashlib.sha256(payload).hexdigest()


def _read_manifest(path: Path) -> Optional[dict[str, Any]]:
    payload = _regular_bytes(path, _MAX_MANIFEST_BYTES)
    if not payload:
        return None
    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _manifest_member_paths(root: Path, manifest: Path) -> List[Path]:
    value = _read_manifest(manifest)
    if value is None:
        return []
    generation = value.get("generation")
    if not isinstance(generation, str) or len(generation) != 32 or any(ch not in "0123456789abcdef" for ch in generation):
        return []
    files = value.get("files")
    if not isinstance(files, dict):
        return []
    expected = {"crawl": f"crawl_state.{generation}.json", "index": f"index.{generation}.json", "pagerank": f"pagerank.{generation}.json"}
    result = []
    for key, filename in expected.items():
        entry = files.get(key)
        name = entry.get("name") if isinstance(entry, dict) else None
        if name != filename or not isinstance(name, str) or Path(name).name != name:
            return []
        result.append(root / name)
    return result


def _storage_signature(storage_dir: str | os.PathLike[str]) -> _StorageSignature:
    root = _absolute_without_resolving(storage_dir)
    manifest = root / "snapshot_manifest.json"
    paths = [manifest, root / "crawl_state.json", root / "index.json", root / "pagerank.json"]
    paths.extend(_manifest_member_paths(root, manifest))
    return str(root), tuple(_file_identity(path) for path in paths)


__all__ = ["_absolute_without_resolving", "_file_identity", "_manifest_member_paths", "_read_manifest", "_storage_signature"]
