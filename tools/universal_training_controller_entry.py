#!/usr/bin/env python3
"""Rate-limit-safe immutable bootstrap for exhaustive training control v37.

v37 preserves the complete byte-pinned v36/v20-v34 scientific/controller stack
and the literal OPF_ADP runtime, but replaces v36's per-blob public GitHub API
materialization path with one pinned repository archive fetch. The archive
contains a bundle-only directory whose entries point directly at the exact 60
historical Git blobs. Every extracted file is reverified by Git blob SHA before
admission to the cache. The scheduler itself is not reimplemented here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from types import ModuleType

BUNDLE_VERSION = "v37"
HOST_REPO = "Anurag9000/RigorousRAG"
HOST_ARCHIVE_COMMIT = "4e623756ea853eff079104ef5f3483bafc01f0fb"
HOST_BUNDLE_DIR = "controller_bundle_v36"
V36_COMMIT = "e498aa3496b1e79c29c4060db4b11433d69bad97"
V36_BLOB = "1840035fdecb5d5fbab06846435cb86787842738"
V36_URL = (
    f"https://raw.githubusercontent.com/{HOST_REPO}/{V36_COMMIT}/"
    "tools/universal_training_controller_entry.py"
)
ARCHIVE_URL = f"https://codeload.github.com/{HOST_REPO}/zip/{HOST_ARCHIVE_COMMIT}"
USER_AGENT = "opf-exhaustive-training-controller/41"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _root() -> Path:
    return Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()


def _prepare_v36_entry(root: Path) -> Path:
    path = root / ".training_control" / "bootstrap" / V36_COMMIT / "universal_training_controller_entry.py"
    if path.is_file() and git_blob_sha(path.read_bytes()) == V36_BLOB:
        return path
    payload = _fetch(V36_URL)
    actual = git_blob_sha(payload)
    if actual != V36_BLOB:
        raise RuntimeError(f"Pinned v36 entry checksum mismatch: {actual} != {V36_BLOB}")
    _atomic_write(path, payload)
    return path


def _load_v36(path: Path) -> ModuleType:
    name = "_training_control_v36_immutable"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned v36 controller entry: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bundle_cache(root: Path, module: ModuleType) -> tuple[Path, dict[str, str], dict[str, object]]:
    files = dict(module.CONTROLLER_FILES)
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    cache = root / ".training_control" / "controller_bundle" / f"v36-{digest}"
    marker = {
        "schema": "rigorousrag.training_control.bundle.v36",
        "repository": module.HOST_REPO,
        "files": files,
    }
    return cache, files, marker


def _cache_valid(cache: Path, files: dict[str, str], marker_payload: dict[str, object]) -> bool:
    marker = cache / "BUNDLE.json"
    try:
        if json.loads(marker.read_text(encoding="utf-8")) != marker_payload:
            return False
        for relative, expected in files.items():
            path = cache / Path(relative).name
            if not path.is_file() or git_blob_sha(path.read_bytes()) != expected:
                return False
        return True
    except Exception:
        return False


def _copy_verified_local(root: Path, cache: Path, files: dict[str, str]) -> set[str]:
    ready: set[str] = set()
    for relative, expected in files.items():
        destination = cache / Path(relative).name
        if destination.is_file() and git_blob_sha(destination.read_bytes()) == expected:
            ready.add(relative)
            continue
        source = root / relative
        try:
            payload = source.read_bytes()
        except Exception:
            continue
        if git_blob_sha(payload) == expected:
            _atomic_write(destination, payload)
            ready.add(relative)
    return ready


def _archive_members(payload: bytes) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            normalized = info.filename.replace("\\", "/")
            if "/" not in normalized:
                continue
            relative = normalized.split("/", 1)[1]
            if not relative or relative.startswith("../") or "/../" in f"/{relative}":
                continue
            members[relative] = archive.read(info)
    return members


def _bundle_member(relative: str) -> str:
    return f"{HOST_BUNDLE_DIR}/{Path(relative).name}"


def _materialize_bundle(root: Path, module: ModuleType) -> Path:
    cache, files, marker_payload = _bundle_cache(root, module)
    if _cache_valid(cache, files, marker_payload):
        return cache

    cache.mkdir(parents=True, exist_ok=True)
    ready = _copy_verified_local(root, cache, files)
    missing = [relative for relative in files if relative not in ready]
    if missing:
        archive = _archive_members(_fetch(ARCHIVE_URL))
        absent = [relative for relative in missing if _bundle_member(relative) not in archive]
        if absent:
            raise RuntimeError(
                "Pinned RigorousRAG bundle archive does not contain required controller files: "
                + ", ".join(absent)
            )
        for relative in missing:
            payload = archive[_bundle_member(relative)]
            expected = files[relative]
            actual = git_blob_sha(payload)
            if actual != expected:
                raise RuntimeError(
                    f"Controller bundle blob mismatch for {relative}: {actual} != {expected}"
                )
            _atomic_write(cache / Path(relative).name, payload)

    for relative, expected in files.items():
        path = cache / Path(relative).name
        actual = git_blob_sha(path.read_bytes()) if path.is_file() else "missing"
        if actual != expected:
            raise RuntimeError(f"Controller cache mismatch for {relative}: {actual} != {expected}")

    _atomic_write(
        cache / "BUNDLE.json",
        (json.dumps(marker_payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return cache


def bootstrap() -> ModuleType:
    root = _root()
    entry = _prepare_v36_entry(root)
    module = _load_v36(entry)
    _materialize_bundle(root, module)
    return module


def main() -> int:
    module = bootstrap()
    return int(module.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
