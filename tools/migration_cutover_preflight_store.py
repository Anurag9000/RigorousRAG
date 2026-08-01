"""Append-only storage for non-mutating migration cutover preflights."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tools.migration_cutover_preflight import CutoverPreflight
from tools.migration_types import digest, exact_integer, identifier

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_FILE_BYTES = 2_000_000
_MAX_HISTORY = 10_000


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _root(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("cutover preflight root must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("cutover preflight root is invalid.")
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
            raise ValueError("cutover preflight root could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("cutover preflight root may not contain redirects.")
    return absolute


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _strict_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise RuntimeError(f"{label} is invalid JSON.") from exc


def _encoded(value: Any) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if not payload or len(payload) > _MAX_FILE_BYTES:
        raise ValueError("cutover preflight exceeds the file-size limit.")
    return payload


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


class MigrationCutoverPreflightStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = _root(root)
        self.root.mkdir(parents=True, exist_ok=True)
        info = self.root.lstat()
        if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("cutover preflight root must be a regular directory.")
        self._identity = (int(info.st_dev), int(info.st_ino))
        self._lock = threading.RLock()

    def _verify(self) -> None:
        info = self.root.lstat()
        if (
            _redirecting(info)
            or not stat.S_ISDIR(info.st_mode)
            or (int(info.st_dev), int(info.st_ino)) != self._identity
        ):
            raise RuntimeError("cutover preflight root identity changed.")

    def _task_directory(self, task_id: str) -> Path:
        return self.root / identifier(task_id, "task_id", 64)

    @staticmethod
    def _read(path: Path) -> bytes:
        info = path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("cutover preflight member is not a regular file.")
        if info.st_size <= 0 or info.st_size > _MAX_FILE_BYTES:
            raise RuntimeError("cutover preflight member exceeds its size limit.")
        with path.open("rb") as handle:
            payload = handle.read(_MAX_FILE_BYTES + 1)
        if len(payload) > _MAX_FILE_BYTES:
            raise RuntimeError("cutover preflight member exceeds its size limit.")
        return payload

    @staticmethod
    def _decode(payload: bytes) -> CutoverPreflight:
        raw = _strict_json(payload, "cutover preflight")
        if not isinstance(raw, dict):
            raise RuntimeError("cutover preflight must be an object.")
        try:
            return CutoverPreflight(**raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("cutover preflight schema is invalid.") from exc

    @staticmethod
    def _atomic(path: Path, payload: bytes) -> None:
        descriptor = -1
        temporary: Path | None = None
        try:
            descriptor, raw = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            temporary = Path(raw)
            try:
                os.fchmod(descriptor, 0o600)
            except OSError:
                pass
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary, path)
            temporary = None
            _fsync_directory(path.parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def write(self, preflight: CutoverPreflight) -> CutoverPreflight:
        if not isinstance(preflight, CutoverPreflight):
            raise ValueError("preflight must be CutoverPreflight.")
        preflight_digest = preflight.preflight_digest
        with self._lock:
            self._verify()
            directory = self._task_directory(preflight.task_id)
            if directory.exists():
                info = directory.lstat()
                if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
                    raise RuntimeError("cutover preflight task directory is invalid.")
            else:
                directory.mkdir(mode=0o700)
                _fsync_directory(self.root)
            path = directory / f"{preflight_digest}.json"
            payload = _encoded(asdict(preflight))
            if path.exists():
                existing = self._decode(self._read(path))
                if existing.preflight_digest != preflight_digest:
                    raise RuntimeError("cutover preflight digest collision detected.")
            else:
                self._atomic(path, payload)
            self._atomic(
                directory / "current.json",
                _encoded({"preflight_digest": preflight_digest}),
            )
        return self.read(preflight.task_id, preflight_digest=preflight_digest)

    def read(
        self,
        task_id: str,
        *,
        preflight_digest: str | None = None,
    ) -> CutoverPreflight:
        with self._lock:
            self._verify()
            directory = self._task_directory(task_id)
            if preflight_digest is None:
                pointer = _strict_json(
                    self._read(directory / "current.json"),
                    "cutover preflight pointer",
                )
                if not isinstance(pointer, dict) or set(pointer) != {"preflight_digest"}:
                    raise RuntimeError("cutover preflight pointer schema is invalid.")
                selected = digest(pointer["preflight_digest"], "preflight_digest")
            else:
                selected = digest(preflight_digest, "preflight_digest")
            value = self._decode(self._read(directory / f"{selected}.json"))
            if value.task_id != identifier(task_id, "task_id", 64):
                raise RuntimeError("cutover preflight task identity is invalid.")
            if value.preflight_digest != selected:
                raise RuntimeError("cutover preflight digest is invalid.")
            return value

    def history(self, task_id: str, *, limit: int = 100) -> tuple[CutoverPreflight, ...]:
        count = exact_integer(limit, "limit", 1, _MAX_HISTORY)
        with self._lock:
            self._verify()
            directory = self._task_directory(task_id)
            info = directory.lstat()
            if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("cutover preflight task directory is invalid.")
            paths = [path for path in directory.glob("*.json") if path.name != "current.json"]
            if len(paths) > _MAX_HISTORY:
                raise RuntimeError("cutover preflight history exceeds the limit.")
            values = [(path, self._decode(self._read(path))) for path in paths]
            expected = identifier(task_id, "task_id", 64)
            if any(
                value.task_id != expected or path.stem != value.preflight_digest
                for path, value in values
            ):
                raise RuntimeError("cutover preflight history contains invalid data.")
            values.sort(
                key=lambda item: (item[1].created_at, item[1].preflight_digest),
                reverse=True,
            )
            return tuple(value for _path, value in values[:count])

    def remove_task(self, task_id: str) -> bool:
        with self._lock:
            self._verify()
            directory = self._task_directory(task_id)
            try:
                info = directory.lstat()
            except FileNotFoundError:
                return False
            if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("cutover preflight task directory is invalid.")
            shutil.rmtree(directory)
            _fsync_directory(self.root)
            return True


__all__ = ["MigrationCutoverPreflightStore"]
