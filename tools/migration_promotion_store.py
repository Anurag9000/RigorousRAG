"""Append-only, manifest-pointer persistence for migration promotion reports."""

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

from tools.migration_promotion import PromotionReport
from tools.migration_types import digest, exact_integer, identifier

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_REPORT_BYTES = 5_000_000
_MAX_HISTORY = 10_000


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _safe_root(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("promotion root must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("promotion root is invalid.")
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
            raise ValueError("promotion root could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("promotion root may not contain redirects.")
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
    if not payload or len(payload) > _MAX_REPORT_BYTES:
        raise ValueError("promotion report exceeds the file-size limit.")
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


class MigrationPromotionStore:
    """Append immutable reports and atomically select the latest report per task."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = _safe_root(root)
        self.root.mkdir(parents=True, exist_ok=True)
        info = self.root.lstat()
        if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("promotion root must be a regular directory.")
        self._identity = (int(info.st_dev), int(info.st_ino))
        self._lock = threading.RLock()

    def _verify_root(self) -> None:
        info = self.root.lstat()
        if (
            _redirecting(info)
            or not stat.S_ISDIR(info.st_mode)
            or (int(info.st_dev), int(info.st_ino)) != self._identity
        ):
            raise RuntimeError("promotion root identity changed.")

    def _task_directory(self, task_id: str) -> Path:
        return self.root / identifier(task_id, "task_id", 64)

    @staticmethod
    def _read_bounded(path: Path) -> bytes:
        info = path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("promotion store member is not a regular file.")
        if info.st_size <= 0 or info.st_size > _MAX_REPORT_BYTES:
            raise RuntimeError("promotion store member exceeds its size limit.")
        with path.open("rb") as handle:
            payload = handle.read(_MAX_REPORT_BYTES + 1)
        if len(payload) > _MAX_REPORT_BYTES:
            raise RuntimeError("promotion store member exceeds its size limit.")
        return payload

    @staticmethod
    def _decode_report(payload: bytes) -> PromotionReport:
        raw = _strict_json(payload, "promotion report")
        if not isinstance(raw, dict):
            raise RuntimeError("promotion report must be an object.")
        try:
            return PromotionReport(**raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("promotion report schema is invalid.") from exc

    @staticmethod
    def _atomic_file(path: Path, payload: bytes) -> None:
        descriptor = -1
        temporary: Path | None = None
        try:
            descriptor, raw = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
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

    def write(self, report: PromotionReport) -> PromotionReport:
        if not isinstance(report, PromotionReport):
            raise ValueError("report must be a PromotionReport.")
        report_digest = report.report_digest
        with self._lock:
            self._verify_root()
            directory = self._task_directory(report.task_id)
            if directory.exists():
                info = directory.lstat()
                if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
                    raise RuntimeError("promotion task directory is invalid.")
            else:
                directory.mkdir(mode=0o700)
                info = directory.lstat()
                if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
                    raise RuntimeError("promotion task directory is invalid.")
                _fsync_directory(self.root)
            report_path = directory / f"{report_digest}.json"
            payload = _encoded(asdict(report))
            if report_path.exists():
                existing = self._decode_report(self._read_bounded(report_path))
                if existing.report_digest != report_digest:
                    raise RuntimeError("promotion report digest collision detected.")
            else:
                self._atomic_file(report_path, payload)
            pointer = _encoded({"report_digest": report_digest})
            self._atomic_file(directory / "current.json", pointer)
        return self.read(report.task_id, report_digest=report_digest)

    def read(
        self,
        task_id: str,
        *,
        report_digest: str | None = None,
    ) -> PromotionReport:
        with self._lock:
            self._verify_root()
            directory = self._task_directory(task_id)
            if report_digest is None:
                pointer = _strict_json(
                    self._read_bounded(directory / "current.json"),
                    "promotion pointer",
                )
                if not isinstance(pointer, dict) or set(pointer) != {"report_digest"}:
                    raise RuntimeError("promotion pointer schema is invalid.")
                selected = digest(pointer["report_digest"], "report_digest")
            else:
                selected = digest(report_digest, "report_digest")
            report = self._decode_report(
                self._read_bounded(directory / f"{selected}.json")
            )
            if report.task_id != identifier(task_id, "task_id", 64):
                raise RuntimeError("promotion report task identity is invalid.")
            if report.report_digest != selected:
                raise RuntimeError("promotion report digest is invalid.")
            return report

    def history(self, task_id: str, *, limit: int = 100) -> tuple[PromotionReport, ...]:
        count = exact_integer(limit, "limit", 1, _MAX_HISTORY)
        with self._lock:
            self._verify_root()
            directory = self._task_directory(task_id)
            info = directory.lstat()
            if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("promotion task directory is invalid.")
            paths = [
                path
                for path in directory.glob("*.json")
                if path.name != "current.json"
            ]
            if len(paths) > _MAX_HISTORY:
                raise RuntimeError("promotion history exceeds the report limit.")
            decoded = [
                (path, self._decode_report(self._read_bounded(path)))
                for path in paths
            ]
            expected = identifier(task_id, "task_id", 64)
            if any(
                report.task_id != expected
                or path.stem != report.report_digest
                for path, report in decoded
            ):
                raise RuntimeError("promotion history contains an invalid report.")
            decoded.sort(
                key=lambda item: (item[1].evaluated_at, item[1].report_digest),
                reverse=True,
            )
            return tuple(report for _path, report in decoded[:count])

    def remove_task(self, task_id: str) -> bool:
        with self._lock:
            self._verify_root()
            directory = self._task_directory(task_id)
            try:
                info = directory.lstat()
            except FileNotFoundError:
                return False
            if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("promotion task directory is invalid.")
            shutil.rmtree(directory)
            _fsync_directory(self.root)
            return True


__all__ = ["MigrationPromotionStore"]
