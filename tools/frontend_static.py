"""Resolve and validate RigorousRAG's bundled frontend assets.

The production service calls :func:`frontend_directory` directly. The optional
``install_portable_frontend_staticfiles`` adapter exists only for compatibility with
legacy code that still constructs ``StaticFiles(directory="frontend")``. It is narrow and
idempotent: only that exact sentinel is rebound, while all other callers retain FastAPI's
normal path semantics.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from fastapi.staticfiles import StaticFiles

_LEGACY_FRONTEND_SENTINEL = "frontend"
_REQUIRED_FRONTEND_FILES = ("index.html", "app.js", "lifecycle.js", "preload.js")
_INSTALL_MARKER = "_rigorousrag_frontend_adapter_installed"
_ORIGINAL_MARKER = "_rigorousrag_original_init"
_MAX_PATH_CHARS = 4096
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _is_link_or_reparse(info: os.stat_result) -> bool:
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _lexical_absolute(value: str | os.PathLike[str], *, label: str) -> Path:
    try:
        rendered = os.fspath(value)
    except TypeError as exc:
        raise RuntimeError(f"{label} is unavailable.") from exc
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH_CHARS
        or _contains_ascii_control(rendered)
    ):
        raise RuntimeError(f"{label} is invalid or too long.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _safe_lexical_ancestry(path: Path, *, label: str) -> None:
    """Reject redirected existing ancestors without resolving through them first."""

    for component in (path, *path.parents):
        try:
            info = os.lstat(component)
        except FileNotFoundError as exc:
            raise RuntimeError(f"{label} is unavailable.") from exc
        except OSError as exc:
            raise RuntimeError(f"{label} could not be inspected safely.") from exc
        if _is_link_or_reparse(info):
            raise RuntimeError(
                f"{label} may not contain symbolic-link or reparse-point components."
            )


def _safe_regular_file(path: Path, *, label: str) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable.") from exc
    if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{label} must be a regular non-symlink file.")
    _safe_lexical_ancestry(path.parent, label=label)


def _safe_directory(path: Path, *, label: str) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable.") from exc
    if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"{label} must be a real non-symlink directory.")
    _safe_lexical_ancestry(path.parent, label=label)


def frontend_directory() -> Path:
    """Return the verified lexical module-relative frontend directory."""

    module_file = _lexical_absolute(__file__, label="The frontend resolver module")
    _safe_regular_file(module_file, label="The frontend resolver module")

    tools_directory = module_file.parent
    package_root = tools_directory.parent
    _safe_directory(tools_directory, label="The frontend resolver package directory")
    _safe_directory(package_root, label="The frontend package root")

    absolute = package_root / _LEGACY_FRONTEND_SENTINEL
    _safe_directory(absolute, label="The bundled frontend directory")

    for filename in _REQUIRED_FRONTEND_FILES:
        if _contains_ascii_control(filename):  # defensive invariant
            raise RuntimeError("A bundled frontend filename is invalid.")
        _safe_regular_file(
            absolute / filename,
            label=f"Bundled frontend asset {filename}",
        )
    return absolute


def install_portable_frontend_staticfiles() -> None:
    """Install the narrow legacy-path adapter once per interpreter."""

    if bool(getattr(StaticFiles, _INSTALL_MARKER, False)):
        return

    original = getattr(StaticFiles, _ORIGINAL_MARKER, StaticFiles.__init__)
    if not callable(original):
        raise RuntimeError("FastAPI StaticFiles initialization is unavailable.")

    def portable_init(
        self: StaticFiles,
        *args: Any,
        directory: str | os.PathLike[str] | None = None,
        **kwargs: Any,
    ) -> None:
        selected: str | os.PathLike[str] | None = directory
        if selected == _LEGACY_FRONTEND_SENTINEL:
            selected = frontend_directory()
        original(self, *args, directory=selected, **kwargs)

    setattr(StaticFiles, _ORIGINAL_MARKER, original)
    StaticFiles.__init__ = portable_init  # type: ignore[method-assign]
    setattr(StaticFiles, _INSTALL_MARKER, True)
