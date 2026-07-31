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


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _safe_regular_file(path: Path, *, label: str) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable.") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{label} must be a regular non-symlink file.")


def frontend_directory() -> Path:
    """Return the verified module-relative frontend directory."""

    try:
        module_file = Path(__file__).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("The frontend resolver module is unavailable.") from exc
    _safe_regular_file(module_file, label="The frontend resolver module")

    candidate = module_file.parent.parent / _LEGACY_FRONTEND_SENTINEL
    absolute = Path(os.path.abspath(candidate))
    try:
        directory_info = os.lstat(absolute)
    except OSError as exc:
        raise RuntimeError("The bundled frontend directory is unavailable.") from exc
    if stat.S_ISLNK(directory_info.st_mode) or not stat.S_ISDIR(directory_info.st_mode):
        raise RuntimeError("The bundled frontend path must be a real directory.")

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
