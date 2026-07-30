"""Bind the legacy relative frontend mount to the repository's verified asset directory.

``server_app`` historically constructs ``StaticFiles(directory="frontend")``. That path
is relative to the process working directory and therefore fails when the service is
launched outside the repository root. Importing this module installs one narrow,
idempotent adapter on FastAPI's ``StaticFiles`` class: only the exact legacy sentinel
``"frontend"`` is replaced, while every other caller and path keeps normal semantics.
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


def frontend_directory() -> Path:
    """Return the verified repository-relative frontend directory."""

    candidate = Path(__file__).parent.parent / _LEGACY_FRONTEND_SENTINEL
    absolute = Path(os.path.abspath(candidate))
    try:
        directory_info = os.lstat(absolute)
    except OSError as exc:
        raise RuntimeError("The bundled frontend directory is unavailable.") from exc
    if stat.S_ISLNK(directory_info.st_mode) or not stat.S_ISDIR(directory_info.st_mode):
        raise RuntimeError("The bundled frontend path must be a regular directory.")

    for filename in _REQUIRED_FRONTEND_FILES:
        if _contains_ascii_control(filename):  # defensive invariant
            raise RuntimeError("A bundled frontend filename is invalid.")
        path = absolute / filename
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise RuntimeError("A required bundled frontend asset is unavailable.") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("A required bundled frontend asset is unsafe.")
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
