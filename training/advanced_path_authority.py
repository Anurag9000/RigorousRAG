"""Fail-closed filesystem path normalization for advanced RAG operator inputs."""
from __future__ import annotations

from pathlib import Path


def safe_advanced_path(
    path: str | Path,
    *,
    label: str,
    must_exist: bool = False,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    """Reject symlink traversal before canonical resolution, then enforce path kind."""
    if require_file and require_directory:
        raise ValueError("path cannot require both file and directory")
    raw = Path(path).expanduser()
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    for candidate in (absolute, *absolute.parents):
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"{label} traverses symlink: {candidate}")
    if must_exist:
        resolved = absolute.resolve(strict=True)
    else:
        resolved = absolute.resolve(strict=False)
    if require_file and (not resolved.exists() or not resolved.is_file()):
        raise ValueError(f"{label} must be a regular file")
    if require_directory and (not resolved.exists() or not resolved.is_dir()):
        raise ValueError(f"{label} must be a directory")
    return resolved


__all__ = ["safe_advanced_path"]
