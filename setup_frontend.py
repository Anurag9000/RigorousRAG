"""Verify the checked-in RigorousRAG frontend without regenerating it.

The browser assets are security-reviewed source files, not generated runtime state.
Historically this helper rewrote them from embedded stale strings, which could undo
later DOM, lifecycle, timeout, and privacy hardening. It now performs a bounded,
read-only verification and never mutates the repository.
"""

from __future__ import annotations

import itertools
import os
import stat
import sys
from pathlib import Path
from typing import Iterable

_MAX_ASSET_BYTES = 2_000_000
_MAX_ARGUMENTS = 1
_REQUIRED_ASSETS = (
    "index.html",
    "preload.js",
    "app.js",
    "lifecycle.js",
)
_FORBIDDEN_TOKENS = (
    ".innerHTML",
    "localStorage",
    "cdn.jsdelivr",
    "fonts.googleapis",
    "X-Owner-ID",
)
_REQUIRED_MARKERS = {
    "index.html": (
        '<script src="preload.js" defer></script>',
        '<script src="app.js" defer></script>',
        '<script src="lifecycle.js" defer></script>',
        'id="query-input" maxlength="20000"',
    ),
    "preload.js": (
        "AbortController",
        "DEFAULT_PRELOAD_TIMEOUT_MS",
    ),
    "app.js": (
        "sessionStorage",
        'fetchApi("/config")',
        '"X-API-Key"',
    ),
    "lifecycle.js": (
        "MAX_CLIENT_UPLOAD_FILES = 100",
        "visual_source_available",
        "AbortController",
    ),
}


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _repository_root(value: str | os.PathLike[str] | None = None) -> Path:
    try:
        raw = Path(__file__).resolve().parent if value is None else Path(os.fspath(value))
    except Exception as exc:
        raise ValueError("Repository root must be a filesystem path.") from exc
    rendered = os.fspath(raw)
    if not rendered or len(rendered) > 4096 or _contains_ascii_control(rendered):
        raise ValueError("Repository root is invalid or too long.")
    candidate = raw if raw.is_absolute() else Path.cwd() / raw
    absolute = Path(os.path.abspath(candidate))
    for component in (absolute, *absolute.parents):
        try:
            if component.is_symlink():
                raise ValueError(
                    "Repository root may not contain symbolic-link components."
                )
        except OSError as exc:
            raise ValueError("Repository root could not be inspected safely.") from exc
    if not absolute.exists() or not absolute.is_dir():
        raise ValueError("Repository root must be an existing directory.")
    return absolute


def _read_asset(path: Path) -> str:
    try:
        if path.is_symlink():
            raise ValueError(f"Frontend asset {path.name} may not be a symbolic link.")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(
            f"Frontend asset {path.name} is missing or could not be opened safely."
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Frontend asset {path.name} must be a regular file.")
        if metadata.st_size <= 0 or metadata.st_size > _MAX_ASSET_BYTES:
            raise ValueError(
                f"Frontend asset {path.name} is empty or exceeds the byte limit."
            )
        payload = bytearray()
        while True:
            remaining = _MAX_ASSET_BYTES + 1 - len(payload)
            if remaining <= 0:
                raise ValueError(f"Frontend asset {path.name} exceeds the byte limit.")
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            payload.extend(chunk)
        try:
            return bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Frontend asset {path.name} must be valid UTF-8.") from exc
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(
            f"Frontend asset {path.name} could not be read safely."
        ) from exc
    finally:
        os.close(descriptor)


def verify_frontend(root: str | os.PathLike[str] | None = None) -> list[str]:
    """Return verified asset names; raise on missing, unsafe, or stale assets."""

    repository = _repository_root(root)
    frontend = repository / "frontend"
    if frontend.is_symlink() or not frontend.exists() or not frontend.is_dir():
        raise ValueError("frontend must be an existing non-symlink directory.")

    verified: list[str] = []
    asset_text: dict[str, str] = {}
    for name in _REQUIRED_ASSETS:
        path = frontend / name
        text = _read_asset(path)
        asset_text[name] = text
        for marker in _REQUIRED_MARKERS.get(name, ()):
            if marker not in text:
                raise ValueError(
                    f"Frontend asset {name} is missing required marker: {marker}"
                )
        verified.append(name)

    combined = "\n".join(asset_text.values())
    for token in _FORBIDDEN_TOKENS:
        if token in combined:
            raise ValueError(f"Frontend contains forbidden token: {token}")
    return verified


def _bounded_arguments(argv: Iterable[str] | None) -> list[str]:
    values = sys.argv[1:] if argv is None else argv
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("Arguments must be an iterable of strings.")
    try:
        arguments = list(itertools.islice(iter(values), _MAX_ARGUMENTS + 1))
    except Exception as exc:
        raise ValueError("Arguments must be iterable.") from exc
    if len(arguments) > _MAX_ARGUMENTS:
        raise ValueError("At most one repository-root argument is supported.")
    if any(
        not isinstance(argument, str) or _contains_ascii_control(argument)
        for argument in arguments
    ):
        raise ValueError("Arguments must be valid strings.")
    return arguments


def main(argv: Iterable[str] | None = None) -> int:
    try:
        arguments = _bounded_arguments(argv)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        verified = verify_frontend(arguments[0] if arguments else None)
    except Exception as exc:
        print(f"Frontend verification failed: {type(exc).__name__}.", file=sys.stderr)
        return 1
    print("Verified checked-in frontend assets: " + ", ".join(verified))
    print("No files were generated or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
