"""Validate that a generated requirements lock is fully pinned and hash-checked."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

_PINNED_REQUIREMENT = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^\s\\]+(?:\s*\\)?$")
_HASH_FRAGMENT = re.compile(r"--hash=sha256:[0-9a-f]{64}")
_MAX_FILE_BYTES = 20_000_000
_MAX_ARGUMENTS = 20


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _bounded_argv(argv: Iterable[str] | None) -> list[str] | None:
    if argv is None:
        return None
    result: list[str] = []
    for index, value in enumerate(argv):
        if index >= _MAX_ARGUMENTS:
            raise ValueError("Too many command-line arguments.")
        if not isinstance(value, str) or len(value) > 4096 or _contains_ascii_control(value):
            raise ValueError("A command-line argument is invalid or too long.")
        result.append(value)
    return result


def _safe_file(value: str | os.PathLike[str]) -> Path:
    rendered = os.fspath(value)
    if (
        not rendered
        or len(rendered) > 4096
        or _contains_ascii_control(rendered)
    ):
        raise ValueError("Lock path is invalid or too long.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    if absolute.is_symlink() or not absolute.exists() or not absolute.is_file():
        raise ValueError("Lock path must be a regular non-symlink file.")
    size = absolute.stat().st_size
    if size <= 0 or size > _MAX_FILE_BYTES:
        raise ValueError("Lock file is empty or exceeds the 20 MB limit.")
    return absolute


def verify_lock(path: Path) -> dict[str, int]:
    lock_path = _safe_file(path)
    text = lock_path.read_text(encoding="utf-8")
    if "--hash=sha256:" not in text:
        raise ValueError("Lock file contains no SHA-256 hashes.")
    if "--index-url" in text or "--trusted-host" in text:
        raise ValueError("Lock file must not embed package-index authority.")

    requirement_count = 0
    hash_count = 0
    current_requirement: str | None = None
    current_hashes = 0

    def finish_requirement() -> None:
        nonlocal requirement_count, current_requirement, current_hashes
        if current_requirement is None:
            return
        if current_hashes <= 0:
            raise ValueError(f"Pinned requirement lacks a SHA-256 hash: {current_requirement}")
        requirement_count += 1
        current_requirement = None
        current_hashes = 0

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("--") and not stripped.startswith("--hash="):
            raise ValueError(f"Unsupported lock directive: {stripped[:100]}")

        hashes = _HASH_FRAGMENT.findall(stripped)
        if hashes:
            hash_count += len(hashes)

        if raw_line[:1].isspace() or stripped.startswith("--hash="):
            if current_requirement is None:
                raise ValueError("Hash continuation appeared before a pinned requirement.")
            current_hashes += len(hashes)
            continue

        finish_requirement()
        requirement_text = stripped[:-1].rstrip() if stripped.endswith("\\") else stripped
        requirement_text = requirement_text.split(" --hash=", 1)[0].rstrip()
        if not _PINNED_REQUIREMENT.fullmatch(requirement_text):
            raise ValueError(f"Requirement is not exactly pinned: {requirement_text[:200]}")
        current_requirement = requirement_text
        current_hashes = len(hashes)

    finish_requirement()
    if requirement_count <= 0:
        raise ValueError("Lock file contains no pinned requirements.")
    return {"requirements": requirement_count, "hashes": hash_count}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate one generated release lock.")
    parser.add_argument("lock_file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = build_parser().parse_args(_bounded_argv(argv))
        result = verify_lock(Path(arguments.lock_file))
        print(f"verified {result['requirements']} requirements and {result['hashes']} hashes")
        return 0
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        print(f"lock verification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
