"""Generate one hashed, platform-specific runtime dependency lock.

This script intentionally delegates resolution to ``pip-tools`` in the target operating
system and Python interpreter. It does not pretend that one lock is portable across
platforms or Python minors.
"""

from __future__ import annotations

import argparse
import os
import platform
import stat
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Iterable, Sequence

_MAX_ARGUMENTS = 50
_MAX_PATH_CHARS = 4096


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _bounded_argv(argv: Iterable[str] | None) -> list[str] | None:
    if argv is None:
        return None
    values: list[str] = []
    for index, value in enumerate(argv):
        if index >= _MAX_ARGUMENTS:
            raise ValueError("Too many command-line arguments.")
        if (
            not isinstance(value, str)
            or len(value) > _MAX_PATH_CHARS
            or _contains_ascii_control(value)
        ):
            raise ValueError("A command-line argument is invalid or too long.")
        values.append(value)
    return values


def _safe_path(value: str | os.PathLike[str], *, label: str) -> Path:
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH_CHARS
        or _contains_ascii_control(rendered)
    ):
        raise ValueError(f"{label} is invalid or too long.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _platform_tag() -> str:
    system = platform.system().strip().lower()
    mapping = {"linux": "linux", "windows": "windows", "darwin": "macos"}
    if system not in mapping:
        raise RuntimeError(f"Unsupported lock-generation platform: {system or 'unknown'}.")
    return mapping[system]


def default_output_path() -> Path:
    python_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    return Path("locks") / f"runtime-{_platform_tag()}-{python_tag}.txt"


def _pip_compile_executable() -> Path:
    """Return this interpreter environment's compile-only pip-tools entry point."""

    scripts_path = sysconfig.get_path("scripts")
    if not isinstance(scripts_path, str) or not scripts_path:
        raise RuntimeError("The interpreter scripts directory is unavailable.")
    filename = "pip-compile.exe" if os.name == "nt" else "pip-compile"
    candidate = _safe_path(Path(scripts_path) / filename, label="pip-compile path")
    try:
        info = os.stat(candidate, follow_symlinks=True)
    except OSError as exc:
        raise RuntimeError("pip-compile is not installed for this interpreter.") from exc
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError("pip-compile is not a regular executable file.")
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a hashed RigorousRAG runtime lock for this OS/Python."
    )
    parser.add_argument("--input", default="requirements.txt")
    parser.add_argument("--output", default=str(default_output_path()))
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Permit pip-tools to upgrade already-resolved versions.",
    )
    return parser


def generate_lock(
    *,
    input_path: Path,
    output_path: Path,
    upgrade: bool,
) -> None:
    source = _safe_path(input_path, label="input path")
    destination = _safe_path(output_path, label="output path")
    if not source.exists() or not source.is_file() or source.is_symlink():
        raise ValueError("The requirements input must be a regular non-symlink file.")
    if source.stat().st_size > 1_000_000:
        raise ValueError("The requirements input exceeds the 1 MB limit.")
    if destination == source:
        raise ValueError("The lock output may not replace the requirements input.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.is_symlink():
        raise ValueError("The lock output may not be a symbolic link.")

    command = [
        str(_pip_compile_executable()),
        str(source),
        "--resolver=backtracking",
        "--generate-hashes",
        "--no-annotate",
        "--no-emit-index-url",
        "--no-emit-trusted-host",
        "--output-file",
        str(destination),
    ]
    if upgrade:
        command.append("--upgrade")

    environment = os.environ.copy()
    environment.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    subprocess.run(command, check=True, env=environment)

    if not destination.exists() or destination.is_symlink():
        raise RuntimeError("pip-tools did not create a safe lock file.")
    if destination.stat().st_size <= 0 or destination.stat().st_size > 20_000_000:
        raise RuntimeError("The generated lock file is empty or unexpectedly large.")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = build_parser().parse_args(_bounded_argv(argv))
        generate_lock(
            input_path=Path(arguments.input),
            output_path=Path(arguments.output),
            upgrade=bool(arguments.upgrade),
        )
        print(_safe_path(arguments.output, label="output path"))
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError, TypeError, ValueError) as exc:
        print(f"lock generation failed: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
