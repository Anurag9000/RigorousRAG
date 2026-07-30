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
_MAX_LOCK_BYTES = 20_000_000
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400


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


def _is_link_or_reparse(info: os.stat_result) -> bool:
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _contains_link_component(path: Path) -> bool:
    """Return whether any existing lexical component is a link/reparse point."""

    absolute = _safe_path(path, label="path")
    for component in (absolute, *absolute.parents):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            continue
        except OSError:
            return True
        if _is_link_or_reparse(info):
            return True
    return False


def _require_safe_ancestry(path: Path, *, label: str) -> None:
    if _contains_link_component(path):
        raise ValueError(f"{label} may not contain symbolic-link or reparse-point components.")


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
    _require_safe_ancestry(candidate, label="pip-compile path")
    try:
        info = os.stat(candidate, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError("pip-compile is not installed for this interpreter.") from exc
    if not stat.S_ISREG(info.st_mode) or _is_link_or_reparse(info):
        raise RuntimeError("pip-compile is not a safe regular executable file.")
    return candidate


def _write_github_output(destination: Path) -> None:
    """Publish the generated absolute path and artifact name to GitHub Actions."""

    output_value = os.environ.get("GITHUB_OUTPUT")
    if not output_value:
        raise RuntimeError("GITHUB_OUTPUT is unavailable.")
    output_path = _safe_path(output_value, label="GITHUB_OUTPUT path")
    _require_safe_ancestry(output_path.parent, label="GITHUB_OUTPUT parent")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _require_safe_ancestry(output_path, label="GITHUB_OUTPUT path")
    if output_path.exists():
        info = os.stat(output_path, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or _is_link_or_reparse(info):
            raise ValueError("GITHUB_OUTPUT must be a safe regular file.")
        if info.st_size > 1_000_000:
            raise ValueError("GITHUB_OUTPUT exceeds the 1 MB limit.")

    rendered_path = destination.as_posix()
    rendered_name = destination.stem
    if (
        _contains_ascii_control(rendered_path)
        or _contains_ascii_control(rendered_name)
        or len(rendered_path) > _MAX_PATH_CHARS
        or len(rendered_name) > 255
    ):
        raise ValueError("Generated workflow output is invalid or too long.")
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"path={rendered_path}\n")
        handle.write(f"name={rendered_name}\n")


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
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Publish path/name values to the GitHub Actions output file.",
    )
    return parser


def generate_lock(
    *,
    input_path: Path,
    output_path: Path,
    upgrade: bool,
) -> Path:
    source = _safe_path(input_path, label="input path")
    destination = _safe_path(output_path, label="output path")
    _require_safe_ancestry(source, label="requirements input")
    if not source.exists() or not source.is_file():
        raise ValueError("The requirements input must be a safe regular file.")
    source_info = os.stat(source, follow_symlinks=False)
    if not stat.S_ISREG(source_info.st_mode) or _is_link_or_reparse(source_info):
        raise ValueError("The requirements input must be a safe regular file.")
    if source_info.st_size > 1_000_000:
        raise ValueError("The requirements input exceeds the 1 MB limit.")
    if destination == source:
        raise ValueError("The lock output may not replace the requirements input.")

    _require_safe_ancestry(destination.parent, label="lock output parent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_safe_ancestry(destination, label="lock output")

    command = [
        str(_pip_compile_executable()),
        str(source),
        "--resolver=backtracking",
        "--generate-hashes",
        "--allow-unsafe",
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

    _require_safe_ancestry(destination, label="generated lock")
    if not destination.exists():
        raise RuntimeError("pip-tools did not create a safe lock file.")
    info = os.stat(destination, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or _is_link_or_reparse(info):
        raise RuntimeError("The generated lock is not a safe regular file.")
    if info.st_size <= 0 or info.st_size > _MAX_LOCK_BYTES:
        raise RuntimeError("The generated lock file is empty or unexpectedly large.")
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = build_parser().parse_args(_bounded_argv(argv))
        destination = generate_lock(
            input_path=Path(arguments.input),
            output_path=Path(arguments.output),
            upgrade=bool(arguments.upgrade),
        )
        if arguments.github_output:
            _write_github_output(destination)
        print(destination)
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError, TypeError, ValueError) as exc:
        print(f"lock generation failed: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
