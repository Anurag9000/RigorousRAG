"""Generate one hashed, platform-specific runtime dependency lock.

Resolution runs against an immutable bounded snapshot of the checked requirements file and
publishes through an atomic private-file replacement. The generator intentionally targets
public PyPI and rejects requirement-file directives that could silently change package
authority or reopen unsnapshotted local files.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import secrets
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

_MAX_ARGUMENTS = 50
_MAX_PATH_CHARS = 4096
_MAX_INPUT_BYTES = 1_000_000
_MAX_LOCK_BYTES = 20_000_000
_MAX_GITHUB_OUTPUT_BYTES = 1_000_000
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
_PUBLIC_INDEX_URL = "https://pypi.org/simple"
_AMBIENT_AUTHORITY_NAMES = frozenset(
    {
        "ALL_PROXY",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "PYTHONHOME",
        "PYTHONPATH",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
    }
)


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
    try:
        rendered = os.fspath(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a filesystem path.") from exc
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


def _identity(info: os.stat_result) -> tuple[int, int]:
    return int(info.st_dev), int(info.st_ino)


def _directory_identity(path: Path, *, label: str) -> tuple[int, int]:
    _require_safe_ancestry(path, label=label)
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable.") from exc
    if not stat.S_ISDIR(info.st_mode) or _is_link_or_reparse(info):
        raise ValueError(f"{label} must be a safe directory.")
    return _identity(info)


def _open_read_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOINHERIT", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("A lock file write made no progress.")
        offset += written


def _read_regular_bytes(path: Path, *, label: str, maximum: int) -> bytes:
    """Read a bounded regular file while verifying one stable identity."""

    absolute = _safe_path(path, label=label)
    _require_safe_ancestry(absolute, label=label)
    try:
        before = os.lstat(absolute)
    except OSError as exc:
        raise ValueError(f"{label} must be a safe regular file.") from exc
    if _is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a safe regular file.")
    if before.st_size <= 0 or before.st_size > maximum:
        raise ValueError(f"{label} is empty or exceeds its byte limit.")
    expected = _identity(before)

    descriptor = -1
    try:
        descriptor = os.open(absolute, _open_read_flags())
        opened = os.fstat(descriptor)
        if (
            _is_link_or_reparse(opened)
            or not stat.S_ISREG(opened.st_mode)
            or _identity(opened) != expected
        ):
            raise ValueError(f"{label} identity changed while it was being opened.")
        data = bytearray()
        while True:
            remaining = maximum + 1 - len(data)
            if remaining <= 0:
                raise ValueError(f"{label} exceeds its byte limit.")
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > maximum:
                raise ValueError(f"{label} exceeds its byte limit.")
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    _require_safe_ancestry(absolute, label=label)
    try:
        after = os.lstat(absolute)
    except OSError as exc:
        raise ValueError(f"{label} changed while it was being read.") from exc
    if (
        _is_link_or_reparse(after)
        or not stat.S_ISREG(after.st_mode)
        or _identity(after) != expected
    ):
        raise ValueError(f"{label} changed while it was being read.")
    return bytes(data)


def _validate_requirements_source(payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("The requirements input must be valid UTF-8.") from exc
    if any(
        (ord(character) < 32 and character not in "\t\r\n") or ord(character) == 127
        for character in text
    ):
        raise ValueError("The requirements input contains unsupported control characters.")
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lowered = stripped.lower()
        if stripped.startswith("-"):
            raise ValueError(
                "The requirements input may not contain resolver options or include another file."
            )
        if (
            "://" in lowered
            or "/" in stripped
            or "\\" in stripped
            or lowered.startswith(("git+", "file:", "./", "../", "~/"))
            or re.match(r"^[a-z]:[\\/]", lowered)
        ):
            raise ValueError("The requirements input may not contain URL or local-path requirements.")


def _exclusive_write(path: Path, payload: bytes, *, label: str) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOINHERIT", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(path, flags, 0o600)
        opened = os.fstat(descriptor)
        if _is_link_or_reparse(opened) or not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} must be a private regular file.")
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _destination_identity(path: Path) -> tuple[int, int] | None:
    if not (path.exists() or path.is_symlink()):
        return None
    info = os.lstat(path)
    if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
        raise ValueError("The lock output must be absent or a safe regular file.")
    return _identity(info)


def _assert_expected_destination(
    destination: Path,
    expected_identity: tuple[int, int] | None,
) -> None:
    exists = destination.exists() or destination.is_symlink()
    if expected_identity is None:
        if exists:
            raise ValueError("The lock output appeared unexpectedly during generation.")
        return
    if not exists:
        raise ValueError("The existing lock output disappeared during generation.")
    info = os.lstat(destination)
    if (
        _is_link_or_reparse(info)
        or not stat.S_ISREG(info.st_mode)
        or _identity(info) != expected_identity
    ):
        raise ValueError("The existing lock output identity changed during generation.")


def _publish_lock(
    destination: Path,
    payload: bytes,
    *,
    expected_parent_identity: tuple[int, int],
    expected_destination_identity: tuple[int, int] | None,
) -> Path:
    """Publish verified lock bytes through an atomic private-file replacement."""

    if not payload or len(payload) > _MAX_LOCK_BYTES:
        raise ValueError("The generated lock is empty or exceeds the 20 MB limit.")
    parent = destination.parent
    if _directory_identity(parent, label="lock output parent") != expected_parent_identity:
        raise ValueError("The lock output parent identity changed during generation.")
    _assert_expected_destination(destination, expected_destination_identity)

    temporary = parent / f".{destination.name}.{secrets.token_hex(16)}.tmp"
    try:
        _exclusive_write(temporary, payload, label="temporary lock output")
        if _directory_identity(parent, label="lock output parent") != expected_parent_identity:
            raise ValueError("The lock output parent identity changed before publication.")
        _assert_expected_destination(destination, expected_destination_identity)
        os.replace(temporary, destination)
        _require_safe_ancestry(destination, label="published lock")
        final = os.lstat(destination)
        if _is_link_or_reparse(final) or not stat.S_ISREG(final.st_mode):
            raise RuntimeError("The published lock is not a safe regular file.")
        if final.st_size != len(payload):
            raise RuntimeError("The published lock size differs from the verified bytes.")
        return destination
    finally:
        try:
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
        except OSError:
            pass


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


def _safe_subprocess_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PIP_")
        and key.upper() not in _AMBIENT_AUTHORITY_NAMES
    }
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PIP_NO_INPUT"] = "1"
    environment["PIP_NO_CACHE_DIR"] = "1"
    environment["PIP_KEYRING_PROVIDER"] = "disabled"
    environment["PIP_CONFIG_FILE"] = os.devnull
    return environment


def _append_github_output(path: Path, payload: bytes) -> None:
    """Append one bounded output payload without following a final symlink."""

    parent_identity = _directory_identity(path.parent, label="GITHUB_OUTPUT parent")
    flags = (
        os.O_WRONLY
        | os.O_APPEND
        | os.O_CREAT
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOINHERIT", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if path.exists() or path.is_symlink():
        before = os.lstat(path)
        if _is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
            raise ValueError("GITHUB_OUTPUT must be a safe regular file.")
        if before.st_size + len(payload) > _MAX_GITHUB_OUTPUT_BYTES:
            raise ValueError("GITHUB_OUTPUT exceeds the 1 MB limit.")
        expected = _identity(before)
    else:
        expected = None

    descriptor = -1
    try:
        descriptor = os.open(path, flags, 0o600)
        opened = os.fstat(descriptor)
        opened_identity = _identity(opened)
        if _is_link_or_reparse(opened) or not stat.S_ISREG(opened.st_mode):
            raise ValueError("GITHUB_OUTPUT must be a safe regular file.")
        path_entry = os.lstat(path)
        if (
            _is_link_or_reparse(path_entry)
            or not stat.S_ISREG(path_entry.st_mode)
            or _identity(path_entry) != opened_identity
        ):
            raise ValueError("GITHUB_OUTPUT path is unsafe before append.")
        if expected is not None and opened_identity != expected:
            raise ValueError("GITHUB_OUTPUT identity changed while it was being opened.")
        if opened.st_size + len(payload) > _MAX_GITHUB_OUTPUT_BYTES:
            raise ValueError("GITHUB_OUTPUT exceeds the 1 MB limit.")
        if _directory_identity(path.parent, label="GITHUB_OUTPUT parent") != parent_identity:
            raise ValueError("GITHUB_OUTPUT parent identity changed before append.")
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        after = os.lstat(path)
        if (
            _is_link_or_reparse(after)
            or not stat.S_ISREG(after.st_mode)
            or _identity(after) != opened_identity
        ):
            raise ValueError("GITHUB_OUTPUT identity changed during append.")
        if _directory_identity(path.parent, label="GITHUB_OUTPUT parent") != parent_identity:
            raise ValueError("GITHUB_OUTPUT parent identity changed during append.")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_github_output(destination: Path) -> None:
    """Publish the generated absolute path and artifact name to GitHub Actions."""

    _read_regular_bytes(
        destination,
        label="generated lock output",
        maximum=_MAX_LOCK_BYTES,
    )
    output_value = os.environ.get("GITHUB_OUTPUT")
    if not output_value:
        raise RuntimeError("GITHUB_OUTPUT is unavailable.")
    output_path = _safe_path(output_value, label="GITHUB_OUTPUT path")
    _require_safe_ancestry(output_path.parent, label="GITHUB_OUTPUT parent")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _require_safe_ancestry(output_path, label="GITHUB_OUTPUT path")

    rendered_path = destination.as_posix()
    rendered_name = destination.stem
    if (
        _contains_ascii_control(rendered_path)
        or _contains_ascii_control(rendered_name)
        or len(rendered_path) > _MAX_PATH_CHARS
        or len(rendered_name) > 255
    ):
        raise ValueError("Generated workflow output is invalid or too long.")
    payload = f"path={rendered_path}\nname={rendered_name}\n".encode("utf-8")
    _append_github_output(output_path, payload)


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
    source_bytes = _read_regular_bytes(
        source,
        label="requirements input",
        maximum=_MAX_INPUT_BYTES,
    )
    _validate_requirements_source(source_bytes)
    if destination == source:
        raise ValueError("The lock output may not replace the requirements input.")

    _require_safe_ancestry(destination.parent, label="lock output parent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_parent_identity = _directory_identity(
        destination.parent,
        label="lock output parent",
    )
    _require_safe_ancestry(destination, label="lock output")
    expected_destination_identity = _destination_identity(destination)

    staging = Path(
        tempfile.mkdtemp(prefix=".rigorousrag-lock-", dir=destination.parent)
    )
    staging_info = os.lstat(staging)
    staging_identity = _identity(staging_info)
    if _is_link_or_reparse(staging_info) or not stat.S_ISDIR(staging_info.st_mode):
        raise RuntimeError("The private lock staging directory is unsafe.")
    snapshot = staging / "requirements.in"
    compiled = staging / "compiled.lock"
    try:
        _exclusive_write(snapshot, source_bytes, label="requirements snapshot")
        command = [
            str(_pip_compile_executable()),
            str(snapshot),
            "--resolver=backtracking",
            "--generate-hashes",
            "--allow-unsafe",
            "--no-header",
            "--no-annotate",
            "--index-url",
            _PUBLIC_INDEX_URL,
            "--no-emit-index-url",
            "--no-emit-trusted-host",
            "--output-file",
            str(compiled),
        ]
        if upgrade:
            command.append("--upgrade")

        subprocess.run(
            command,
            check=True,
            env=_safe_subprocess_environment(),
        )
        compiled_bytes = _read_regular_bytes(
            compiled,
            label="generated lock",
            maximum=_MAX_LOCK_BYTES,
        )
        try:
            compiled_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("The generated lock must be valid UTF-8.") from exc
        return _publish_lock(
            destination,
            compiled_bytes,
            expected_parent_identity=expected_parent_identity,
            expected_destination_identity=expected_destination_identity,
        )
    finally:
        try:
            current = os.lstat(staging)
            if (
                stat.S_ISDIR(current.st_mode)
                and not _is_link_or_reparse(current)
                and _identity(current) == staging_identity
            ):
                shutil.rmtree(staging)
        except OSError:
            pass


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
