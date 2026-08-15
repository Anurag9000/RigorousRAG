"""Concrete bounded subprocess adapters for parser isolation and malware scanning.

These adapters use fixed argv templates, ``shell=False``, temporary private working
directories, minimal environments, timeouts and byte ceilings. They improve process
isolation but are explicitly *not* a substitute for an OS/container sandbox when
untrusted parser execution requires kernel-level isolation.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.production_runtime import ParserSandbox, ScanReceipt

_MAX_ARGV = 64
_MAX_ARG_LENGTH = 4096


def _text(value: Any, label: str, maximum: int = 1000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or "\x00" in cleaned:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _argv(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not 1 <= len(values) <= _MAX_ARGV:
        raise ValueError("command argv is invalid")
    return tuple(_text(item, "command argument", _MAX_ARG_LENGTH) for item in values)


def _minimal_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env: dict[str, str] = {}
    for name in ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TMP", "TEMP"):
        value = os.environ.get(name)
        if value:
            env[name] = value
    if extra:
        if len(extra) > 32:
            raise ValueError("extra environment exceeds the item limit")
        for key, value in extra.items():
            name = _text(key, "environment key", 128)
            if name.upper() in {"LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "PYTHONPATH", "PYTHONHOME"}:
                raise ValueError("unsafe process environment override")
            env[name] = _text(value, "environment value", 4096)
    return env


def _private_tempdir(prefix: str) -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(prefix=prefix)


@dataclass(frozen=True)
class SubprocessParserIsolation(ParserSandbox):
    """Execute a fixed parser executable against one private input path.

    The argv template may contain ``{input}``, ``{output}``, ``{filename}`` and
    ``{media_type}``. Parser stdout is ignored; output must be written to ``{output}``.
    """

    command_template: tuple[str, ...]
    extra_environment: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_template", _argv(self.command_template))
        placeholders = " ".join(self.command_template)
        if "{input}" not in placeholders or "{output}" not in placeholders:
            raise ValueError("parser command must reference {input} and {output}")
        _minimal_environment(self.extra_environment)

    def parse(
        self,
        payload: bytes,
        *,
        filename: str,
        media_type: str,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> bytes:
        if not isinstance(payload, bytes):
            raise ValueError("parser payload must be bytes")
        safe_filename = Path(_text(Path(filename).name, "filename", 500)).name
        safe_media_type = _text(media_type, "media_type", 200)
        timeout = float(timeout_seconds)
        if not 0.1 <= timeout <= 3600.0:
            raise ValueError("timeout_seconds is invalid")
        if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or not 1 <= max_output_bytes <= 2_000_000_000:
            raise ValueError("max_output_bytes is invalid")

        with _private_tempdir("rigorousrag-parser-") as temp_path:
            root = Path(temp_path)
            try:
                root.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            except OSError:
                pass
            input_path = root / "input.bin"
            output_path = root / "output.bin"
            with input_path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                input_path.chmod(stat.S_IRUSR)
            except OSError:
                pass
            replacements = {
                "{input}": str(input_path),
                "{output}": str(output_path),
                "{filename}": safe_filename,
                "{media_type}": safe_media_type,
            }
            argv: list[str] = []
            for raw in self.command_template:
                value = raw
                for token, replacement in replacements.items():
                    value = value.replace(token, replacement)
                argv.append(value)
            try:
                completed = subprocess.run(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=root,
                    env=_minimal_environment(self.extra_environment),
                    timeout=timeout,
                    check=False,
                    shell=False,
                    close_fds=(os.name != "nt"),
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("parser subprocess exceeded its deadline") from exc
            if completed.returncode != 0:
                raise RuntimeError(f"parser subprocess failed with exit code {completed.returncode}")
            if len(completed.stdout) > 1_000_000 or len(completed.stderr) > 1_000_000:
                raise RuntimeError("parser diagnostic output exceeded its limit")
            try:
                metadata = output_path.lstat()
            except OSError as exc:
                raise RuntimeError("parser did not create its expected output") from exc
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("parser output must be a regular non-symlink file")
            if metadata.st_size > max_output_bytes:
                raise RuntimeError("parser output exceeded its byte limit")
            with output_path.open("rb") as handle:
                result = handle.read(max_output_bytes + 1)
            if len(result) > max_output_bytes:
                raise RuntimeError("parser output exceeded its byte limit")
            return result


@dataclass(frozen=True)
class CommandMalwareScanner:
    """Fixed-command malware scanner adapter with explicit exit-code policy."""

    scanner_id: str
    engine_version: str
    command_template: tuple[str, ...]
    clean_exit_codes: tuple[int, ...] = (0,)
    infected_exit_codes: tuple[int, ...] = (1,)
    timeout_seconds: float = 60.0
    extra_environment: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scanner_id", _text(self.scanner_id, "scanner_id", 256))
        object.__setattr__(self, "engine_version", _text(self.engine_version, "engine_version", 128))
        object.__setattr__(self, "command_template", _argv(self.command_template))
        if "{input}" not in " ".join(self.command_template):
            raise ValueError("scanner command must reference {input}")
        clean = tuple(dict.fromkeys(int(code) for code in self.clean_exit_codes))
        infected = tuple(dict.fromkeys(int(code) for code in self.infected_exit_codes))
        if not clean or set(clean) & set(infected):
            raise ValueError("scanner exit-code policy is invalid")
        object.__setattr__(self, "clean_exit_codes", clean)
        object.__setattr__(self, "infected_exit_codes", infected)
        timeout = float(self.timeout_seconds)
        if not 0.1 <= timeout <= 3600.0:
            raise ValueError("timeout_seconds is invalid")
        object.__setattr__(self, "timeout_seconds", timeout)
        _minimal_environment(self.extra_environment)

    def scan(self, payload: bytes) -> ScanReceipt:
        if not isinstance(payload, bytes):
            raise ValueError("scanner payload must be bytes")
        digest = hashlib.sha256(payload).hexdigest()
        with _private_tempdir("rigorousrag-scan-") as temp_path:
            root = Path(temp_path)
            input_path = root / "sample.bin"
            with input_path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            argv = [part.replace("{input}", str(input_path)) for part in self.command_template]
            try:
                completed = subprocess.run(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=root,
                    env=_minimal_environment(self.extra_environment),
                    timeout=self.timeout_seconds,
                    check=False,
                    shell=False,
                    close_fds=(os.name != "nt"),
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("malware scanner exceeded its deadline") from exc
            if len(completed.stdout) > 1_000_000 or len(completed.stderr) > 1_000_000:
                raise RuntimeError("scanner diagnostic output exceeded its limit")
            if completed.returncode in self.clean_exit_codes:
                status = "clean"
            elif completed.returncode in self.infected_exit_codes:
                status = "infected"
            else:
                raise RuntimeError(f"scanner returned an unclassified exit code {completed.returncode}")
        return ScanReceipt(self.scanner_id, self.engine_version, digest, status, time.time())


__all__ = ["CommandMalwareScanner", "SubprocessParserIsolation"]
