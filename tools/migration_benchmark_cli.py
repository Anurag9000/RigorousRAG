"""CLI for producing promotion evidence from governed paired benchmark fixtures."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.migration_benchmark import fixture_from_mapping, run_promotion_benchmark

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_INPUT_BYTES = 50_000_000
_MAX_OUTPUT_BYTES = 10_000_000
_MAX_PATH = 4096


def _print(payload: Any, *, stream: Any = None) -> None:
    destination = stream if stream is not None else sys.stdout
    destination.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str], label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{label} must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError(f"{label} path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"{label} path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError(f"{label} path may not contain redirects.")
    return absolute


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_fixture(value: str | os.PathLike[str]) -> Mapping[str, Any]:
    path = _path(value, "fixture")
    before = path.lstat()
    if _redirecting(before) or not stat.S_ISREG(before.st_mode):
        raise ValueError("fixture must be a regular file.")
    if before.st_size <= 0 or before.st_size > _MAX_INPUT_BYTES:
        raise ValueError("fixture exceeds the byte limit.")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("fixture changed before reading.")
        payload = handle.read(_MAX_INPUT_BYTES + 1)
    if len(payload) > _MAX_INPUT_BYTES:
        raise ValueError("fixture exceeds the byte limit.")
    after = path.lstat()
    if (
        _redirecting(after)
        or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        or after.st_size != before.st_size
    ):
        raise ValueError("fixture changed during reading.")
    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("fixture must contain strict JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("fixture must contain a JSON object.")
    return parsed


def _encoded(value: Any) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    if not payload or len(payload) > _MAX_OUTPUT_BYTES:
        raise ValueError("benchmark output exceeds the byte limit.")
    return payload


def _atomic_output(value: str | os.PathLike[str], payload: bytes, label: str) -> Path:
    destination = _path(value, label)
    parent = destination.parent
    info = parent.lstat()
    if _redirecting(info) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} parent must be a regular directory.")
    if destination.exists():
        existing = destination.lstat()
        if _redirecting(existing) or not stat.S_ISREG(existing.st_mode):
            raise ValueError(f"{label} destination is invalid.")
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, raw = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=parent,
        )
        temporary = Path(raw)
        try:
            os.fchmod(descriptor, 0o600)
        except OSError:
            pass
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, destination)
        temporary = None
        try:
            parent_descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        except OSError:
            if os.name != "nt":
                raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.migration_benchmark_cli",
        description=(
            "Produce aggregate migration promotion evidence from strict paired "
            "query-ID-only fixtures. Raw query text and passages are unsupported."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run one paired benchmark fixture.")
    run.add_argument("--fixture-file", required=True)
    run.add_argument("--evidence-output", required=True)
    run.add_argument("--report-output")

    inspect = commands.add_parser(
        "inspect",
        help="Validate a fixture and print only its governed contract identity.",
    )
    inspect.add_argument("--fixture-file", required=True)
    return parser


def _run(args: argparse.Namespace) -> int:
    fixture = fixture_from_mapping(_read_fixture(args.fixture_file))
    result = run_promotion_benchmark(fixture)
    evidence_path = _atomic_output(
        args.evidence_output,
        _encoded(asdict(result.evidence)),
        "evidence output",
    )
    report_path = None
    if args.report_output:
        report_path = _atomic_output(
            args.report_output,
            _encoded(asdict(result)),
            "report output",
        )
    _print(
        {
            "task_id": result.evidence.task_id,
            "benchmark_fingerprint": result.evidence.benchmark_fingerprint,
            "evidence_digest": result.evidence.evidence_digest,
            "query_count": result.evidence.current_quality.query_count,
            "repeated_runs": result.evidence.repeated_runs,
            "seed_count": result.evidence.seed_count,
            "evidence_written": evidence_path.name,
            "report_written": report_path.name if report_path is not None else None,
        }
    )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    fixture = fixture_from_mapping(_read_fixture(args.fixture_file))
    _print(
        {
            "task_id": fixture.task_id,
            "benchmark_fingerprint": fixture.benchmark_fingerprint,
            "rank_cutoff": fixture.rank_cutoff,
            "query_count": len(fixture.runs[0].cases),
            "repeated_runs": len(fixture.runs),
            "seed_count": len({run.seed for run in fixture.runs}),
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "run":
            return _run(args)
        if args.command == "inspect":
            return _inspect(args)
        raise ValueError("unsupported migration benchmark command.")
    except (OSError, TypeError, ValueError, RuntimeError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
