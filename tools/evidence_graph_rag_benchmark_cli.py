"""Strict local CLI for query-digest-only evidence-graph benchmarks."""

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

from tools.evidence_graph_rag_benchmark import fixture_from_mapping, run_graph_rag_benchmark

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_INPUT_BYTES = 256 * 1024 * 1024
_MAX_OUTPUT_BYTES = 256 * 1024 * 1024


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
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


def _read_object(value: str | os.PathLike[str]) -> Mapping[str, Any]:
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
    after = path.lstat()
    if (
        len(payload) > _MAX_INPUT_BYTES
        or _redirecting(after)
        or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        or after.st_size != before.st_size
    ):
        raise ValueError("fixture changed or exceeds the byte limit.")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("fixture must contain strict JSON.") from exc
    if not isinstance(value, Mapping):
        raise ValueError("fixture must contain a JSON object.")
    return value


def _write_atomic(path_value: str | os.PathLike[str], value: Any) -> None:
    path = _path(path_value, "output")
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.lstat()
    if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
        raise ValueError("output parent must be a directory.")
    if path.exists():
        info = path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise ValueError("output must be a regular file.")
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_OUTPUT_BYTES:
        raise ValueError("benchmark report exceeds the output byte limit.")
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _summary(fixture: Any) -> dict[str, Any]:
    return {
        "benchmark_id": fixture.benchmark_id,
        "benchmark_fingerprint": fixture.benchmark_fingerprint,
        "run_count": len(fixture.runs),
        "seed_count": len({value.seed for value in fixture.runs}),
        "case_count_per_run": len(fixture.runs[0].cases),
        "contains_raw_query": False,
        "contains_evidence_text": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_rag_benchmark_cli",
        description=(
            "Inspect or run strict query-digest-only evidence-graph benchmarks. "
            "Fixtures and reports contain no raw query or evidence text."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("fixture_file")
    run = commands.add_parser("run")
    run.add_argument("fixture_file")
    run.add_argument("--output-file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        fixture = fixture_from_mapping(_read_object(args.fixture_file))
        if args.command == "inspect":
            _print(_summary(fixture))
            return 0
        if args.command == "run":
            report = run_graph_rag_benchmark(fixture)
            payload = asdict(report)
            payload["report_digest"] = report.report_digest
            payload["contains_raw_query"] = False
            payload["contains_evidence_text"] = False
            if args.output_file:
                _write_atomic(args.output_file, payload)
            _print(payload)
            return 0
        raise ValueError("unsupported graph RAG benchmark command.")
    except (OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
