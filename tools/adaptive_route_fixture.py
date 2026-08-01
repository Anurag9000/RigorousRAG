"""Strict JSONL fixtures for reproducible adaptive route experiments."""

from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tools.adaptive_route_experiments import (
    ROUTES,
    RouteExecution,
    RouteExperimentCase,
)

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_FILE_BYTES = 50_000_000
_MAX_LINE_BYTES = 1_000_000
_MAX_CASES = 10_000
_MAX_EVIDENCE = 100
_ALLOWED_TOP_LEVEL = {"case", "routes"}
_ALLOWED_CASE = {"case_id", "query", "scope", "domain", "relevant_ids"}
_ALLOWED_EXECUTION = {"evidence", "cost_units", "latency_ms"}
_ALLOWED_EVIDENCE = {
    "chunk_id",
    "doc_id",
    "doi",
    "evidence_id",
    "generation_sequence",
    "page_number",
    "score",
    "source_id",
    "source_kind",
    "url",
    "metadata",
}
_ALLOWED_METADATA = {"evidence_kind", "fused_score", "generation_sequence", "relevance"}


def _redirecting(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        int(getattr(metadata, "st_file_attributes", 0)) & _WINDOWS_REPARSE_POINT
    )


def _safe_path(path: str | os.PathLike[str]) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError("fixture path must be a filesystem path.")
    rendered = os.fspath(path)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > 4_096
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("fixture path is invalid.")
    raw = Path(rendered)
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    absolute = Path(os.path.abspath(raw))
    for candidate in (absolute, *absolute.parents):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("fixture path could not be validated.") from exc
        if _redirecting(metadata):
            raise ValueError("fixture path may not contain symbolic links or reparse points.")
    return absolute


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        rendered = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(rendered) or not minimum <= rendered <= maximum:
        raise ValueError(f"{label} is outside the supported range.")
    return rendered


def _identifier(value: Any, label: str, maximum: int = 1_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if (
        not rendered
        or len(rendered) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError(f"{label} is invalid.")
    return rendered


def _keys(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {sorted(unknown)!r}.")
    return value


def _evidence(value: Any) -> dict[str, Any]:
    row = _keys(value, _ALLOWED_EVIDENCE, "evidence")
    result: dict[str, Any] = {}
    for key, item in row.items():
        if key in {"chunk_id", "doc_id", "doi", "evidence_id", "source_id", "source_kind", "url"}:
            result[key] = _identifier(item, key)
        elif key in {"generation_sequence", "page_number"}:
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise ValueError(f"{key} must be a positive integer.")
            result[key] = item
        elif key == "score":
            result[key] = _finite(item, "score", 0.0, 1.0)
        elif key == "metadata":
            metadata = _keys(item, _ALLOWED_METADATA, "evidence metadata")
            clean_metadata: dict[str, Any] = {}
            for metadata_key, metadata_value in metadata.items():
                if metadata_key == "evidence_kind":
                    clean_metadata[metadata_key] = _identifier(metadata_value, metadata_key, 100)
                elif metadata_key == "generation_sequence":
                    if (
                        isinstance(metadata_value, bool)
                        or not isinstance(metadata_value, int)
                        or metadata_value <= 0
                    ):
                        raise ValueError("metadata generation_sequence must be positive.")
                    clean_metadata[metadata_key] = metadata_value
                else:
                    clean_metadata[metadata_key] = _finite(
                        metadata_value, metadata_key, 0.0, 1.0
                    )
            result[key] = clean_metadata
    return result


@dataclass(frozen=True)
class RouteFixture:
    case: RouteExperimentCase
    executions: Mapping[str, RouteExecution]


def load_route_fixtures(path: str | os.PathLike[str]) -> tuple[RouteFixture, ...]:
    source = _safe_path(path)
    try:
        before = source.lstat()
    except OSError as exc:
        raise ValueError("fixture file is unavailable.") from exc
    if _redirecting(before) or not stat.S_ISREG(before.st_mode):
        raise ValueError("fixture file must be a regular non-redirected file.")
    if before.st_size > _MAX_FILE_BYTES:
        raise ValueError("fixture file exceeds the byte limit.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValueError("fixture file could not be opened safely.") from exc
    chunks: list[bytes] = []
    total = 0
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("fixture file changed before reading.")
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > _MAX_FILE_BYTES:
            raise ValueError("fixture file is not a bounded regular file.")
        while total <= _MAX_FILE_BYTES:
            chunk = os.read(descriptor, min(1_048_576, _MAX_FILE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > _MAX_FILE_BYTES:
            raise ValueError("fixture file exceeds the byte limit.")
        final_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = source.lstat()
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (opened.st_dev, opened.st_ino, opened.st_size)
        or (after.st_dev, after.st_ino, after.st_size)
        != (opened.st_dev, opened.st_ino, opened.st_size)
        or (final_descriptor.st_dev, final_descriptor.st_ino, final_descriptor.st_size)
        != (opened.st_dev, opened.st_ino, opened.st_size)
        or total != opened.st_size
    ):
        raise ValueError("fixture file changed during reading.")
    raw = b"".join(chunks)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("fixture file must contain UTF-8 JSONL.") from exc
    fixtures: list[RouteFixture] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > _MAX_LINE_BYTES:
            raise ValueError(f"fixture line {line_number} exceeds the byte limit.")
        if len(fixtures) >= _MAX_CASES:
            raise ValueError("fixture case limit exceeded.")
        try:
            payload = json.loads(
                line,
                object_pairs_hook=_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-standard JSON constant {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise ValueError(f"fixture line {line_number} is invalid JSON.") from exc
        top = _keys(payload, _ALLOWED_TOP_LEVEL, "fixture")
        case_data = _keys(top.get("case"), _ALLOWED_CASE, "case")
        relevant = case_data.get("relevant_ids", [])
        if not isinstance(relevant, list) or len(relevant) > 1_000:
            raise ValueError("case relevant_ids must be a bounded list.")
        case = RouteExperimentCase(
            case_id=case_data.get("case_id"),
            query=case_data.get("query"),
            scope=case_data.get("scope", "mixed"),
            domain=case_data.get("domain", "general"),
            relevant_ids=frozenset(_identifier(item, "relevant_id") for item in relevant),
        )
        if case.case_id in seen:
            raise ValueError(f"duplicate fixture case ID: {case.case_id}.")
        seen.add(case.case_id)
        route_data = top.get("routes")
        if type(route_data) is not dict or not route_data:
            raise ValueError("fixture routes must be a non-empty JSON object.")
        if len(route_data) > len(ROUTES):
            raise ValueError("fixture contains too many routes.")
        executions: dict[str, RouteExecution] = {}
        for route, raw_execution in route_data.items():
            if route not in ROUTES:
                raise ValueError(f"unsupported fixture route: {route}.")
            execution_data = _keys(raw_execution, _ALLOWED_EXECUTION, "route execution")
            raw_evidence = execution_data.get("evidence", [])
            if not isinstance(raw_evidence, list) or len(raw_evidence) > _MAX_EVIDENCE:
                raise ValueError("route evidence must be a bounded list.")
            executions[route] = RouteExecution(
                evidence=tuple(_evidence(item) for item in raw_evidence),
                cost_units=_finite(
                    execution_data.get("cost_units", 0.0),
                    "cost_units",
                    0.0,
                    1_000_000_000.0,
                ),
                latency_ms=_finite(
                    execution_data.get("latency_ms", 0.0),
                    "latency_ms",
                    0.0,
                    86_400_000.0,
                ),
            )
        fixtures.append(RouteFixture(case=case, executions=executions))
    return tuple(fixtures)


def fixture_adapters(
    fixtures: tuple[RouteFixture, ...],
) -> dict[str, Any]:
    if any(not isinstance(fixture, RouteFixture) for fixture in fixtures):
        raise ValueError("fixtures must contain RouteFixture values.")
    by_case = {fixture.case.case_id: fixture for fixture in fixtures}
    routes = sorted({route for fixture in fixtures for route in fixture.executions})
    adapters: dict[str, Any] = {}
    for route in routes:
        def adapter(case: RouteExperimentCase, selected_route: str = route) -> RouteExecution:
            fixture = by_case.get(case.case_id)
            if fixture is None:
                raise KeyError("route fixture case is unavailable")
            return fixture.executions.get(selected_route, RouteExecution(()))
        adapters[route] = adapter
    return adapters


__all__ = ["RouteFixture", "fixture_adapters", "load_route_fixtures"]
