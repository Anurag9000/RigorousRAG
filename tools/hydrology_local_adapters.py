"""Local, no-download hydrology adapters for CHIRPS metadata and HEC exports.

The adapters intentionally read operator-provided local artifacts only. HEC project text
is inspected conservatively; time-series evidence is loaded from a strict exported CSV
contract rather than attempting to execute HEC-HMS/HEC-RAS or reverse-engineer opaque
binary result stores.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.hydrology_domain import (
    CHIRPSAdapter,
    CRSRef,
    HECHMSAdapter,
    HECRASAdapter,
    HydroScenario,
    HydroTimeSeries,
    RasterWindowEvidence,
    TimeSeriesPoint,
)

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_FILE_BYTES = 500_000_000
_MAX_CSV_ROWS = 10_000_000


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json_load(payload: bytes) -> Any:
    if len(payload) > 10_000_000:
        raise ValueError("manifest exceeds the size limit")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON number {value!r} is forbidden")

    def unique_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key {key!r}")
            output[key] = value
        return output

    return json.loads(payload.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_pairs)


class LocalArtifactRoot:
    def __init__(self, root: str | Path) -> None:
        path = Path(root)
        if not path.is_absolute():
            path = Path.cwd() / path
        self.root = Path(os.path.abspath(path))
        self.root.mkdir(parents=True, exist_ok=True)
        self._validate_component(self.root)

    @staticmethod
    def _validate_component(path: Path) -> None:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise RuntimeError("hydrology artifact path could not be inspected") from exc
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or bool(attributes & _WINDOWS_REPARSE_POINT):
            raise RuntimeError("hydrology artifact path may not use symlinks/reparse points")

    def resolve(self, relative: str | Path, *, max_bytes: int = _MAX_FILE_BYTES) -> Path:
        raw = Path(relative)
        if raw.is_absolute():
            raise ValueError("hydrology artifact paths must be relative to the configured root")
        path = Path(os.path.abspath(self.root / raw))
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("hydrology artifact escapes the configured root") from exc
        current = self.root
        for part in path.relative_to(self.root).parts:
            current = current / part
            self._validate_component(current)
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
            raise ValueError("hydrology artifact is not a bounded regular file")
        return path

    def read_bytes(self, relative: str | Path, *, max_bytes: int = _MAX_FILE_BYTES) -> bytes:
        path = self.resolve(relative, max_bytes=max_bytes)
        with path.open("rb") as handle:
            payload = handle.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError("hydrology artifact exceeds the byte limit")
        return payload



def inspect_hec_text(payload: bytes) -> Mapping[str, Any]:
    """Conservative metadata extraction from text-like HEC project/plan artifacts."""
    if len(payload) > 20_000_000:
        raise ValueError("HEC project text exceeds the size limit")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = payload.decode("cp1252")
        except UnicodeDecodeError as exc:
            raise ValueError("HEC project metadata is not supported text") from exc
    lines = text.splitlines()
    fields: dict[str, list[str]] = {}
    for line in lines[:200_000]:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        separator = "=" if "=" in stripped else (":" if ":" in stripped else "")
        if not separator:
            continue
        key, value = stripped.split(separator, 1)
        key = " ".join(key.split())[:200]
        value = " ".join(value.split())[:2000]
        if not key or not value:
            continue
        fields.setdefault(key, []).append(value)
    public_fields = {
        key: values[:50]
        for key, values in fields.items()
        if key.casefold() in {
            "proj title", "project title", "current plan", "plan file", "plan title",
            "geom file", "geometry file", "flow file", "run", "run name", "basin",
            "meteorology", "control specifications", "compute interval", "computation interval",
            "program version", "version", "start date", "end date", "start time", "end time",
        }
    }
    return {
        "format": "hec-text-metadata-v1",
        "sha256": _sha_bytes(payload),
        "line_count": len(lines),
        "fields": public_fields,
    }


def load_hydrology_csv(
    payload: bytes,
    *,
    default_scenario_id: str = "",
    source_id: str,
) -> tuple[HydroTimeSeries, ...]:
    if len(payload) > _MAX_FILE_BYTES:
        raise ValueError("hydrology CSV exceeds the byte limit")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("hydrology CSV must be UTF-8") from exc
    reader = csv.DictReader(text.splitlines())
    required = {"timestamp", "location_id", "variable", "value", "unit"}
    fields = set(reader.fieldnames or ())
    if not required.issubset(fields):
        raise ValueError("hydrology CSV is missing required columns")
    unknown = fields - (required | {"quality", "scenario_id", "series_id"})
    if unknown:
        raise ValueError(f"hydrology CSV contains unsupported columns: {sorted(unknown)!r}")
    grouped: dict[tuple[str, str, str, str], list[TimeSeriesPoint]] = {}
    series_ids: dict[tuple[str, str, str, str], str] = {}
    row_count = 0
    for row_count, row in enumerate(reader, start=1):
        if row_count > _MAX_CSV_ROWS:
            raise ValueError("hydrology CSV exceeds the row limit")
        timestamp_raw = _text(row.get("timestamp", ""), "timestamp", 100)
        try:
            timestamp = dt.datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid timestamp at hydrology CSV row {row_count}") from exc
        variable = _text(row.get("variable", ""), "variable", 128).lower()
        location = _text(row.get("location_id", ""), "location_id", 256)
        unit = _text(row.get("unit", ""), "unit", 64)
        scenario = _text(row.get("scenario_id") or default_scenario_id or "default", "scenario_id", 256)
        value_raw = row.get("value")
        try:
            value = float(value_raw) if value_raw is not None else math.nan
        except ValueError as exc:
            raise ValueError(f"invalid numeric value at hydrology CSV row {row_count}") from exc
        if not math.isfinite(value):
            raise ValueError(f"non-finite numeric value at hydrology CSV row {row_count}")
        quality = _text(row.get("quality") or "simulated", "quality", 32).lower()
        key = (scenario, variable, location, unit)
        grouped.setdefault(key, []).append(TimeSeriesPoint(timestamp, value, quality))
        series_ids.setdefault(key, _text(row.get("series_id") or f"{scenario}:{variable}:{location}", "series_id", 256))
    output: list[HydroTimeSeries] = []
    for key in sorted(grouped):
        scenario, variable, location, unit = key
        points = tuple(sorted(grouped[key], key=lambda item: item.timestamp))
        output.append(HydroTimeSeries(series_ids[key], variable, unit, location, points, source_id, scenario))
    return tuple(output)


class _LocalHECExportBase:
    model_type: str

    def __init__(self, root: str | Path, catalog_path: str | Path) -> None:
        self.artifacts = LocalArtifactRoot(root)
        manifest_payload = self.artifacts.read_bytes(catalog_path, max_bytes=10_000_000)
        catalog = _strict_json_load(manifest_payload)
        if not isinstance(catalog, Mapping) or catalog.get("schema") != "rigorousrag-hec-export-catalog-v1":
            raise ValueError("HEC export catalog schema is invalid")
        scenarios = catalog.get("scenarios")
        if not isinstance(scenarios, list) or len(scenarios) > 10_000:
            raise ValueError("HEC export catalog scenarios are invalid")
        self._catalog = catalog
        self._by_name: dict[str, Mapping[str, Any]] = {}
        for item in scenarios:
            if not isinstance(item, Mapping):
                raise ValueError("HEC scenario catalog entry must be an object")
            if str(item.get("model_type", "")).casefold() != self.model_type:
                continue
            name = _text(item.get("name", ""), "scenario name", 256)
            if name in self._by_name:
                raise ValueError("duplicate HEC scenario name in catalog")
            self._by_name[name] = item

    def inspect_project(self, project_path: str) -> Mapping[str, Any]:
        payload = self.artifacts.read_bytes(project_path, max_bytes=20_000_000)
        result = dict(inspect_hec_text(payload))
        result["project_relative_path"] = _text(project_path, "project path", 1000)
        return result

    def _read(self, name: str) -> HydroScenario:
        item = self._by_name.get(_text(name, "scenario name", 256))
        if item is None:
            raise KeyError(name)
        project_rel = _text(item.get("project", ""), "project", 1000)
        csv_rel = _text(item.get("timeseries_csv", ""), "timeseries_csv", 1000)
        project_payload = self.artifacts.read_bytes(project_rel, max_bytes=20_000_000)
        csv_payload = self.artifacts.read_bytes(csv_rel, max_bytes=_MAX_FILE_BYTES)
        source_id = f"{self.model_type}:{name}:{_sha_bytes(csv_payload)}"
        series = load_hydrology_csv(csv_payload, default_scenario_id=name, source_id=source_id)
        selected_series = tuple(row for row in series if row.scenario_id == name)
        if not selected_series:
            raise ValueError("HEC export scenario contains no matching time series")
        start = min(row.points[0].timestamp for row in selected_series)
        end = max(row.points[-1].timestamp for row in selected_series)
        parameter_payload = item.get("parameters", {})
        parameter_sha = _sha_bytes(json.dumps(parameter_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8"))
        return HydroScenario(
            scenario_id=name,
            model_type=self.model_type,
            model_version=_text(item.get("model_version", "unknown"), "model_version", 100),
            project_sha256=_sha_bytes(project_payload),
            plan_name=_text(item.get("plan_name") or name, "plan_name", 500),
            start_time=start,
            end_time=end,
            parameters_sha256=parameter_sha,
            series=selected_series,
        )


class LocalHECHMSExportAdapter(_LocalHECExportBase, HECHMSAdapter):
    model_type = "hec-hms"

    def read_scenario(self, project_path: str, run_name: str) -> HydroScenario:
        item = self._by_name.get(_text(run_name, "run_name", 256))
        if item is None or _text(item.get("project", ""), "project", 1000) != _text(project_path, "project_path", 1000):
            raise KeyError(run_name)
        return self._read(run_name)


class LocalHECRASExportAdapter(_LocalHECExportBase, HECRASAdapter):
    model_type = "hec-ras"

    def read_scenario(self, project_path: str, plan_name: str) -> HydroScenario:
        item = self._by_name.get(_text(plan_name, "plan_name", 256))
        if item is None or _text(item.get("project", ""), "project", 1000) != _text(project_path, "project_path", 1000):
            raise KeyError(plan_name)
        return self._read(plan_name)

    def read_profile(self, project_path: str, plan_name: str, profile_name: str) -> Sequence[Mapping[str, Any]]:
        item = self._by_name.get(_text(plan_name, "plan_name", 256))
        if item is None or _text(item.get("project", ""), "project", 1000) != _text(project_path, "project_path", 1000):
            raise KeyError(plan_name)
        profiles = item.get("profiles", {})
        if not isinstance(profiles, Mapping):
            raise ValueError("HEC-RAS profiles catalog is invalid")
        relative = profiles.get(profile_name)
        if not isinstance(relative, str):
            raise KeyError(profile_name)
        payload = self.artifacts.read_bytes(relative, max_bytes=_MAX_FILE_BYTES)
        text = payload.decode("utf-8-sig")
        reader = csv.DictReader(text.splitlines())
        if not reader.fieldnames or len(reader.fieldnames) > 256:
            raise ValueError("HEC-RAS profile CSV has invalid headers")
        output: list[Mapping[str, Any]] = []
        for index, row in enumerate(reader, start=1):
            if index > _MAX_CSV_ROWS:
                raise ValueError("HEC-RAS profile CSV exceeds the row limit")
            output.append({str(key): str(value) for key, value in row.items()})
        return tuple(output)


class LocalCHIRPSManifestAdapter(CHIRPSAdapter):
    """Resolve an operator-provided CHIRPS window manifest plus local time-series CSV."""

    def __init__(self, root: str | Path, manifest_path: str | Path) -> None:
        self.artifacts = LocalArtifactRoot(root)
        manifest_payload = self.artifacts.read_bytes(manifest_path, max_bytes=10_000_000)
        manifest = _strict_json_load(manifest_payload)
        if not isinstance(manifest, Mapping) or manifest.get("schema") != "rigorousrag-chirps-local-v1":
            raise ValueError("CHIRPS local manifest schema is invalid")
        windows = manifest.get("windows")
        if not isinstance(windows, list) or len(windows) > 100_000:
            raise ValueError("CHIRPS window manifest is invalid")
        self._windows = tuple(windows)

    def rainfall_window(self, *, bbox: tuple[float, float, float, float], crs: CRSRef, start: dt.datetime, end: dt.datetime) -> tuple[RasterWindowEvidence, HydroTimeSeries]:
        target = tuple(float(value) for value in bbox)
        for item in self._windows:
            if not isinstance(item, Mapping):
                continue
            item_bbox = tuple(float(value) for value in item.get("bbox", ()))
            if len(item_bbox) != 4 or item_bbox != target:
                continue
            if _text(item.get("crs_authority", ""), "crs_authority", 32).upper() != crs.authority or _text(item.get("crs_code", ""), "crs_code", 64) != crs.code:
                continue
            item_start = dt.datetime.fromisoformat(_text(item.get("start", ""), "start", 64).replace("Z", "+00:00"))
            item_end = dt.datetime.fromisoformat(_text(item.get("end", ""), "end", 64).replace("Z", "+00:00"))
            if item_start != start and item_start.astimezone(dt.timezone.utc) != start.astimezone(dt.timezone.utc):
                continue
            if item_end != end and item_end.astimezone(dt.timezone.utc) != end.astimezone(dt.timezone.utc):
                continue
            raster_rel = _text(item.get("raster", ""), "raster", 1000)
            csv_rel = _text(item.get("timeseries_csv", ""), "timeseries_csv", 1000)
            raster_payload = self.artifacts.read_bytes(raster_rel, max_bytes=_MAX_FILE_BYTES)
            csv_payload = self.artifacts.read_bytes(csv_rel, max_bytes=_MAX_FILE_BYTES)
            source_id = f"chirps:{_sha_bytes(raster_payload)}"
            series = load_hydrology_csv(csv_payload, default_scenario_id="observed", source_id=source_id)
            rainfall = [row for row in series if row.variable in {"rainfall", "precipitation"}]
            if len(rainfall) != 1:
                raise ValueError("CHIRPS manifest window must resolve to exactly one rainfall series")
            evidence = RasterWindowEvidence(
                source_id=source_id,
                source_sha256=_sha_bytes(raster_payload),
                crs=crs,
                x_min=target[0], y_min=target[1], x_max=target[2], y_max=target[3],
                band=_text(item.get("band", "precipitation"), "band", 128),
                start_time=start,
                end_time=end,
                data_sha256=_sha_bytes(raster_payload),
                units=_text(item.get("units", rainfall[0].unit), "units", 64),
            )
            return evidence, rainfall[0]
        raise KeyError("no CHIRPS local window matches the requested extent/time range")


__all__ = [
    "LocalArtifactRoot",
    "LocalCHIRPSManifestAdapter",
    "LocalHECHMSExportAdapter",
    "LocalHECRASExportAdapter",
    "inspect_hec_text",
    "load_hydrology_csv",
]
