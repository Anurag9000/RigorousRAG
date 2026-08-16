"""Deterministic report projections for governed hydrology evidence.

Hydrology evidence reports are not model-generated research results. They are immutable,
projection-bound tabular summaries of authoritative evidence identities and selection
traces. Exporters never invent citations, claims, coordinates, units, or source text.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.hydrology_projection import (
    HydrologyEvidenceProjection,
    HydrologyEvidenceRow,
    evidence_row_from_payload,
    evidence_row_payload,
)

_MAX_ROWS = 100_000
_MAX_TEXT = 20_000


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _digest(value: Any, label: str) -> str:
    cleaned = _text(value, label, 64).lower()
    if len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned):
        raise ValueError(f"{label} must be SHA-256")
    return cleaned


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _time(value: dt.datetime | None) -> str:
    return "" if value is None else value.isoformat()


def _spatial(value: Any) -> str:
    if value is None:
        return ""
    return f"{value.crs.authority}:{value.crs.code}[{value.crs.axis_order}] {value.min_x},{value.min_y},{value.max_x},{value.max_y}"


@dataclass(frozen=True)
class HydrologyReportSummary:
    row_count: int
    source_count: int
    scenario_count: int
    node_count: int
    reach_count: int
    modalities: tuple[tuple[str, int], ...]
    variables: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for name in ("row_count", "source_count", "scenario_count", "node_count", "reach_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} is invalid")
        for name in ("modalities", "variables"):
            values = getattr(self, name)
            if len(values) > _MAX_ROWS:
                raise ValueError(f"{name} exceeds the item limit")
            normalized: list[tuple[str, int]] = []
            for key, count in values:
                key_value = _text(key, name, 256, allow_empty=name == "variables")
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(f"{name} count is invalid")
                normalized.append((key_value, count))
            object.__setattr__(self, name, tuple(normalized))


@dataclass(frozen=True)
class HydrologyEvidenceReport:
    report_id: str
    project_id: str
    title: str
    research_question: str
    projection_fingerprint: str
    package_fingerprint: str
    topology_fingerprint: str
    plan_fingerprint: str
    index_fingerprint: str
    summary: HydrologyReportSummary
    rows: tuple[HydrologyEvidenceRow, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "report_id", _text(self.report_id, "report_id", 500))
        object.__setattr__(self, "project_id", _text(self.project_id, "project_id", 256))
        object.__setattr__(self, "title", _text(self.title, "title", 1000))
        object.__setattr__(self, "research_question", _text(self.research_question, "research_question", _MAX_TEXT, allow_empty=True))
        for name in ("projection_fingerprint", "package_fingerprint", "topology_fingerprint", "plan_fingerprint", "index_fingerprint"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if not isinstance(self.summary, HydrologyReportSummary):
            raise TypeError("summary must be HydrologyReportSummary")
        if len(self.rows) > _MAX_ROWS or any(not isinstance(item, HydrologyEvidenceRow) for item in self.rows):
            raise ValueError("report rows are invalid")
        if self.summary.row_count != len(self.rows):
            raise ValueError("report summary row_count does not match rows")
        object.__setattr__(self, "diagnostics", tuple(dict.fromkeys(_text(item, "diagnostic", 2000) for item in self.diagnostics)))

    @property
    def fingerprint(self) -> str:
        payload = {
            "report_id": self.report_id,
            "project_id": self.project_id,
            "title": self.title,
            "research_question": self.research_question,
            "projection_fingerprint": self.projection_fingerprint,
            "package_fingerprint": self.package_fingerprint,
            "topology_fingerprint": self.topology_fingerprint,
            "plan_fingerprint": self.plan_fingerprint,
            "index_fingerprint": self.index_fingerprint,
            "summary": asdict(self.summary),
            "rows": [asdict(item) for item in self.rows],
            "diagnostics": list(self.diagnostics),
        }
        return hashlib.sha256(_canonical(payload)).hexdigest()

    @property
    def complete(self) -> bool:
        return not any(item.startswith("fatal:") for item in self.diagnostics)


def _summary(rows: Sequence[HydrologyEvidenceRow]) -> HydrologyReportSummary:
    modality = Counter(item.modality for item in rows)
    variables = Counter(item.variable for item in rows if item.variable)
    return HydrologyReportSummary(
        row_count=len(rows),
        source_count=len({item.source_id for item in rows}),
        scenario_count=len({item.scenario_id for item in rows if item.scenario_id}),
        node_count=len({item.topology_id for item in rows if item.topology_kind == "node"}),
        reach_count=len({item.topology_id for item in rows if item.topology_kind == "reach"}),
        modalities=tuple(sorted(modality.items())),
        variables=tuple(sorted(variables.items())),
    )


def build_hydrology_report(
    projection: HydrologyEvidenceProjection,
    *,
    report_id: str,
    project_id: str,
    title: str,
    research_question: str = "",
) -> HydrologyEvidenceReport:
    if not isinstance(projection, HydrologyEvidenceProjection):
        raise TypeError("projection must be HydrologyEvidenceProjection")
    return HydrologyEvidenceReport(
        report_id=report_id,
        project_id=project_id,
        title=title,
        research_question=research_question,
        projection_fingerprint=projection.fingerprint,
        package_fingerprint=projection.package_fingerprint,
        topology_fingerprint=projection.topology_fingerprint,
        plan_fingerprint=projection.plan_fingerprint,
        index_fingerprint=projection.index_fingerprint,
        summary=_summary(projection.rows),
        rows=projection.rows,
        diagnostics=tuple(dict.fromkeys((*projection.package_diagnostics, *projection.plan_unresolved))),
    )


def report_payload(report: HydrologyEvidenceReport) -> Mapping[str, Any]:
    if not isinstance(report, HydrologyEvidenceReport):
        raise TypeError("report must be HydrologyEvidenceReport")
    return {
        "report_id": report.report_id,
        "project_id": report.project_id,
        "title": report.title,
        "research_question": report.research_question,
        "projection_fingerprint": report.projection_fingerprint,
        "package_fingerprint": report.package_fingerprint,
        "topology_fingerprint": report.topology_fingerprint,
        "plan_fingerprint": report.plan_fingerprint,
        "index_fingerprint": report.index_fingerprint,
        "summary": {
            "row_count": report.summary.row_count,
            "source_count": report.summary.source_count,
            "scenario_count": report.summary.scenario_count,
            "node_count": report.summary.node_count,
            "reach_count": report.summary.reach_count,
            "modalities": [list(item) for item in report.summary.modalities],
            "variables": [list(item) for item in report.summary.variables],
        },
        "rows": [evidence_row_payload(item) for item in report.rows],
        "diagnostics": list(report.diagnostics),
        "fingerprint": report.fingerprint,
    }


def report_from_payload(value: Any) -> HydrologyEvidenceReport:
    if not isinstance(value, Mapping) or not isinstance(value.get("rows"), list) or not isinstance(value.get("summary"), Mapping):
        raise ValueError("hydrology evidence report payload is invalid")
    if len(value["rows"]) > _MAX_ROWS:
        raise ValueError("hydrology evidence report rows exceed the item limit")
    rows = tuple(evidence_row_from_payload(item) for item in value["rows"])
    raw = value["summary"]
    report = HydrologyEvidenceReport(
        report_id=str(value["report_id"]),
        project_id=str(value["project_id"]),
        title=str(value["title"]),
        research_question=str(value.get("research_question", "")),
        projection_fingerprint=str(value["projection_fingerprint"]),
        package_fingerprint=str(value["package_fingerprint"]),
        topology_fingerprint=str(value["topology_fingerprint"]),
        plan_fingerprint=str(value["plan_fingerprint"]),
        index_fingerprint=str(value["index_fingerprint"]),
        summary=HydrologyReportSummary(
            row_count=int(raw["row_count"]),
            source_count=int(raw["source_count"]),
            scenario_count=int(raw["scenario_count"]),
            node_count=int(raw["node_count"]),
            reach_count=int(raw["reach_count"]),
            modalities=tuple((str(item[0]), int(item[1])) for item in raw.get("modalities", ())),
            variables=tuple((str(item[0]), int(item[1])) for item in raw.get("variables", ())),
        ),
        rows=rows,
        diagnostics=tuple(value.get("diagnostics") or ()),
    )
    if report.fingerprint != _digest(value.get("fingerprint"), "report fingerprint"):
        raise RuntimeError("stored hydrology evidence report failed integrity check")
    if report.summary != _summary(report.rows):
        raise RuntimeError("stored hydrology evidence report summary does not match rows")
    return report


def report_csv(report: HydrologyEvidenceReport) -> str:
    if not isinstance(report, HydrologyEvidenceReport):
        raise TypeError("report must be HydrologyEvidenceReport")
    output = io.StringIO(newline="")
    fields = (
        "record_id", "source_id", "content_sha256", "variable", "modality", "scenario_id",
        "topology_kind", "topology_id", "start_time", "end_time", "spatial_scope", "selection_reasons",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in report.rows:
        writer.writerow({
            "record_id": row.record_id,
            "source_id": row.source_id,
            "content_sha256": row.content_sha256,
            "variable": row.variable,
            "modality": row.modality,
            "scenario_id": row.scenario_id,
            "topology_kind": row.topology_kind,
            "topology_id": row.topology_id,
            "start_time": _time(row.start_time),
            "end_time": _time(row.end_time),
            "spatial_scope": _spatial(row.spatial),
            "selection_reasons": ";".join(row.selection_reasons),
        })
    return output.getvalue()


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def report_markdown(report: HydrologyEvidenceReport) -> str:
    if not isinstance(report, HydrologyEvidenceReport):
        raise TypeError("report must be HydrologyEvidenceReport")
    summary = report.summary
    lines = [
        f"# {report.title}",
        "",
        "## Provenance",
        f"- Project: `{report.project_id}`",
        f"- Report fingerprint: `{report.fingerprint}`",
        f"- Projection fingerprint: `{report.projection_fingerprint}`",
        f"- Engineering package fingerprint: `{report.package_fingerprint}`",
        f"- Topology fingerprint: `{report.topology_fingerprint}`",
        f"- Retrieval plan fingerprint: `{report.plan_fingerprint}`",
        f"- Index fingerprint: `{report.index_fingerprint}`",
        "",
        "## Deterministic evidence summary",
        f"- Selected records: {summary.row_count}",
        f"- Distinct sources: {summary.source_count}",
        f"- Distinct scenarios: {summary.scenario_count}",
        f"- Selected nodes: {summary.node_count}",
        f"- Selected reaches: {summary.reach_count}",
    ]
    if report.research_question:
        lines.extend(["", "## Research question", report.research_question])
    if summary.modalities:
        lines.extend(["", "### Modalities", *[f"- {name}: {count}" for name, count in summary.modalities]])
    if summary.variables:
        lines.extend(["", "### Variables", *[f"- {name}: {count}" for name, count in summary.variables]])
    if report.diagnostics:
        lines.extend(["", "## Diagnostics", *[f"- {item}" for item in report.diagnostics]])
    lines.extend([
        "",
        "## Evidence records",
        "| Record | Source | Variable | Modality | Scenario | Topology | Time | Selection reasons |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in report.rows:
        time_range = "" if row.start_time is None else f"{row.start_time.isoformat()} – {row.end_time.isoformat()}"
        values = (
            row.record_id, row.source_id, row.variable, row.modality, row.scenario_id,
            f"{row.topology_kind}:{row.topology_id}", time_range, "; ".join(row.selection_reasons),
        )
        lines.append("| " + " | ".join(_escape(item) for item in values) + " |")
    lines.extend([
        "",
        "> This is a deterministic hydrology evidence projection, not a model-generated scientific conclusion. "
        "Record selection and provenance are reported exactly as stored; unresolved/fatal diagnostics remain visible.",
    ])
    return "\n".join(lines).strip() + "\n"


__all__ = [
    "HydrologyEvidenceReport",
    "HydrologyReportSummary",
    "build_hydrology_report",
    "report_csv",
    "report_from_payload",
    "report_markdown",
    "report_payload",
]
