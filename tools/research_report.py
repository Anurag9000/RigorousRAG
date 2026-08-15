"""Structured research-report, evidence-matrix and citation-chasing primitives.

Reports are assembled only from server-owned citations and already-governed claim/study
records.  Exporters do not ask a model to invent citations or bibliography entries.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from tools.claim_entailment import ClaimAssessment
from tools.models import Citation
from tools.scientific_synthesis import StudyEvidence

_MAX_ITEMS = 10_000


def _text(value: Any, label: str, maximum: int = 20_000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _citation_key(citation: Citation) -> str:
    payload = {
        "source": citation.source_id or citation.url,
        "doc": citation.doc_id or "",
        "chunk": citation.chunk_id or "",
        "page": citation.page_number,
        "quote": citation.quote or citation.snippet or "",
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


@dataclass(frozen=True)
class EvidenceMatrixRow:
    claim_id: str
    claim_text: str
    support_status: str
    study_id: str = ""
    population: str = ""
    intervention_or_exposure: str = ""
    comparator: str = ""
    outcome: str = ""
    result: str = ""
    uncertainty: str = ""
    limitation: str = ""
    citation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id", 256))
        object.__setattr__(self, "claim_text", _text(self.claim_text, "claim_text"))
        status = _text(self.support_status, "support_status", 32).lower()
        if status not in {"supported", "unsupported", "contradicted", "mixed", "unreviewed"}:
            raise ValueError("unsupported support_status")
        object.__setattr__(self, "support_status", status)
        for name in ("study_id", "population", "intervention_or_exposure", "comparator", "outcome", "result", "uncertainty", "limitation"):
            object.__setattr__(self, name, _text(getattr(self, name), name, 5000, allow_empty=True))
        if len(self.citation_ids) > 100:
            raise ValueError("citation_ids exceed the item limit")
        object.__setattr__(self, "citation_ids", tuple(dict.fromkeys(_text(item, "citation_id", 128) for item in self.citation_ids)))


@dataclass(frozen=True)
class ReportSection:
    heading: str
    body: str
    claim_ids: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "heading", _text(self.heading, "heading", 500))
        object.__setattr__(self, "body", _text(self.body, "body", 100_000, allow_empty=True))
        if len(self.claim_ids) > _MAX_ITEMS or len(self.citation_ids) > _MAX_ITEMS:
            raise ValueError("section references exceed the item limit")
        object.__setattr__(self, "claim_ids", tuple(dict.fromkeys(_text(item, "claim_id", 256) for item in self.claim_ids)))
        object.__setattr__(self, "citation_ids", tuple(dict.fromkeys(_text(item, "citation_id", 128) for item in self.citation_ids)))


@dataclass(frozen=True)
class ResearchReport:
    title: str
    question: str
    search_strategy: str
    sections: tuple[ReportSection, ...]
    evidence_matrix: tuple[EvidenceMatrixRow, ...]
    citations: tuple[Citation, ...]
    conflicts: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", _text(self.title, "title", 1000))
        object.__setattr__(self, "question", _text(self.question, "question", 20_000))
        object.__setattr__(self, "search_strategy", _text(self.search_strategy, "search_strategy", 20_000))
        if len(self.sections) > 256 or any(not isinstance(item, ReportSection) for item in self.sections):
            raise ValueError("sections are invalid")
        if len(self.evidence_matrix) > _MAX_ITEMS or any(not isinstance(item, EvidenceMatrixRow) for item in self.evidence_matrix):
            raise ValueError("evidence_matrix is invalid")
        if len(self.citations) > 100 or any(not isinstance(item, Citation) for item in self.citations):
            raise ValueError("citations are invalid")
        keys = [_citation_key(item) for item in self.citations]
        if len(keys) != len(set(keys)):
            raise ValueError("report citations must be deduplicated")
        for name in ("conflicts", "limitations", "warnings"):
            values = getattr(self, name)
            if len(values) > 1000:
                raise ValueError(f"{name} exceeds the item limit")
            object.__setattr__(self, name, tuple(_text(item, name, 5000) for item in values))

    @property
    def fingerprint(self) -> str:
        payload = {
            "title": self.title,
            "question": self.question,
            "search_strategy": self.search_strategy,
            "sections": [asdict(item) for item in self.sections],
            "evidence_matrix": [asdict(item) for item in self.evidence_matrix],
            "citations": [_citation_key(item) for item in self.citations],
            "conflicts": self.conflicts,
            "limitations": self.limitations,
            "warnings": self.warnings,
        }
        return hashlib.sha256(_canonical(payload)).hexdigest()


def build_evidence_matrix(
    claims: Sequence[ClaimAssessment],
    *,
    studies: Mapping[str, StudyEvidence] | None = None,
    claim_to_study: Mapping[str, str] | None = None,
) -> tuple[EvidenceMatrixRow, ...]:
    study_map = studies or {}
    linkage = claim_to_study or {}
    output: list[EvidenceMatrixRow] = []
    for assessment in claims:
        if assessment.contradicted:
            status = "contradicted"
        elif assessment.supported:
            status = "supported"
        else:
            status = "unsupported"
        study_id = linkage.get(assessment.claim.claim_id, "")
        study = study_map.get(study_id) if study_id else None
        output.append(
            EvidenceMatrixRow(
                claim_id=assessment.claim.claim_id,
                claim_text=assessment.claim.text,
                support_status=status,
                study_id=study_id,
                population=study.population if study else "",
                intervention_or_exposure=study.intervention_or_exposure if study else "",
                comparator=study.comparator if study else "",
                outcome=study.outcome if study else "",
                limitation="; ".join(study.limitations) if study else "",
                citation_ids=assessment.supporting_citation_ids + assessment.contradicting_citation_ids,
            )
        )
    return tuple(output)


def evidence_matrix_csv(rows: Sequence[EvidenceMatrixRow]) -> str:
    output = io.StringIO(newline="")
    fields = ["claim_id", "claim_text", "support_status", "study_id", "population", "intervention_or_exposure", "comparator", "outcome", "result", "uncertainty", "limitation", "citation_ids"]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        record = asdict(row)
        record["citation_ids"] = ";".join(row.citation_ids)
        writer.writerow(record)
    return output.getvalue()


def evidence_matrix_markdown(rows: Sequence[EvidenceMatrixRow]) -> str:
    headers = ("Claim", "Status", "Study", "Population", "Intervention/Exposure", "Comparator", "Outcome", "Citations")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = (
            row.claim_text,
            row.support_status,
            row.study_id,
            row.population,
            row.intervention_or_exposure,
            row.comparator,
            row.outcome,
            ", ".join(row.citation_ids),
        )
        escaped = [value.replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def report_markdown(report: ResearchReport) -> str:
    lines = [f"# {report.title}", "", f"**Research question:** {report.question}", "", "## Search strategy", report.search_strategy]
    for section in report.sections:
        lines.extend(["", f"## {section.heading}", section.body])
    if report.conflicts:
        lines.extend(["", "## Conflicting evidence", *[f"- {item}" for item in report.conflicts]])
    if report.limitations:
        lines.extend(["", "## Limitations", *[f"- {item}" for item in report.limitations]])
    lines.extend(["", "## Evidence matrix", evidence_matrix_markdown(report.evidence_matrix), "", "## Sources"])
    for index, citation in enumerate(report.citations, start=1):
        page = f", p. {citation.page_number}" if citation.page_number else ""
        lines.append(f"{index}. {citation.title}{page} — {citation.url}")
    if report.warnings:
        lines.extend(["", "## Warnings", *[f"- {item}" for item in report.warnings]])
    return "\n".join(lines).strip() + "\n"


@dataclass(frozen=True)
class ReferenceRecord:
    reference_id: str
    title: str
    doi: str = ""
    year: int | None = None
    raw_reference_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_id", _text(self.reference_id, "reference_id", 256))
        object.__setattr__(self, "title", _text(self.title, "reference title", 2000))
        object.__setattr__(self, "doi", _text(self.doi, "doi", 500, allow_empty=True))
        if self.year is not None and (isinstance(self.year, bool) or not isinstance(self.year, int) or not 1500 <= self.year <= 3000):
            raise ValueError("reference year is invalid")
        if self.raw_reference_sha256:
            digest = self.raw_reference_sha256.lower().strip()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("raw_reference_sha256 is invalid")
            object.__setattr__(self, "raw_reference_sha256", digest)


class CitationResolver(Protocol):
    def resolve(self, reference: ReferenceRecord) -> Sequence[Citation]: ...
    def cited_by(self, reference: ReferenceRecord, *, limit: int) -> Sequence[Citation]: ...


def chase_references(
    references: Sequence[ReferenceRecord],
    resolver: CitationResolver,
    *,
    forward: bool = False,
    max_results: int = 50,
) -> tuple[Citation, ...]:
    if len(references) > 1000 or not 1 <= max_results <= 500:
        raise ValueError("citation chase limits are invalid")
    output: list[Citation] = []
    seen: set[str] = set()
    for reference in references:
        try:
            candidates = resolver.cited_by(reference, limit=max_results) if forward else resolver.resolve(reference)
        except Exception:
            continue
        for citation in candidates:
            if not isinstance(citation, Citation):
                continue
            key = _citation_key(citation)
            if key in seen:
                continue
            seen.add(key)
            output.append(citation)
            if len(output) >= max_results:
                return tuple(output)
    return tuple(output)


__all__ = [
    "CitationResolver", "EvidenceMatrixRow", "ReferenceRecord", "ReportSection",
    "ResearchReport", "build_evidence_matrix", "chase_references", "evidence_matrix_csv",
    "evidence_matrix_markdown", "report_markdown",
]
