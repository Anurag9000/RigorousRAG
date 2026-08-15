"""Durable owner-scoped storage for reports derived from authoritative result records."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from tools.models import Citation
from tools.research_report import EvidenceMatrixRow, ReportSection, ResearchReport
from tools.security import normalize_owner_id

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_REPORT_JSON_BYTES = 20_000_000


def _safe_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    if len(str(absolute)) > 4096:
        raise ValueError("research report database path is too long")
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or bool(attributes & _WINDOWS_REPARSE_POINT):
            raise RuntimeError("research report database path may not traverse symlinks/reparse points")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return absolute


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _text(value: str, label: str, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.replace("\x00", " ").strip()
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str)
    if len(encoded.encode("utf-8")) > _MAX_REPORT_JSON_BYTES:
        raise ValueError("research report exceeds the persistence size limit")
    return encoded


def _report_payload(report: ResearchReport) -> Mapping[str, Any]:
    return {
        "title": report.title,
        "question": report.question,
        "search_strategy": report.search_strategy,
        "sections": [asdict(item) for item in report.sections],
        "evidence_matrix": [asdict(item) for item in report.evidence_matrix],
        "citations": [item.model_dump(exclude_none=True) for item in report.citations],
        "conflicts": list(report.conflicts),
        "limitations": list(report.limitations),
        "warnings": list(report.warnings),
    }


def _report_from_payload(value: Mapping[str, Any]) -> ResearchReport:
    return ResearchReport(
        title=str(value["title"]),
        question=str(value["question"]),
        search_strategy=str(value["search_strategy"]),
        sections=tuple(ReportSection(**item) for item in value.get("sections", ())),
        evidence_matrix=tuple(EvidenceMatrixRow(**item) for item in value.get("evidence_matrix", ())),
        citations=tuple(Citation(**item) for item in value.get("citations", ())),
        conflicts=tuple(str(item) for item in value.get("conflicts", ())),
        limitations=tuple(str(item) for item in value.get("limitations", ())),
        warnings=tuple(str(item) for item in value.get("warnings", ())),
    )


@dataclass(frozen=True)
class StoredResearchReport:
    owner_id: str
    report_id: str
    result_id: str
    project_id: str
    report: ResearchReport
    created_at: float


class ResearchReportStore:
    def __init__(self, path: str | Path) -> None:
        self.path = _safe_path(path)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_reports (
                    owner_id TEXT NOT NULL,
                    report_id CHAR(64) NOT NULL,
                    result_id CHAR(64) NOT NULL,
                    project_id TEXT NOT NULL,
                    report_fingerprint CHAR(64) NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, report_id)
                );
                CREATE INDEX IF NOT EXISTS research_reports_owner_time_idx
                  ON research_reports(owner_id, created_at DESC, report_id);
                CREATE INDEX IF NOT EXISTS research_reports_owner_project_idx
                  ON research_reports(owner_id, project_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS research_reports_owner_result_idx
                  ON research_reports(owner_id, result_id, created_at DESC);
                """
            )

    def put(
        self,
        owner_id: str,
        *,
        result_id: str,
        project_id: str,
        report: ResearchReport,
    ) -> StoredResearchReport:
        owner = normalize_owner_id(owner_id)
        result = _sha(result_id, "result_id")
        project = _text(project_id, "project_id")
        if not isinstance(report, ResearchReport):
            raise TypeError("report must be ResearchReport")
        payload_json = _json(_report_payload(report))
        identity = {
            "owner_id": owner,
            "result_id": result,
            "project_id": project,
            "report_fingerprint": report.fingerprint,
        }
        report_id = hashlib.sha256(_json(identity).encode("utf-8")).hexdigest()
        created_at = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT payload_json,created_at FROM research_reports WHERE owner_id=? AND report_id=?",
                    (owner, report_id),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """INSERT INTO research_reports
                           (owner_id,report_id,result_id,project_id,report_fingerprint,payload_json,created_at)
                           VALUES(?,?,?,?,?,?,?)""",
                        (owner, report_id, result, project, report.fingerprint, payload_json, created_at),
                    )
                else:
                    created_at = float(existing["created_at"])
                    if str(existing["payload_json"]) != payload_json:
                        raise RuntimeError("research report identity collision")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return StoredResearchReport(owner, report_id, result, project, report, created_at)

    def get(self, owner_id: str, report_id: str) -> StoredResearchReport:
        owner = normalize_owner_id(owner_id)
        identifier = _sha(report_id, "report_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_reports WHERE owner_id=? AND report_id=?",
                (owner, identifier),
            ).fetchone()
        if row is None:
            raise KeyError(identifier)
        report = _report_from_payload(json.loads(str(row["payload_json"])))
        return StoredResearchReport(
            owner,
            identifier,
            _sha(str(row["result_id"]), "result_id"),
            str(row["project_id"]),
            report,
            float(row["created_at"]),
        )

    def list(self, owner_id: str, *, project_id: str | None = None, limit: int = 100) -> tuple[StoredResearchReport, ...]:
        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")
        with self._connect() as connection:
            if project_id is None:
                rows = connection.execute(
                    "SELECT report_id FROM research_reports WHERE owner_id=? ORDER BY created_at DESC,report_id LIMIT ?",
                    (owner, limit),
                ).fetchall()
            else:
                project = _text(project_id, "project_id")
                rows = connection.execute(
                    "SELECT report_id FROM research_reports WHERE owner_id=? AND project_id=? ORDER BY created_at DESC,report_id LIMIT ?",
                    (owner, project, limit),
                ).fetchall()
        return tuple(self.get(owner, str(row["report_id"])) for row in rows)


__all__ = ["ResearchReportStore", "StoredResearchReport"]
