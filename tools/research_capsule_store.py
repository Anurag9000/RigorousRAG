"""Durable immutable owner-scoped storage for research capsules.

Capsules contain content identities and replay structure, not decrypted replay queries or
raw private evidence. The store verifies the full manifest fingerprint whenever data is
read and models supersession as an append-only relation between immutable capsules.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from tools.research_capsule import CapsuleReference, ReplayStep, ResearchCapsule
from tools.security import normalize_owner_id

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_MANIFEST_BYTES = 8_000_000


def _safe_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    if len(str(absolute)) > 4096:
        raise ValueError("research capsule database path is too long")
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or bool(attributes & _WINDOWS_REPARSE_POINT):
            raise RuntimeError("research capsule path may not traverse symlinks/reparse points")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return absolute


def _text(value: Any, label: str, maximum: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.replace("\x00", " ").strip()
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha(value: Any, label: str) -> str:
    digest = _text(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _canonical(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > _MAX_MANIFEST_BYTES:
        raise ValueError("research capsule manifest exceeds the size limit")
    return encoded


def _capsule_payload(capsule: ResearchCapsule) -> Mapping[str, Any]:
    payload = asdict(capsule)
    payload["fingerprint"] = capsule.fingerprint
    return payload


def _capsule_from_json(raw: str) -> ResearchCapsule:
    if len(raw.encode("utf-8")) > _MAX_MANIFEST_BYTES:
        raise RuntimeError("stored research capsule manifest exceeds the size limit")
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise RuntimeError("stored research capsule manifest is invalid")
    expected = _sha(value.get("fingerprint"), "fingerprint")
    references_raw = value.get("references", ())
    steps_raw = value.get("replay_steps", ())
    if not isinstance(references_raw, list) or not isinstance(steps_raw, list):
        raise RuntimeError("stored research capsule collections are invalid")
    references = tuple(CapsuleReference(**item) for item in references_raw)
    replay_steps = tuple(
        ReplayStep(
            step_id=item["step_id"],
            operation=item["operation"],
            input_ref_ids=tuple(item.get("input_ref_ids", ())),
            output_ref_ids=tuple(item.get("output_ref_ids", ())),
            capability_ref_id=item.get("capability_ref_id", ""),
            policy_ref_id=item.get("policy_ref_id", ""),
            deterministic=item.get("deterministic", False),
            seed=item.get("seed"),
        )
        for item in steps_raw
    )
    capsule = ResearchCapsule(
        capsule_id=value["capsule_id"],
        project_id=value["project_id"],
        run_id=value["run_id"],
        code_revision=value["code_revision"],
        references=references,
        replay_steps=replay_steps,
        created_at=float(value["created_at"]),
        schema_version=value.get("schema_version", "1.0.0"),
        notes=tuple(value.get("notes", ())),
    )
    if capsule.fingerprint != expected:
        raise RuntimeError("research capsule manifest fingerprint mismatch")
    return capsule


@dataclass(frozen=True)
class StoredResearchCapsule:
    owner_id: str
    project_id: str
    session_id: str
    result_id: str
    capsule: ResearchCapsule
    supersedes_capsule_id: str

    @property
    def capsule_id(self) -> str:
        return self.capsule.capsule_id

    @property
    def fingerprint(self) -> str:
        return self.capsule.fingerprint

    @property
    def created_at(self) -> float:
        return self.capsule.created_at


class ResearchCapsuleStore:
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
                CREATE TABLE IF NOT EXISTS research_capsules (
                    owner_id TEXT NOT NULL,
                    capsule_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    result_id CHAR(64) NOT NULL,
                    fingerprint CHAR(64) NOT NULL,
                    manifest_json TEXT NOT NULL,
                    supersedes_capsule_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, capsule_id),
                    UNIQUE(owner_id, fingerprint)
                );
                CREATE INDEX IF NOT EXISTS research_capsules_owner_project_idx
                  ON research_capsules(owner_id, project_id, created_at DESC, capsule_id);
                CREATE INDEX IF NOT EXISTS research_capsules_owner_result_idx
                  ON research_capsules(owner_id, result_id, created_at DESC, capsule_id);
                """
            )

    @staticmethod
    def _stored(owner: str, row: sqlite3.Row) -> StoredResearchCapsule:
        capsule = _capsule_from_json(str(row["manifest_json"]))
        if capsule.capsule_id != str(row["capsule_id"]) or capsule.project_id != str(row["project_id"]):
            raise RuntimeError("research capsule row identity mismatch")
        if capsule.fingerprint != _sha(str(row["fingerprint"]), "fingerprint"):
            raise RuntimeError("research capsule row fingerprint mismatch")
        return StoredResearchCapsule(
            owner_id=owner,
            project_id=str(row["project_id"]),
            session_id=str(row["session_id"]),
            result_id=_sha(str(row["result_id"]), "result_id"),
            capsule=capsule,
            supersedes_capsule_id=str(row["supersedes_capsule_id"]),
        )

    def put(
        self,
        owner_id: str,
        *,
        project_id: str,
        session_id: str,
        result_id: str,
        capsule: ResearchCapsule,
        supersedes_capsule_id: str = "",
    ) -> StoredResearchCapsule:
        owner = normalize_owner_id(owner_id)
        project = _text(project_id, "project_id", 256)
        session = _text(session_id, "session_id", 256)
        result = _sha(result_id, "result_id")
        supersedes = _text(supersedes_capsule_id, "supersedes_capsule_id", 256, allow_empty=True)
        if not isinstance(capsule, ResearchCapsule):
            raise TypeError("capsule must be ResearchCapsule")
        if capsule.project_id != project or capsule.run_id != result:
            raise ValueError("capsule project/run identities do not match the durable binding")
        manifest_json = _canonical(_capsule_payload(capsule))
        fingerprint = capsule.fingerprint

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if supersedes:
                    predecessor = connection.execute(
                        "SELECT project_id,result_id FROM research_capsules WHERE owner_id=? AND capsule_id=?",
                        (owner, supersedes),
                    ).fetchone()
                    if predecessor is None:
                        raise KeyError(supersedes)
                    if str(predecessor["project_id"]) != project:
                        raise ValueError("a capsule may only supersede a capsule in the same project")
                existing_fingerprint = connection.execute(
                    "SELECT * FROM research_capsules WHERE owner_id=? AND fingerprint=?",
                    (owner, fingerprint),
                ).fetchone()
                if existing_fingerprint is not None:
                    connection.commit()
                    return self._stored(owner, existing_fingerprint)
                existing_id = connection.execute(
                    "SELECT fingerprint FROM research_capsules WHERE owner_id=? AND capsule_id=?",
                    (owner, capsule.capsule_id),
                ).fetchone()
                if existing_id is not None:
                    raise RuntimeError("research capsule ID collision")
                connection.execute(
                    """INSERT INTO research_capsules
                       (owner_id,capsule_id,project_id,session_id,result_id,fingerprint,
                        manifest_json,supersedes_capsule_id,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        owner,
                        capsule.capsule_id,
                        project,
                        session,
                        result,
                        fingerprint,
                        manifest_json,
                        supersedes,
                        capsule.created_at,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return StoredResearchCapsule(owner, project, session, result, capsule, supersedes)

    def get(self, owner_id: str, capsule_id: str) -> StoredResearchCapsule:
        owner = normalize_owner_id(owner_id)
        identifier = _text(capsule_id, "capsule_id", 256)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_capsules WHERE owner_id=? AND capsule_id=?",
                (owner, identifier),
            ).fetchone()
        if row is None:
            raise KeyError(identifier)
        return self._stored(owner, row)

    def list(
        self,
        owner_id: str,
        *,
        project_id: str | None = None,
        result_id: str | None = None,
        limit: int = 100,
    ) -> tuple[StoredResearchCapsule, ...]:
        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")
        clauses = ["owner_id=?"]
        params: list[Any] = [owner]
        if project_id is not None:
            clauses.append("project_id=?")
            params.append(_text(project_id, "project_id", 256))
        if result_id is not None:
            clauses.append("result_id=?")
            params.append(_sha(result_id, "result_id"))
        params.append(limit)
        query = (
            "SELECT * FROM research_capsules WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at DESC,capsule_id LIMIT ?"
        )
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._stored(owner, row) for row in rows)


__all__ = ["ResearchCapsuleStore", "StoredResearchCapsule"]
