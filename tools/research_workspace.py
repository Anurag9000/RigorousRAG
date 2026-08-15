"""Persistent research-project/session workspace contracts.

The workspace stores project/corpus/session identities, query/result fingerprints and
user-authored notes without copying raw private evidence into generic chat history. A
storage backend can persist these immutable/append-only records for browser/API clients.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence

from tools.security import normalize_owner_id


def _text(value: Any, label: str, maximum: int = 5000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.replace("\x00", " ").strip()
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class CorpusBinding:
    corpus_id: str
    generation_sha256: str
    retrieval_profile_sha256: str
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "corpus_id", _text(self.corpus_id, "corpus_id", 256))
        object.__setattr__(self, "generation_sha256", _sha(self.generation_sha256, "generation_sha256"))
        object.__setattr__(self, "retrieval_profile_sha256", _sha(self.retrieval_profile_sha256, "retrieval_profile_sha256"))
        object.__setattr__(self, "label", _text(self.label, "label", 1000, allow_empty=True))


@dataclass(frozen=True)
class ResearchProject:
    owner_id: str
    project_id: str
    title: str
    research_question: str
    corpora: tuple[CorpusBinding, ...]
    tags: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)
    archived: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "project_id", _text(self.project_id, "project_id", 256))
        object.__setattr__(self, "title", _text(self.title, "title", 1000))
        object.__setattr__(self, "research_question", _text(self.research_question, "research_question", 20_000))
        if len(self.corpora) > 1000 or any(not isinstance(item, CorpusBinding) for item in self.corpora):
            raise ValueError("corpora are invalid")
        if len({item.corpus_id for item in self.corpora}) != len(self.corpora):
            raise ValueError("duplicate corpus bindings")
        if len(self.tags) > 100:
            raise ValueError("tags exceed the item limit")
        object.__setattr__(self, "tags", tuple(dict.fromkeys(_text(item, "tag", 100) for item in self.tags)))
        if not isinstance(self.archived, bool):
            raise ValueError("archived must be boolean")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class ResearchTurn:
    turn_id: str
    query_sha256: str
    strategy: str
    result_sha256: str
    citation_ids: tuple[str, ...]
    plan_sha256: str = ""
    policy_sha256: str = ""
    notes: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "turn_id", _text(self.turn_id, "turn_id", 256))
        object.__setattr__(self, "query_sha256", _sha(self.query_sha256, "query_sha256"))
        object.__setattr__(self, "strategy", _text(self.strategy, "strategy", 64).lower())
        object.__setattr__(self, "result_sha256", _sha(self.result_sha256, "result_sha256"))
        if len(self.citation_ids) > 1000:
            raise ValueError("citation_ids exceed the item limit")
        object.__setattr__(self, "citation_ids", tuple(dict.fromkeys(_text(item, "citation_id", 256) for item in self.citation_ids)))
        object.__setattr__(self, "plan_sha256", _sha(self.plan_sha256, "plan_sha256", allow_empty=True))
        object.__setattr__(self, "policy_sha256", _sha(self.policy_sha256, "policy_sha256", allow_empty=True))
        object.__setattr__(self, "notes", _text(self.notes, "notes", 20_000, allow_empty=True))


@dataclass(frozen=True)
class ResearchSession:
    owner_id: str
    project_id: str
    session_id: str
    turns: tuple[ResearchTurn, ...] = ()
    created_at: float = field(default_factory=time.time)
    closed_at: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "project_id", _text(self.project_id, "project_id", 256))
        object.__setattr__(self, "session_id", _text(self.session_id, "session_id", 256))
        if len(self.turns) > 100_000 or any(not isinstance(item, ResearchTurn) for item in self.turns):
            raise ValueError("turns are invalid")
        if len({item.turn_id for item in self.turns}) != len(self.turns):
            raise ValueError("duplicate turn IDs")
        if self.closed_at is not None and self.closed_at < self.created_at:
            raise ValueError("closed_at precedes created_at")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


class ResearchWorkspaceStore(Protocol):
    def create_project(self, project: ResearchProject) -> None: ...
    def get_project(self, owner_id: str, project_id: str) -> ResearchProject: ...
    def put_session(self, session: ResearchSession, *, expected_fingerprint: str | None = None) -> None: ...
    def get_session(self, owner_id: str, session_id: str) -> ResearchSession: ...


class InMemoryResearchWorkspaceStore:
    def __init__(self) -> None:
        self._projects: dict[tuple[str, str], ResearchProject] = {}
        self._sessions: dict[tuple[str, str], ResearchSession] = {}

    def create_project(self, project: ResearchProject) -> None:
        key = (project.owner_id, project.project_id)
        existing = self._projects.get(key)
        if existing is not None and existing != project:
            raise ValueError("project already exists with different content")
        self._projects[key] = project

    def get_project(self, owner_id: str, project_id: str) -> ResearchProject:
        return self._projects[(normalize_owner_id(owner_id), _text(project_id, "project_id", 256))]

    def put_session(self, session: ResearchSession, *, expected_fingerprint: str | None = None) -> None:
        key = (session.owner_id, session.session_id)
        existing = self._sessions.get(key)
        if expected_fingerprint is not None:
            if existing is None or existing.fingerprint != expected_fingerprint:
                raise RuntimeError("research session optimistic concurrency check failed")
        self._sessions[key] = session

    def get_session(self, owner_id: str, session_id: str) -> ResearchSession:
        return self._sessions[(normalize_owner_id(owner_id), _text(session_id, "session_id", 256))]


def append_turn(session: ResearchSession, turn: ResearchTurn) -> ResearchSession:
    if session.closed_at is not None:
        raise ValueError("cannot append to a closed research session")
    if turn.turn_id in {item.turn_id for item in session.turns}:
        raise ValueError("turn_id already exists")
    return replace(session, turns=(*session.turns, turn))


__all__ = [
    "CorpusBinding", "InMemoryResearchWorkspaceStore", "ResearchProject", "ResearchSession",
    "ResearchTurn", "ResearchWorkspaceStore", "append_turn",
]
