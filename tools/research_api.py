"""Owner-scoped research workspace API router.

The router exposes persistent project/session/turn state and capability/domain metadata.
It stores only query/result fingerprints and citation IDs for turns, not raw retrieved
private evidence.  Citation authority remains with the research agent/result pipeline.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from tools.capability_registry import CapabilityRegistry
from tools.domain_adapter import DomainAdapterRegistry
from tools.research_workspace import CorpusBinding, ResearchProject, ResearchSession, ResearchTurn, append_turn


class WorkspaceStore(Protocol):
    def create_project(self, project: ResearchProject) -> None: ...
    def get_project(self, owner_id: str, project_id: str) -> ResearchProject: ...
    def list_projects(self, owner_id: str, *, limit: int = 200) -> Sequence[ResearchProject]: ...
    def put_session(self, session: ResearchSession, *, expected_fingerprint: str | None = None) -> None: ...
    def get_session(self, owner_id: str, session_id: str) -> ResearchSession: ...
    def list_sessions(self, owner_id: str, project_id: str, *, limit: int = 200) -> Sequence[ResearchSession]: ...


class CorpusBindingRequest(BaseModel):
    corpus_id: str = Field(..., min_length=1, max_length=256)
    generation_sha256: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")
    retrieval_profile_sha256: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")
    label: str = Field(default="", max_length=1000)


class CreateProjectRequest(BaseModel):
    project_id: Optional[str] = Field(default=None, min_length=1, max_length=256)
    title: str = Field(..., min_length=1, max_length=1000)
    research_question: str = Field(..., min_length=1, max_length=20_000)
    corpora: list[CorpusBindingRequest] = Field(default_factory=list, max_length=1000)
    tags: list[str] = Field(default_factory=list, max_length=100)


class CreateSessionRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, min_length=1, max_length=256)


class AppendTurnRequest(BaseModel):
    query_sha256: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")
    strategy: str = Field(..., min_length=1, max_length=64)
    result_sha256: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")
    citation_ids: list[str] = Field(default_factory=list, max_length=1000)
    plan_sha256: str = Field(default="", max_length=64)
    policy_sha256: str = Field(default="", max_length=64)
    notes: str = Field(default="", max_length=20_000)
    expected_session_fingerprint: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")


class CloseSessionRequest(BaseModel):
    expected_session_fingerprint: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")


def _principal_owner(principal: Any) -> str:
    owner = getattr(principal, "owner_id", None)
    if not isinstance(owner, str) or not owner:
        raise HTTPException(status_code=401, detail="Authenticated owner identity is required.")
    return owner


def _project_payload(project: ResearchProject) -> Mapping[str, Any]:
    return {
        "project_id": project.project_id,
        "title": project.title,
        "research_question": project.research_question,
        "corpora": [asdict(item) for item in project.corpora],
        "tags": list(project.tags),
        "created_at": project.created_at,
        "archived": project.archived,
        "fingerprint": project.fingerprint,
    }


def _turn_payload(turn: ResearchTurn) -> Mapping[str, Any]:
    return {
        "turn_id": turn.turn_id,
        "query_sha256": turn.query_sha256,
        "strategy": turn.strategy,
        "result_sha256": turn.result_sha256,
        "citation_ids": list(turn.citation_ids),
        "plan_sha256": turn.plan_sha256,
        "policy_sha256": turn.policy_sha256,
        "notes": turn.notes,
        "created_at": turn.created_at,
    }


def _session_payload(session: ResearchSession) -> Mapping[str, Any]:
    return {
        "project_id": session.project_id,
        "session_id": session.session_id,
        "turns": [_turn_payload(item) for item in session.turns],
        "created_at": session.created_at,
        "closed_at": session.closed_at,
        "fingerprint": session.fingerprint,
    }


def build_research_router(
    *,
    principal_dependency: Callable[..., Any],
    workspace_store: WorkspaceStore,
    capability_registry: CapabilityRegistry | None = None,
    domain_registry: DomainAdapterRegistry | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/research", tags=["research"])

    @router.get("/projects")
    async def list_projects(
        limit: int = Query(default=100, ge=1, le=1000),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _principal_owner(principal)
        try:
            projects = workspace_store.list_projects(owner, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research workspace is unavailable.") from exc
        return {"projects": [_project_payload(item) for item in projects]}

    @router.post("/projects", status_code=201)
    async def create_project(
        request: CreateProjectRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _principal_owner(principal)
        project_id = request.project_id or f"project_{uuid.uuid4().hex}"
        try:
            project = ResearchProject(
                owner_id=owner,
                project_id=project_id,
                title=request.title,
                research_question=request.research_question,
                corpora=tuple(
                    CorpusBinding(
                        corpus_id=item.corpus_id,
                        generation_sha256=item.generation_sha256.lower(),
                        retrieval_profile_sha256=item.retrieval_profile_sha256.lower(),
                        label=item.label,
                    )
                    for item in request.corpora
                ),
                tags=tuple(request.tags),
            )
            workspace_store.create_project(project)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Research project is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research workspace is unavailable.") from exc
        return _project_payload(project)

    @router.get("/projects/{project_id}")
    async def get_project(
        project_id: str,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _principal_owner(principal)
        try:
            project = workspace_store.get_project(owner, project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Research project not found.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research workspace is unavailable.") from exc
        return _project_payload(project)

    @router.get("/projects/{project_id}/sessions")
    async def list_sessions(
        project_id: str,
        limit: int = Query(default=100, ge=1, le=1000),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _principal_owner(principal)
        try:
            sessions = workspace_store.list_sessions(owner, project_id, limit=limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Research project not found.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research workspace is unavailable.") from exc
        return {"sessions": [_session_payload(item) for item in sessions]}

    @router.post("/projects/{project_id}/sessions", status_code=201)
    async def create_session(
        project_id: str,
        request: CreateSessionRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _principal_owner(principal)
        try:
            workspace_store.get_project(owner, project_id)
            session = ResearchSession(
                owner_id=owner,
                project_id=project_id,
                session_id=request.session_id or f"session_{uuid.uuid4().hex}",
            )
            workspace_store.put_session(session)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Research project not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Research session is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research workspace is unavailable.") from exc
        return _session_payload(session)

    @router.get("/sessions/{session_id}")
    async def get_session(
        session_id: str,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _principal_owner(principal)
        try:
            session = workspace_store.get_session(owner, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Research session not found.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research workspace is unavailable.") from exc
        return _session_payload(session)

    @router.post("/sessions/{session_id}/turns")
    async def append_session_turn(
        session_id: str,
        request: AppendTurnRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _principal_owner(principal)
        try:
            current = workspace_store.get_session(owner, session_id)
            turn = ResearchTurn(
                turn_id=f"turn_{uuid.uuid4().hex}",
                query_sha256=request.query_sha256.lower(),
                strategy=request.strategy,
                result_sha256=request.result_sha256.lower(),
                citation_ids=tuple(request.citation_ids),
                plan_sha256=request.plan_sha256.lower(),
                policy_sha256=request.policy_sha256.lower(),
                notes=request.notes,
            )
            updated = append_turn(current, turn)
            workspace_store.put_session(
                updated,
                expected_fingerprint=request.expected_session_fingerprint.lower(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Research session not found.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail="Research session changed; reload before appending.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Research turn is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research workspace is unavailable.") from exc
        return _session_payload(updated)

    @router.post("/sessions/{session_id}/close")
    async def close_session(
        session_id: str,
        request: CloseSessionRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _principal_owner(principal)
        try:
            current = workspace_store.get_session(owner, session_id)
            if current.closed_at is not None:
                return _session_payload(current)
            updated = ResearchSession(
                owner_id=current.owner_id,
                project_id=current.project_id,
                session_id=current.session_id,
                turns=current.turns,
                created_at=current.created_at,
                closed_at=time.time(),
            )
            workspace_store.put_session(
                updated,
                expected_fingerprint=request.expected_session_fingerprint.lower(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Research session not found.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail="Research session changed; reload before closing.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research workspace is unavailable.") from exc
        return _session_payload(updated)

    @router.get("/capabilities")
    async def capabilities(
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        _principal_owner(principal)
        capabilities_payload: list[Mapping[str, Any]] = []
        if capability_registry is not None:
            for descriptor in capability_registry.snapshot():
                capabilities_payload.append(
                    {
                        "capability_id": descriptor.capability_id,
                        "version": descriptor.version,
                        "kind": descriptor.kind,
                        "provider": descriptor.provider,
                        "modalities": list(descriptor.modalities),
                        "trust_level": descriptor.trust_level,
                        "enabled": descriptor.enabled,
                        "fingerprint": descriptor.fingerprint,
                    }
                )
        domains_payload: list[Mapping[str, Any]] = []
        if domain_registry is not None:
            for descriptor in domain_registry.descriptors():
                domains_payload.append(
                    {
                        "domain_id": descriptor.domain_id,
                        "version": descriptor.version,
                        "label": descriptor.label,
                        "capabilities": list(descriptor.capabilities),
                        "fingerprint": descriptor.fingerprint,
                    }
                )
        fingerprint = hashlib.sha256(
            repr((capabilities_payload, domains_payload)).encode("utf-8")
        ).hexdigest()
        return {
            "capabilities": capabilities_payload,
            "domains": domains_payload,
            "fingerprint": fingerprint,
        }

    return router


__all__ = ["WorkspaceStore", "build_research_router"]
