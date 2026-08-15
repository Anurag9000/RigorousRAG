"""Server-owned research-query execution, result persistence and session append API."""

from __future__ import annotations

import hashlib
import inspect
import uuid
from typing import Any, Awaitable, Callable, Mapping, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from tools.dependency_invalidation import DependencyInvalidationStore, DependencyRef
from tools.models import AgentAnswer
from tools.research_dependencies import register_result_dependencies, stale_reasons
from tools.research_result_store import ResearchResultStore, StoredResearchResult
from tools.research_workspace import ResearchSession, ResearchTurn, append_turn
from tools.runtime_composition import RuntimeComposition


class WorkspaceStore(Protocol):
    def get_session(self, owner_id: str, session_id: str) -> ResearchSession: ...
    def put_session(self, session: ResearchSession, *, expected_fingerprint: str | None = None) -> None: ...


class ResearchQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=20_000)
    model: str | None = Field(default=None, max_length=200)
    session_id: str | None = Field(default=None, min_length=1, max_length=256)
    expected_session_fingerprint: str | None = Field(default=None, min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")
    notes: str = Field(default="", max_length=20_000)


def _owner(principal: Any) -> str:
    value = getattr(principal, "owner_id", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=401, detail="Authenticated owner identity is required.")
    return value


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _maybe_sha(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    digest = value.strip().lower()
    return digest if len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest) else ""


def _result_payload(
    result: StoredResearchResult,
    *,
    owner_id: str | None = None,
    invalidation_store: DependencyInvalidationStore | None = None,
) -> Mapping[str, Any]:
    stale = ()
    if owner_id is not None and invalidation_store is not None:
        stale = stale_reasons(
            invalidation_store,
            owner_id,
            DependencyRef("result", result.result_id),
        )
    return {
        "result_id": result.result_id,
        "query_sha256": result.query_sha256,
        "answer": result.answer,
        "citations": [item.model_dump(exclude_none=True) for item in result.citations],
        "warnings": list(result.warnings),
        "metadata": dict(result.metadata),
        "strategy": result.strategy,
        "model": result.model,
        "created_at": result.created_at,
        "citation_ids": list(result.citation_ids),
        "stale": bool(stale),
        "stale_reasons": list(stale),
    }


async def _await_maybe(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def build_research_query_router(
    *,
    principal_dependency: Callable[..., Any],
    agent_factory: Callable[[str, str | None], Any],
    run_research_task: Callable[..., Awaitable[Any] | Any],
    result_store: ResearchResultStore,
    workspace_store: WorkspaceStore,
    composition: RuntimeComposition,
    invalidation_store: DependencyInvalidationStore | None = None,
) -> APIRouter:
    if not isinstance(result_store, ResearchResultStore):
        raise TypeError("result_store must be ResearchResultStore")
    if not isinstance(composition, RuntimeComposition):
        raise TypeError("composition must be RuntimeComposition")
    if invalidation_store is not None and not isinstance(invalidation_store, DependencyInvalidationStore):
        raise TypeError("invalidation_store must be DependencyInvalidationStore or null")
    router = APIRouter(prefix="/research", tags=["research-query"])

    @router.post("/query")
    async def run_research_query(request: ResearchQueryRequest, principal: Any = Depends(principal_dependency)) -> Mapping[str, Any]:
        owner = _owner(principal)
        if (request.session_id is None) != (request.expected_session_fingerprint is None):
            raise HTTPException(status_code=400, detail="session_id and expected_session_fingerprint must be supplied together.")
        session: ResearchSession | None = None
        if request.session_id is not None:
            try:
                session = workspace_store.get_session(owner, request.session_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Research session not found.") from exc
            except Exception as exc:
                raise HTTPException(status_code=503, detail="Research workspace is unavailable.") from exc
            if session.closed_at is not None:
                raise HTTPException(status_code=409, detail="Research session is closed.")
            if session.fingerprint != request.expected_session_fingerprint.lower():
                raise HTTPException(status_code=409, detail="Research session changed; reload before querying.")

        try:
            agent = agent_factory(owner, request.model)
            answer = await _await_maybe(run_research_task(agent.run, request.query))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research execution is unavailable.") from exc
        if not isinstance(answer, AgentAnswer):
            raise HTTPException(status_code=502, detail="Research agent returned an invalid result.")

        query_sha256 = _sha_text(request.query)
        strategy = composition.config.retrieval.default_strategy
        try:
            stored = result_store.put(
                owner,
                query_sha256=query_sha256,
                answer=answer,
                strategy=strategy,
                model=str(getattr(agent, "model", request.model or "")),
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research result persistence is unavailable.") from exc

        if invalidation_store is not None:
            try:
                register_result_dependencies(
                    invalidation_store,
                    owner,
                    stored,
                    composition=composition,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Research result was persisted, but dependency lineage could not be "
                        f"registered. Result ID: {stored.result_id}"
                    ),
                ) from exc

        updated_session: ResearchSession | None = None
        if session is not None:
            metadata = dict(answer.metadata or {})
            plan_sha = _maybe_sha(metadata.get("plan_fingerprint") or metadata.get("plan_sha256"))
            policy_sha = _maybe_sha(metadata.get("policy_fingerprint") or metadata.get("policy_sha256"))
            if not policy_sha:
                policy_id = composition.selected_capabilities.get("policy", "")
                descriptor = composition.capabilities.active(policy_id) if policy_id else None
                policy_sha = descriptor.fingerprint if descriptor is not None else ""
            turn = ResearchTurn(
                turn_id=f"turn_{uuid.uuid4().hex}",
                query_sha256=query_sha256,
                strategy=strategy,
                result_sha256=stored.result_id,
                citation_ids=stored.citation_ids,
                plan_sha256=plan_sha,
                policy_sha256=policy_sha,
                notes=request.notes,
            )
            try:
                updated_session = append_turn(session, turn)
                workspace_store.put_session(updated_session, expected_fingerprint=request.expected_session_fingerprint.lower())
            except RuntimeError as exc:
                raise HTTPException(
                    status_code=409,
                    detail=("Research result was persisted, but the session changed before the turn could be appended. " f"Result ID: {stored.result_id}"),
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=("Research result was persisted, but the session turn could not be appended. " f"Result ID: {stored.result_id}"),
                ) from exc

        payload = dict(_result_payload(stored, owner_id=owner, invalidation_store=invalidation_store))
        payload["session_id"] = updated_session.session_id if updated_session else None
        payload["session_fingerprint"] = updated_session.fingerprint if updated_session else None
        return payload

    @router.get("/results")
    async def list_research_results(limit: int = Query(default=100, ge=1, le=1000), principal: Any = Depends(principal_dependency)) -> Mapping[str, Any]:
        owner = _owner(principal)
        try:
            values = result_store.list(owner, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research result persistence is unavailable.") from exc
        rows = []
        for item in values:
            stale = stale_reasons(invalidation_store, owner, DependencyRef("result", item.result_id), maximum=20) if invalidation_store is not None else ()
            rows.append(
                {
                    "result_id": item.result_id,
                    "query_sha256": item.query_sha256,
                    "strategy": item.strategy,
                    "model": item.model,
                    "created_at": item.created_at,
                    "citation_count": len(item.citations),
                    "stale": bool(stale),
                    "stale_reasons": list(stale),
                }
            )
        return {"results": rows}

    @router.get("/results/{result_id}")
    async def get_research_result(result_id: str, principal: Any = Depends(principal_dependency)) -> Mapping[str, Any]:
        owner = _owner(principal)
        try:
            result = result_store.get(owner, result_id)
            return _result_payload(result, owner_id=owner, invalidation_store=invalidation_store)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Research result not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Research result ID is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research result persistence is unavailable.") from exc

    return router


__all__ = ["ResearchQueryRequest", "WorkspaceStore", "build_research_query_router"]
