"""ACL-aware API for immutable research answer replacement history."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Mapping

from fastapi import APIRouter, Depends, HTTPException, Query

from tools.artifact_replacements import ArtifactReplacementStore
from tools.research_access import ResearchAccessResolver
from tools.research_answer_history import answer_history
from tools.research_result_store import ResearchResultStore


def _owner(principal: Any) -> str:
    value = getattr(principal, "owner_id", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=401, detail="Authenticated owner identity is required.")
    return value


def build_research_answer_history_router(
    *,
    principal_dependency: Callable[..., Any],
    result_store: ResearchResultStore,
    replacement_store: ArtifactReplacementStore,
    access_resolver: ResearchAccessResolver | None = None,
) -> APIRouter:
    if not isinstance(result_store, ResearchResultStore):
        raise TypeError("result_store must be ResearchResultStore")
    if not isinstance(replacement_store, ArtifactReplacementStore):
        raise TypeError("replacement_store must be ArtifactReplacementStore")
    if access_resolver is not None and not isinstance(access_resolver, ResearchAccessResolver):
        raise TypeError("access_resolver must be ResearchAccessResolver or null")

    router = APIRouter(prefix="/research/results", tags=["research-answer-history"])

    @router.get("/{result_id}/history")
    async def get_answer_history(
        result_id: str,
        max_depth: int = Query(default=64, ge=1, le=256),
        session_id: str | None = Query(default=None, min_length=1, max_length=256),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        storage_owner = actor
        try:
            if session_id is not None:
                if access_resolver is None:
                    raise RuntimeError("session-scoped answer history is not configured")
                access = access_resolver.session(actor, session_id, permission="result.read")
                storage_owner = access.storage_owner_id
                if result_id.lower() not in {turn.result_sha256 for turn in access.session.turns}:
                    raise KeyError(result_id)
            origin, transitions = answer_history(
                storage_owner,
                result_id,
                results=result_store,
                replacements=replacement_store,
                max_depth=max_depth,
            )
        except (KeyError, PermissionError) as exc:
            raise HTTPException(status_code=404, detail="Research result or replacement result not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Research result identity is invalid.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research answer history is unavailable.") from exc

        current = transitions[-1].new if transitions else origin
        return {
            "origin": asdict(origin),
            "current": asdict(current),
            "superseded": current.result_id != origin.result_id,
            "version_count": 1 + len(transitions),
            "transitions": [asdict(item) for item in transitions],
        }

    return router


__all__ = ["build_research_answer_history_router"]
