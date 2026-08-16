"""ACL-aware replay recipe status and privacy controls.

This router deliberately does not expose replay plaintext and deliberately does not
execute model/provider work. Encrypted recipe decryption remains an operator-side
recomputation concern.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from tools.replay_recipe_store import EncryptedReplayRecipeStore, ReplayRecipeMetadata
from tools.research_access import ResearchAccessResolver


def _owner(principal: Any) -> str:
    value = getattr(principal, "owner_id", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=401, detail="Authenticated owner identity is required.")
    return value


def _payload(item: ReplayRecipeMetadata) -> Mapping[str, Any]:
    return {
        "result_id": item.result_id,
        "query_sha256": item.query_sha256,
        "model": item.model,
        "strategy": item.strategy,
        "created_at": item.created_at,
    }


def _session_scope(
    actor: str,
    session_id: str | None,
    *,
    permission: str,
    access_resolver: ResearchAccessResolver | None,
) -> tuple[str, set[str] | None]:
    if session_id is None:
        return actor, None
    if access_resolver is None:
        raise HTTPException(status_code=409, detail="Session-scoped replay access is not configured.")
    try:
        access = access_resolver.session(actor, session_id, permission=permission)
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="Research session not found.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    result_ids = {turn.result_sha256 for turn in access.session.turns if turn.result_sha256}
    return access.storage_owner_id, result_ids


def build_replay_router(
    *,
    principal_dependency: Callable[..., Any],
    replay_recipe_store: EncryptedReplayRecipeStore | None,
    access_resolver: ResearchAccessResolver | None = None,
) -> APIRouter:
    """Build non-executing replay status/privacy routes."""

    if replay_recipe_store is not None and not isinstance(replay_recipe_store, EncryptedReplayRecipeStore):
        raise TypeError("replay_recipe_store must be EncryptedReplayRecipeStore or null")
    if access_resolver is not None and not isinstance(access_resolver, ResearchAccessResolver):
        raise TypeError("access_resolver must be ResearchAccessResolver or null")

    router = APIRouter(prefix="/research/replay", tags=["research-replay"])

    @router.get("")
    async def list_replay_recipes(
        limit: int = Query(default=100, ge=1, le=1000),
        session_id: str | None = Query(default=None, min_length=1, max_length=256),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        storage_owner, allowed_ids = _session_scope(
            actor,
            session_id,
            permission="result.read",
            access_resolver=access_resolver,
        )
        if replay_recipe_store is None:
            return {"configured": False, "recipes": []}
        try:
            values = replay_recipe_store.list_metadata(storage_owner, limit=1000 if allowed_ids is not None else limit)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Replay recipe persistence is unavailable.") from exc
        if allowed_ids is not None:
            values = tuple(item for item in values if item.result_id in allowed_ids)[:limit]
        return {"configured": True, "recipes": [_payload(item) for item in values]}

    @router.get("/{result_id}")
    async def get_replay_recipe_status(
        result_id: str,
        session_id: str | None = Query(default=None, min_length=1, max_length=256),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        storage_owner, allowed_ids = _session_scope(
            actor,
            session_id,
            permission="result.read",
            access_resolver=access_resolver,
        )
        if allowed_ids is not None and result_id.lower() not in allowed_ids:
            raise HTTPException(status_code=404, detail="Replay recipe not found.")
        if replay_recipe_store is None:
            raise HTTPException(status_code=409, detail="Encrypted replay is not configured for this deployment.")
        try:
            item = replay_recipe_store.metadata(storage_owner, result_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Replay recipe not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Research result ID is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Replay recipe persistence is unavailable.") from exc
        return {"configured": True, "recipe": _payload(item)}

    @router.delete("/{result_id}", status_code=204)
    async def delete_replay_recipe(
        result_id: str,
        session_id: str | None = Query(default=None, min_length=1, max_length=256),
        principal: Any = Depends(principal_dependency),
    ) -> Response:
        actor = _owner(principal)
        storage_owner, allowed_ids = _session_scope(
            actor,
            session_id,
            permission="replay.manage",
            access_resolver=access_resolver,
        )
        if allowed_ids is not None and result_id.lower() not in allowed_ids:
            raise HTTPException(status_code=404, detail="Replay recipe not found.")
        if replay_recipe_store is None:
            raise HTTPException(status_code=409, detail="Encrypted replay is not configured for this deployment.")
        try:
            deleted = replay_recipe_store.delete(storage_owner, result_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Research result ID is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Replay recipe persistence is unavailable.") from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Replay recipe not found.")
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    return router


__all__ = ["build_replay_router"]
