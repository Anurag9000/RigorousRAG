"""Owner-scoped replay recipe status and privacy controls.

This router deliberately does not expose replay plaintext and deliberately does not
execute model/provider work. Encrypted recipe decryption remains an operator-side
recomputation concern. Product callers may only inspect non-secret availability metadata
or delete their own stored replay recipe.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from tools.replay_recipe_store import EncryptedReplayRecipeStore, ReplayRecipeMetadata


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


def build_replay_router(
    *,
    principal_dependency: Callable[..., Any],
    replay_recipe_store: EncryptedReplayRecipeStore | None,
) -> APIRouter:
    """Build non-executing replay status/privacy routes.

    A null store is a supported deployment state: the status endpoint reports replay as
    unconfigured while result generation remains hash-only. Per-result reads/deletes fail
    closed rather than pretending an exact replay recipe exists.
    """

    if replay_recipe_store is not None and not isinstance(replay_recipe_store, EncryptedReplayRecipeStore):
        raise TypeError("replay_recipe_store must be EncryptedReplayRecipeStore or null")

    router = APIRouter(prefix="/research/replay", tags=["research-replay"])

    @router.get("")
    async def list_replay_recipes(
        limit: int = Query(default=100, ge=1, le=1000),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        if replay_recipe_store is None:
            return {"configured": False, "recipes": []}
        try:
            values = replay_recipe_store.list_metadata(owner, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Replay recipe persistence is unavailable.") from exc
        return {"configured": True, "recipes": [_payload(item) for item in values]}

    @router.get("/{result_id}")
    async def get_replay_recipe_status(
        result_id: str,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        if replay_recipe_store is None:
            raise HTTPException(status_code=409, detail="Encrypted replay is not configured for this deployment.")
        try:
            item = replay_recipe_store.metadata(owner, result_id)
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
        principal: Any = Depends(principal_dependency),
    ) -> Response:
        owner = _owner(principal)
        if replay_recipe_store is None:
            raise HTTPException(status_code=409, detail="Encrypted replay is not configured for this deployment.")
        try:
            deleted = replay_recipe_store.delete(owner, result_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Research result ID is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Replay recipe persistence is unavailable.") from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Replay recipe not found.")
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    return router


__all__ = ["build_replay_router"]
