"""Server-owned hydrology status/inspection projections for browser clients.

The browser receives typed summaries instead of interpreting stored engineering payloads.
Staleness is joined against immutable artifact fingerprints and a truncated stale-ledger
scan is reported as unknown rather than incorrectly calling unseen artifacts current.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Mapping, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query

from tools.hydrology_report import report_payload
from tools.hydrology_store import HydrologyArtifactEnvelope, HydrologyArtifactStore, decode_artifact
from tools.research_access import ResearchAccessResolver

_SHA_PATTERN = r"^[0-9a-fA-F]{64}$"
_KIND_TO_STALE = {
    "topology": "hydrology_topology",
    "engineering_package": "hydrology_package",
    "retrieval_plan": "hydrology_plan",
    "evidence_projection": "hydrology_projection",
    "evidence_report": "hydrology_report",
}
_MAX_STALE_SCAN = 10_000


class StaleStore(Protocol):
    def list_stale(self, owner_id: str, *, kind: str | None = None, limit: int = 1000): ...


def _owner(principal: Any) -> str:
    value = getattr(principal, "owner_id", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=401, detail="Authenticated owner identity is required.")
    return value


def _access(resolver: ResearchAccessResolver, actor: str, project_id: str):
    try:
        return resolver.project(actor, project_id, permission="hydrology.read")
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="Research project not found.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Research access control is unavailable.") from exc


def _artifact(
    store: HydrologyArtifactStore,
    owner_id: str,
    project_id: str,
    kind: str,
    logical_id: str,
    fingerprint: str | None = None,
) -> HydrologyArtifactEnvelope:
    try:
        return store.get(owner_id, project_id, kind, logical_id, fingerprint=fingerprint)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Hydrology artifact not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Hydrology persistence is unavailable.") from exc


def _stale_index(store: StaleStore | None, owner_id: str) -> tuple[dict[tuple[str, str], list[Mapping[str, Any]]], bool]:
    if store is None:
        return {}, True
    try:
        rows = tuple(store.list_stale(owner_id, limit=_MAX_STALE_SCAN))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Hydrology stale-state ledger is unavailable.") from exc
    index: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        artifact = getattr(row, "artifact", None)
        kind = getattr(artifact, "kind", None)
        resource_id = getattr(artifact, "resource_id", None)
        if not isinstance(kind, str) or not isinstance(resource_id, str):
            continue
        index.setdefault((kind, resource_id), []).append(
            {
                "event_sha256": str(getattr(row, "triggering_event_sha256", "")),
                "reason": str(getattr(row, "reason", "")),
                "stale_at": float(getattr(row, "stale_at", 0.0)),
                "replacement_id": str(getattr(row, "replacement_id", "")),
            }
        )
    return index, len(rows) < _MAX_STALE_SCAN


def _stale_state(
    index: Mapping[tuple[str, str], list[Mapping[str, Any]]],
    complete: bool,
    kind: str,
    fingerprint: str,
) -> Mapping[str, Any]:
    stale_kind = _KIND_TO_STALE[kind]
    reasons = list(index.get((stale_kind, fingerprint), ()))
    return {
        "stale": bool(reasons) if complete or reasons else None,
        "stale_state_complete": complete,
        "stale_reasons": reasons,
    }


def _preview(values: Any, maximum: int = 100) -> tuple[list[Any], int]:
    rows = list(values)
    return rows[:maximum], len(rows)


def _summary(kind: str, typed: Any) -> Mapping[str, Any]:
    if kind == "topology":
        return {"node_count": len(typed.nodes), "reach_count": len(typed.reaches)}
    if kind == "engineering_package":
        diagnostics, count = _preview(typed.diagnostics)
        return {
            "package_id": typed.package_id,
            "model_type": typed.model_type,
            "record_count": len(typed.records),
            "object_count": len(typed.objects),
            "complete": typed.complete,
            "diagnostic_count": count,
            "diagnostics_preview": diagnostics,
            "source_fingerprint": typed.source_fingerprint,
            "topology_fingerprint": typed.topology_fingerprint,
            "scenario_fingerprint": typed.scenario_fingerprint,
        }
    if kind == "retrieval_plan":
        unresolved, count = _preview(typed.unresolved)
        return {
            "scope": typed.spec.scope,
            "anchor_node_id": typed.spec.anchor_node_id,
            "target_node_id": typed.spec.target_node_id,
            "variable": typed.spec.variable,
            "scenario_ids": list(typed.spec.scenario_ids),
            "modalities": list(typed.spec.modalities),
            "record_count": len(typed.record_ids),
            "node_count": len(typed.node_ids),
            "reach_count": len(typed.reach_ids),
            "time_window_count": len(typed.time_windows),
            "executable": typed.executable,
            "unresolved_count": count,
            "unresolved_preview": unresolved,
            "package_fingerprint": typed.package_fingerprint,
            "topology_fingerprint": typed.topology_fingerprint,
            "index_fingerprint": typed.index_fingerprint,
        }
    if kind == "evidence_projection":
        diagnostics = tuple(dict.fromkeys((*typed.package_diagnostics, *typed.plan_unresolved)))
        preview, count = _preview(diagnostics)
        return {
            "row_count": len(typed.rows),
            "complete": typed.complete,
            "diagnostic_count": count,
            "diagnostics_preview": preview,
            "package_fingerprint": typed.package_fingerprint,
            "topology_fingerprint": typed.topology_fingerprint,
            "plan_fingerprint": typed.plan_fingerprint,
            "index_fingerprint": typed.index_fingerprint,
        }
    if kind == "evidence_report":
        payload = report_payload(typed)
        preview, count = _preview(typed.diagnostics)
        return {
            "complete": typed.complete,
            "summary": payload["summary"],
            "diagnostic_count": count,
            "diagnostics_preview": preview,
            "projection_fingerprint": typed.projection_fingerprint,
            "package_fingerprint": typed.package_fingerprint,
            "topology_fingerprint": typed.topology_fingerprint,
            "plan_fingerprint": typed.plan_fingerprint,
            "index_fingerprint": typed.index_fingerprint,
        }
    raise ValueError("unsupported hydrology artifact kind")


def _detail(kind: str, typed: Any, limit: int) -> Mapping[str, Any]:
    base = dict(_summary(kind, typed))
    if kind == "topology":
        nodes = [
            {
                "node_id": item.node_id,
                "kind": item.kind,
                "source_id": item.source_id,
                "location": None if item.location is None else {
                    "x": item.location.x,
                    "y": item.location.y,
                    "crs": f"{item.location.crs.authority}:{item.location.crs.code}",
                    "axis_order": item.location.crs.axis_order,
                },
            }
            for item in (typed.nodes[key] for key in sorted(typed.nodes))
        ]
        reaches = [
            {
                "reach_id": item.reach_id,
                "upstream_node_id": item.upstream_node_id,
                "downstream_node_id": item.downstream_node_id,
                "length_m": item.length_m,
                "source_id": item.source_id,
            }
            for item in (typed.reaches[key] for key in sorted(typed.reaches))
        ]
        base.update({"nodes": nodes[:limit], "reaches": reaches[:limit], "nodes_truncated": len(nodes) > limit, "reaches_truncated": len(reaches) > limit})
    elif kind == "engineering_package":
        objects = [asdict(item) for item in typed.objects[:limit]]
        records = [
            {
                "record_id": item.record_id,
                "source_id": item.source_id,
                "variable": item.variable,
                "modality": item.modality,
                "content_sha256": item.content_sha256,
                "scenario_id": str(item.metadata.get("scenario_id", "")),
                "topology_kind": str(item.metadata.get("topology_kind", "")),
                "topology_id": str(item.metadata.get("topology_id", "")),
                "has_spatial_scope": item.spatial is not None,
                "has_temporal_scope": item.temporal is not None,
            }
            for item in typed.records[:limit]
        ]
        base.update({"objects": objects, "records": records, "objects_truncated": len(typed.objects) > limit, "records_truncated": len(typed.records) > limit})
    elif kind == "retrieval_plan":
        base.update({
            "selected_records": [asdict(item) for item in typed.selected_records[:limit]],
            "time_windows": [asdict(item) for item in typed.time_windows[:limit]],
            "selected_records_truncated": len(typed.selected_records) > limit,
            "time_windows_truncated": len(typed.time_windows) > limit,
        })
    elif kind == "evidence_projection":
        from tools.hydrology_projection import evidence_row_payload
        base.update({"rows": [evidence_row_payload(item) for item in typed.rows[:limit]], "rows_truncated": len(typed.rows) > limit})
    elif kind == "evidence_report":
        payload = report_payload(typed)
        base.update({"rows": list(payload["rows"][:limit]), "rows_truncated": len(typed.rows) > limit, "title": typed.title, "research_question": typed.research_question})
    return base


def build_hydrology_status_router(
    *,
    principal_dependency: Callable[..., Any],
    store: HydrologyArtifactStore,
    access_resolver: ResearchAccessResolver,
    invalidation_store: StaleStore | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/research", tags=["research-hydrology-status"])

    @router.get("/projects/{project_id}/hydrology/status")
    async def project_status(
        project_id: str,
        include_history: bool = Query(default=False),
        limit: int = Query(default=500, ge=1, le=5000),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        access = _access(access_resolver, _owner(principal), project_id)
        try:
            summaries = store.list(access.storage_owner_id, access.project.project_id, include_history=include_history, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Hydrology persistence is unavailable.") from exc
        stale_index, stale_complete = _stale_index(invalidation_store, access.storage_owner_id)
        artifacts: list[Mapping[str, Any]] = []
        for item in summaries:
            try:
                envelope = store.get(
                    access.storage_owner_id,
                    access.project.project_id,
                    item.kind,
                    item.logical_id,
                    fingerprint=item.fingerprint,
                )
                typed = decode_artifact(item.kind, envelope.payload)
                typed_summary = _summary(item.kind, typed)
            except Exception as exc:
                typed_summary = {"integrity_error": str(exc)}
            artifacts.append({
                "kind": item.kind,
                "logical_id": item.logical_id,
                "fingerprint": item.fingerprint,
                "version": item.version,
                "created_at": item.created_at,
                "is_current": item.is_current,
                **_stale_state(stale_index, stale_complete, item.kind, item.fingerprint),
                "summary": typed_summary,
            })
        return {
            "project": {
                "project_id": access.project.project_id,
                "title": access.project.title,
                "research_question": access.project.research_question,
                "role": access.role,
                "permissions": sorted(access.permissions),
            },
            "stale_state_complete": stale_complete,
            "artifacts": artifacts,
        }

    @router.get("/projects/{project_id}/hydrology/status/{kind}/{logical_id}")
    async def artifact_status(
        project_id: str,
        kind: str,
        logical_id: str,
        fingerprint: str | None = Query(default=None, pattern=_SHA_PATTERN),
        detail_limit: int = Query(default=500, ge=1, le=5000),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        if kind not in _KIND_TO_STALE:
            raise HTTPException(status_code=400, detail="Unsupported hydrology artifact kind.")
        access = _access(access_resolver, _owner(principal), project_id)
        envelope = _artifact(store, access.storage_owner_id, access.project.project_id, kind, logical_id, fingerprint)
        try:
            typed = decode_artifact(kind, envelope.payload)
            details = _detail(kind, typed, detail_limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        stale_index, stale_complete = _stale_index(invalidation_store, access.storage_owner_id)
        return {
            "project_id": access.project.project_id,
            "kind": kind,
            "logical_id": logical_id,
            "fingerprint": envelope.fingerprint,
            "created_at": envelope.created_at,
            **_stale_state(stale_index, stale_complete, kind, envelope.fingerprint),
            "details": details,
        }

    return router


__all__ = ["StaleStore", "build_hydrology_status_router"]
