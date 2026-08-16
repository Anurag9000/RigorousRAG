"""ACL-scoped live API for governed hydrology topology, evidence and retrieval plans.

The API never performs local-file discovery or executes HEC models. It reconstructs and
verifies typed payloads, binds packages to persisted topology generations, creates plans
server-side from authoritative package records, materializes citation-neutral evidence
projections, and registers generation-aware dependency lineage for invalidation.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from tools.dependency_invalidation import DependencyRef
from tools.hydrology_projection import build_hydrology_projection, projection_payload
from tools.hydrology_retrieval import plan_hydrology_retrieval
from tools.hydrology_store import (
    HydrologyArtifactEnvelope,
    HydrologyArtifactStore,
    HydrologyArtifactSummary,
    decode_artifact,
    make_envelope,
    package_from_payload,
    plan_payload,
    query_spec_from_payload,
    topology_from_payload,
    topology_payload,
)
from tools.research_access import ResearchAccessResolver
from tools.spatiotemporal_index import SpatiotemporalIndex

_SHA_PATTERN = r"^[0-9a-fA-F]{64}$"
_KINDS = frozenset({"topology", "engineering_package", "retrieval_plan", "evidence_projection"})
_MAX_SOURCE_DEPENDENCIES = 10_000
_RECOMPUTABLE_DOWNSTREAM = ("result", "report", "capsule")


class DependencyStore(Protocol):
    def register_dependency(
        self,
        owner_id: str,
        *,
        upstream: DependencyRef,
        downstream: DependencyRef,
        relation: str,
    ) -> Any: ...

    def invalidate(
        self,
        owner_id: str,
        *,
        root: DependencyRef,
        reason: str,
        event_type: str,
        replacement_id: str = "",
        recomputable_kinds: tuple[str, ...] = (),
    ) -> Any: ...


class PutTopologyRequest(BaseModel):
    payload: dict[str, Any]
    expected_current_fingerprint: str | None = Field(default=None, pattern=_SHA_PATTERN)


class PutPackageRequest(BaseModel):
    topology_id: str = Field(..., min_length=1, max_length=500)
    payload: dict[str, Any]
    expected_current_fingerprint: str | None = Field(default=None, pattern=_SHA_PATTERN)


class CreateHydrologyPlanRequest(BaseModel):
    plan_id: str = Field(..., min_length=1, max_length=500)
    topology_id: str = Field(..., min_length=1, max_length=500)
    package_id: str = Field(..., min_length=1, max_length=500)
    spec: dict[str, Any]
    reach_travel_seconds: dict[str, float] = Field(default_factory=dict)
    limit: int = Field(default=1000, ge=1, le=10_000)
    expected_current_fingerprint: str | None = Field(default=None, pattern=_SHA_PATTERN)


class CreateHydrologyProjectionRequest(BaseModel):
    projection_id: str = Field(..., min_length=1, max_length=500)
    package_id: str = Field(..., min_length=1, max_length=500)
    plan_id: str = Field(..., min_length=1, max_length=500)
    expected_current_fingerprint: str | None = Field(default=None, pattern=_SHA_PATTERN)


def _owner(principal: Any) -> str:
    value = getattr(principal, "owner_id", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=401, detail="Authenticated owner identity is required.")
    return value


def _summary(value: HydrologyArtifactSummary) -> Mapping[str, Any]:
    return {
        "project_id": value.project_id,
        "kind": value.kind,
        "logical_id": value.logical_id,
        "fingerprint": value.fingerprint,
        "version": value.version,
        "created_at": value.created_at,
        "is_current": value.is_current,
    }


def _detail(envelope: HydrologyArtifactEnvelope) -> Mapping[str, Any]:
    return {
        "project_id": envelope.project_id,
        "kind": envelope.kind,
        "logical_id": envelope.logical_id,
        "fingerprint": envelope.fingerprint,
        "schema_version": envelope.schema_version,
        "created_at": envelope.created_at,
        "payload": dict(envelope.payload),
    }


def _project_access(
    resolver: ResearchAccessResolver,
    actor: str,
    project_id: str,
    permission: str,
):
    try:
        return resolver.project(actor, project_id, permission=permission)
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="Research project not found.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Research access control is unavailable.") from exc


def _get_artifact(
    store: HydrologyArtifactStore,
    owner_id: str,
    project_id: str,
    kind: str,
    logical_id: str,
    *,
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


def _current_optional(
    store: HydrologyArtifactStore,
    owner_id: str,
    project_id: str,
    kind: str,
    logical_id: str,
) -> HydrologyArtifactEnvelope | None:
    try:
        return store.get(owner_id, project_id, kind, logical_id)
    except KeyError:
        return None


def _dep(kind: str, fingerprint: str) -> DependencyRef:
    return DependencyRef(f"hydrology_{kind}", fingerprint)


def _register(
    dependency_store: DependencyStore | None,
    owner_id: str,
    *,
    upstream: DependencyRef,
    downstream: DependencyRef,
    relation: str,
) -> None:
    if dependency_store is None:
        return
    dependency_store.register_dependency(
        owner_id,
        upstream=upstream,
        downstream=downstream,
        relation=relation,
    )


def _invalidate_replaced(
    dependency_store: DependencyStore | None,
    owner_id: str,
    *,
    kind: str,
    previous: HydrologyArtifactEnvelope | None,
    current: HydrologyArtifactSummary,
) -> None:
    if dependency_store is None or previous is None or previous.fingerprint == current.fingerprint:
        return
    dependency_store.invalidate(
        owner_id,
        root=_dep(kind, previous.fingerprint),
        reason=f"hydrology {kind} generation replaced",
        event_type="hydrology_generation_replaced",
        replacement_id=current.fingerprint,
        # Hydrology descendants are marked stale by graph traversal, but no hydrology
        # recompute task is queued until a deterministic hydrology recompute executor exists.
        recomputable_kinds=_RECOMPUTABLE_DOWNSTREAM,
    )


def _source_dependencies(package: Any) -> tuple[str, ...]:
    values = {item.source_id for item in package.records}
    values.update(item.source_id for item in package.objects)
    output = tuple(sorted(item for item in values if item))
    if len(output) > _MAX_SOURCE_DEPENDENCIES:
        raise RuntimeError(
            f"engineering package has {len(output)} distinct source IDs; dependency lineage limit is {_MAX_SOURCE_DEPENDENCIES}"
        )
    return output


def build_hydrology_router(
    *,
    principal_dependency: Callable[..., Any],
    store: HydrologyArtifactStore,
    access_resolver: ResearchAccessResolver,
    invalidation_store: DependencyStore | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/research", tags=["research-hydrology"])

    @router.get("/projects/{project_id}/hydrology/artifacts")
    async def list_hydrology_artifacts(
        project_id: str,
        kind: str | None = Query(default=None),
        include_history: bool = Query(default=False),
        limit: int = Query(default=200, ge=1, le=5000),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        access = _project_access(access_resolver, actor, project_id, "hydrology.read")
        if kind is not None and kind not in _KINDS:
            raise HTTPException(status_code=400, detail="Unsupported hydrology artifact kind.")
        try:
            values = store.list(
                access.storage_owner_id,
                access.project.project_id,
                kind=kind,
                include_history=include_history,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Hydrology persistence is unavailable.") from exc
        return {"project_id": access.project.project_id, "artifacts": [_summary(item) for item in values]}

    @router.get("/projects/{project_id}/hydrology/artifacts/{kind}/{logical_id}")
    async def get_hydrology_artifact(
        project_id: str,
        kind: str,
        logical_id: str,
        fingerprint: str | None = Query(default=None, pattern=_SHA_PATTERN),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        access = _project_access(access_resolver, actor, project_id, "hydrology.read")
        if kind not in _KINDS:
            raise HTTPException(status_code=400, detail="Unsupported hydrology artifact kind.")
        return _detail(
            _get_artifact(
                store,
                access.storage_owner_id,
                access.project.project_id,
                kind,
                logical_id,
                fingerprint=fingerprint,
            )
        )

    @router.put("/projects/{project_id}/hydrology/topologies/{topology_id}")
    async def put_hydrology_topology(
        project_id: str,
        topology_id: str,
        request: PutTopologyRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        access = _project_access(access_resolver, actor, project_id, "hydrology.write")
        previous = _current_optional(store, access.storage_owner_id, access.project.project_id, "topology", topology_id)
        try:
            network = topology_from_payload(request.payload)
            stored = store.put(
                make_envelope(access.storage_owner_id, access.project.project_id, "topology", topology_id, network),
                expected_current_fingerprint=request.expected_current_fingerprint,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Hydrology topology persistence is unavailable.") from exc
        try:
            _invalidate_replaced(invalidation_store, access.storage_owner_id, kind="topology", previous=previous, current=stored)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Hydrology topology was persisted ({stored.fingerprint}), but dependency invalidation failed.") from exc
        return {**_summary(stored), "payload": topology_payload(network)}

    @router.put("/projects/{project_id}/hydrology/packages/{package_id}")
    async def put_hydrology_package(
        project_id: str,
        package_id: str,
        request: PutPackageRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        access = _project_access(access_resolver, actor, project_id, "hydrology.write")
        topology_envelope = _get_artifact(store, access.storage_owner_id, access.project.project_id, "topology", request.topology_id)
        previous = _current_optional(store, access.storage_owner_id, access.project.project_id, "engineering_package", package_id)
        try:
            network = decode_artifact("topology", topology_envelope.payload)
            package = package_from_payload(request.payload)
            if package.package_id != package_id:
                raise ValueError("package payload ID does not match the route package_id")
            if package.topology_fingerprint != network.fingerprint:
                raise RuntimeError("engineering package is not bound to the selected persisted topology")
            stored = store.put(
                make_envelope(access.storage_owner_id, access.project.project_id, "engineering_package", package_id, package),
                expected_current_fingerprint=request.expected_current_fingerprint,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Hydrology package persistence is unavailable.") from exc
        try:
            package_ref = _dep("package", stored.fingerprint)
            _register(invalidation_store, access.storage_owner_id, upstream=_dep("topology", network.fingerprint), downstream=package_ref, relation="hydrology_topology_package")
            if invalidation_store is not None:
                for source_id in _source_dependencies(package):
                    _register(invalidation_store, access.storage_owner_id, upstream=DependencyRef("source", source_id), downstream=package_ref, relation="source_hydrology_package")
            _invalidate_replaced(invalidation_store, access.storage_owner_id, kind="package", previous=previous, current=stored)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Hydrology package was persisted ({stored.fingerprint}), but dependency lineage could not be completed.") from exc
        return {
            **_summary(stored),
            "topology_id": request.topology_id,
            "complete": package.complete,
            "record_count": len(package.records),
            "object_count": len(package.objects),
            "diagnostics": list(package.diagnostics),
        }

    @router.post("/projects/{project_id}/hydrology/plans", status_code=201)
    async def create_hydrology_plan(
        project_id: str,
        request: CreateHydrologyPlanRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        access = _project_access(access_resolver, actor, project_id, "hydrology.write")
        topology_envelope = _get_artifact(store, access.storage_owner_id, access.project.project_id, "topology", request.topology_id)
        package_envelope = _get_artifact(store, access.storage_owner_id, access.project.project_id, "engineering_package", request.package_id)
        previous = _current_optional(store, access.storage_owner_id, access.project.project_id, "retrieval_plan", request.plan_id)
        try:
            if len(request.reach_travel_seconds) > 10_000:
                raise ValueError("reach_travel_seconds exceed the item limit")
            network = decode_artifact("topology", topology_envelope.payload)
            package = decode_artifact("engineering_package", package_envelope.payload)
            if package.topology_fingerprint != network.fingerprint:
                raise RuntimeError("selected package and topology generations are incompatible")
            spec = query_spec_from_payload(request.spec)
            index = SpatiotemporalIndex()
            package.populate_index(index)
            plan = plan_hydrology_retrieval(
                network,
                index,
                spec,
                reach_travel_seconds=request.reach_travel_seconds,
                limit=request.limit,
                package=package,
                expected_index_fingerprint=index.fingerprint,
            )
            stored = store.put(
                make_envelope(access.storage_owner_id, access.project.project_id, "retrieval_plan", request.plan_id, plan),
                expected_current_fingerprint=request.expected_current_fingerprint,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Hydrology plan creation is unavailable.") from exc
        try:
            plan_ref = _dep("plan", stored.fingerprint)
            _register(invalidation_store, access.storage_owner_id, upstream=_dep("topology", network.fingerprint), downstream=plan_ref, relation="hydrology_topology_plan")
            _register(invalidation_store, access.storage_owner_id, upstream=_dep("package", package.fingerprint), downstream=plan_ref, relation="hydrology_package_plan")
            _invalidate_replaced(invalidation_store, access.storage_owner_id, kind="plan", previous=previous, current=stored)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Hydrology plan was persisted ({stored.fingerprint}), but dependency lineage could not be completed.") from exc
        return {
            **_summary(stored),
            "topology_id": request.topology_id,
            "package_id": request.package_id,
            "executable": plan.executable,
            "record_count": len(plan.record_ids),
            "unresolved": list(plan.unresolved),
            "payload": plan_payload(plan),
        }

    @router.post("/projects/{project_id}/hydrology/projections", status_code=201)
    async def create_hydrology_projection(
        project_id: str,
        request: CreateHydrologyProjectionRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        access = _project_access(access_resolver, actor, project_id, "hydrology.write")
        package_envelope = _get_artifact(store, access.storage_owner_id, access.project.project_id, "engineering_package", request.package_id)
        plan_envelope = _get_artifact(store, access.storage_owner_id, access.project.project_id, "retrieval_plan", request.plan_id)
        previous = _current_optional(store, access.storage_owner_id, access.project.project_id, "evidence_projection", request.projection_id)
        try:
            package = decode_artifact("engineering_package", package_envelope.payload)
            plan = decode_artifact("retrieval_plan", plan_envelope.payload)
            projection = build_hydrology_projection(package, plan, projection_id=request.projection_id)
            stored = store.put(
                make_envelope(access.storage_owner_id, access.project.project_id, "evidence_projection", request.projection_id, projection),
                expected_current_fingerprint=request.expected_current_fingerprint,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Hydrology projection creation is unavailable.") from exc
        try:
            projection_ref = _dep("projection", stored.fingerprint)
            _register(invalidation_store, access.storage_owner_id, upstream=_dep("package", package.fingerprint), downstream=projection_ref, relation="hydrology_package_projection")
            _register(invalidation_store, access.storage_owner_id, upstream=_dep("plan", plan.fingerprint), downstream=projection_ref, relation="hydrology_plan_projection")
            _invalidate_replaced(invalidation_store, access.storage_owner_id, kind="projection", previous=previous, current=stored)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Hydrology projection was persisted ({stored.fingerprint}), but dependency lineage could not be completed.") from exc
        return {
            **_summary(stored),
            "package_id": request.package_id,
            "plan_id": request.plan_id,
            "complete": projection.complete,
            "row_count": len(projection.rows),
            "package_diagnostics": list(projection.package_diagnostics),
            "plan_unresolved": list(projection.plan_unresolved),
            "payload": projection_payload(projection),
        }

    return router


__all__ = [
    "CreateHydrologyPlanRequest",
    "CreateHydrologyProjectionRequest",
    "DependencyStore",
    "PutPackageRequest",
    "PutTopologyRequest",
    "build_hydrology_router",
]
