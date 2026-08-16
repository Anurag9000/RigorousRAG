"""Read-only governed hydrology tools for the research agent.

The tools deliberately expose no mutation surface. Project authorization is resolved at
call time through the same ResearchAccessResolver used by HTTP routes. Evidence citations
point at original source identities and label their snippets as derived hydrology metadata;
no tool invents verbatim source text.
"""
from __future__ import annotations

from dataclasses import asdict
from types import ModuleType
from typing import Any, Mapping
from urllib.parse import quote

from tools.agent_tool_registry import AgentToolSpec, ToolResult
from tools.agent_tool_registry_integration import register_governed_agent_tool
from tools.hydrology_dependency_policy import require_fresh, require_sources_active, stale_reasons
from tools.hydrology_store import HydrologyArtifactStore, decode_artifact
from tools.models import Citation
from tools.research_access import ResearchAccessResolver

_KINDS = ("topology", "engineering_package", "retrieval_plan", "evidence_projection", "evidence_report")
_EVIDENCE_KINDS = ("evidence_projection", "evidence_report")
_MAX_STATUS = 100
_MAX_EVIDENCE = 50
_MAX_TOPOLOGY = 200


def _project(resolver: ResearchAccessResolver, actor: str, project_id: str):
    return resolver.project(actor, project_id, permission="hydrology.read")


def _artifact(
    store: HydrologyArtifactStore,
    owner_id: str,
    project_id: str,
    kind: str,
    logical_id: str,
    fingerprint: str = "",
):
    return store.get(
        owner_id,
        project_id,
        kind,
        logical_id,
        fingerprint=fingerprint or None,
    )


def _stale_payload(ledger: Any, owner_id: str, kind: str, fingerprint: str) -> list[Mapping[str, Any]]:
    if ledger is None:
        return []
    lookup_kind = {
        "topology": "topology",
        "engineering_package": "package",
        "retrieval_plan": "plan",
        "evidence_projection": "projection",
        "evidence_report": "report",
    }[kind]
    return [item.as_dict() for item in stale_reasons(ledger, owner_id, kind=lookup_kind, fingerprint=fingerprint)]


def _derived_citation(
    *,
    label: str,
    project_id: str,
    artifact_kind: str,
    artifact_id: str,
    artifact_fingerprint: str,
    row: Any,
) -> Citation:
    source = row.source_id
    topology = f"{row.topology_kind}:{row.topology_id}"
    parts = [f"Hydrology evidence record {row.record_id}", f"topology={topology}", f"modality={row.modality}"]
    if row.variable:
        parts.append(f"variable={row.variable}")
    if row.scenario_id:
        parts.append(f"scenario={row.scenario_id}")
    if row.content_sha256:
        parts.append(f"content_sha256={row.content_sha256}")
    return Citation(
        label=label,
        title=f"Hydrology evidence — {row.record_id}",
        url=f"local://hydrology/{quote(project_id, safe='')}/{quote(source, safe='')}",
        source_type="tool_output",
        snippet="; ".join(parts),
        source_id=source,
        chunk_id=row.record_id,
        metadata={
            "derived_evidence": True,
            "artifact_kind": artifact_kind,
            "artifact_id": artifact_id,
            "artifact_fingerprint": artifact_fingerprint,
            "content_sha256": row.content_sha256,
            "variable": row.variable,
            "scenario_id": row.scenario_id,
            "topology_kind": row.topology_kind,
            "topology_id": row.topology_id,
            "selection_reasons": list(row.selection_reasons),
        },
    )


def register_hydrology_agent_tools(
    module: ModuleType,
    *,
    store: HydrologyArtifactStore,
    access_resolver: ResearchAccessResolver,
    invalidation_store: Any = None,
) -> None:
    """Register bounded, read-only hydrology tools into the governed agent registry."""

    def status_handler(owner_id: str, arguments: Mapping[str, Any], budget: Any) -> ToolResult:
        access = _project(access_resolver, owner_id, str(arguments["project_id"]))
        selected_kind = str(arguments.get("kind", ""))
        include_history = bool(arguments.get("include_history", False))
        limit = min(int(arguments.get("limit", 25)), _MAX_STATUS)
        rows = store.list(
            access.storage_owner_id,
            access.project.project_id,
            kind=selected_kind or None,
            include_history=include_history,
            limit=limit,
        )
        artifacts = []
        for item in rows:
            artifacts.append(
                {
                    "kind": item.kind,
                    "logical_id": item.logical_id,
                    "fingerprint": item.fingerprint,
                    "version": item.version,
                    "created_at": item.created_at,
                    "is_current": item.is_current,
                    "stale_reasons": _stale_payload(
                        invalidation_store,
                        access.storage_owner_id,
                        item.kind,
                        item.fingerprint,
                    ),
                }
            )
        return ToolResult(
            content={
                "project_id": access.project.project_id,
                "project_title": access.project.title,
                "role": access.role,
                "artifacts": artifacts,
            },
            warnings=("Hydrology status metadata is not itself a source quotation.",),
        )

    def topology_handler(owner_id: str, arguments: Mapping[str, Any], budget: Any) -> ToolResult:
        access = _project(access_resolver, owner_id, str(arguments["project_id"]))
        logical_id = str(arguments["topology_id"])
        envelope = _artifact(
            store,
            access.storage_owner_id,
            access.project.project_id,
            "topology",
            logical_id,
            str(arguments.get("fingerprint", "")),
        )
        require_fresh(
            invalidation_store,
            access.storage_owner_id,
            kind="topology",
            fingerprint=envelope.fingerprint,
        )
        network = decode_artifact("topology", envelope.payload)
        anchor = str(arguments["anchor_node_id"])
        if anchor not in network.nodes:
            raise KeyError("hydrology anchor node was not found")
        direction = str(arguments.get("direction", "local"))
        max_hops = int(arguments.get("max_hops", 20))
        limit = min(int(arguments.get("limit", 100)), _MAX_TOPOLOGY)
        if direction == "upstream":
            node_ids = (anchor, *network.upstream_nodes(anchor, max_hops=max_hops))
        elif direction == "downstream":
            node_ids = (anchor, *network.downstream_nodes(anchor, max_hops=max_hops))
        else:
            node_ids = (anchor,)
        selected_nodes = tuple(dict.fromkeys(node_ids))[:limit]
        selected_set = set(selected_nodes)
        reaches = tuple(
            item
            for item in sorted(network.reaches.values(), key=lambda row: row.reach_id)
            if item.upstream_node_id in selected_set and item.downstream_node_id in selected_set
        )[:limit]
        source_ids = [network.nodes[item].source_id for item in selected_nodes]
        source_ids.extend(item.source_id for item in reaches)
        require_sources_active(invalidation_store, access.storage_owner_id, source_ids)
        return ToolResult(
            content={
                "project_id": access.project.project_id,
                "topology_id": logical_id,
                "topology_fingerprint": envelope.fingerprint,
                "anchor_node_id": anchor,
                "direction": direction,
                "nodes": [
                    {
                        "node_id": network.nodes[node_id].node_id,
                        "kind": network.nodes[node_id].kind,
                        "source_id": network.nodes[node_id].source_id,
                        "location": None
                        if network.nodes[node_id].location is None
                        else {
                            "x": network.nodes[node_id].location.x,
                            "y": network.nodes[node_id].location.y,
                            "crs": f"{network.nodes[node_id].location.crs.authority}:{network.nodes[node_id].location.crs.code}",
                        },
                    }
                    for node_id in selected_nodes
                ],
                "reaches": [
                    {
                        "reach_id": item.reach_id,
                        "upstream_node_id": item.upstream_node_id,
                        "downstream_node_id": item.downstream_node_id,
                        "length_m": item.length_m,
                        "source_id": item.source_id,
                    }
                    for item in reaches
                ],
                "truncated": len(node_ids) > limit,
            },
            warnings=("Topology output is structured derived metadata, not verbatim source evidence.",),
        )

    def evidence_handler(owner_id: str, arguments: Mapping[str, Any], budget: Any) -> ToolResult:
        access = _project(access_resolver, owner_id, str(arguments["project_id"]))
        kind = str(arguments["kind"])
        logical_id = str(arguments["logical_id"])
        envelope = _artifact(
            store,
            access.storage_owner_id,
            access.project.project_id,
            kind,
            logical_id,
            str(arguments.get("fingerprint", "")),
        )
        policy_kind = "projection" if kind == "evidence_projection" else "report"
        require_fresh(
            invalidation_store,
            access.storage_owner_id,
            kind=policy_kind,
            fingerprint=envelope.fingerprint,
        )
        artifact = decode_artifact(kind, envelope.payload)
        require_sources_active(
            invalidation_store,
            access.storage_owner_id,
            [item.source_id for item in artifact.rows],
        )
        limit = min(int(arguments.get("limit", 20)), _MAX_EVIDENCE)
        selected = artifact.rows[:limit]
        citations = tuple(
            _derived_citation(
                label=f"[H{index}]",
                project_id=access.project.project_id,
                artifact_kind=kind,
                artifact_id=logical_id,
                artifact_fingerprint=envelope.fingerprint,
                row=row,
            )
            for index, row in enumerate(selected, start=1)
        )
        content_rows = [
            {
                "record_id": row.record_id,
                "source_id": row.source_id,
                "content_sha256": row.content_sha256,
                "variable": row.variable,
                "modality": row.modality,
                "scenario_id": row.scenario_id,
                "topology_kind": row.topology_kind,
                "topology_id": row.topology_id,
                "selection_reasons": list(row.selection_reasons),
                "start_time": row.start_time.isoformat() if row.start_time is not None else None,
                "end_time": row.end_time.isoformat() if row.end_time is not None else None,
            }
            for row in selected
        ]
        return ToolResult(
            content={
                "project_id": access.project.project_id,
                "kind": kind,
                "logical_id": logical_id,
                "artifact_fingerprint": envelope.fingerprint,
                "rows": content_rows,
                "row_count": len(artifact.rows),
                "truncated": len(artifact.rows) > limit,
            },
            citations=citations,
            warnings=(
                "Citation snippets describe persisted hydrology evidence metadata; they are not verbatim source quotations.",
            ),
        )

    register_governed_agent_tool(
        module,
        AgentToolSpec(
            name="hydrology_status",
            description="Inspect versioned hydrology artifacts and stale-state for an accessible research project. Read only.",
            schema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "minLength": 1, "maxLength": 256},
                    "kind": {"type": "string", "enum": list(_KINDS), "maxLength": 64},
                    "include_history": {"type": "boolean"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_STATUS},
                },
                "required": ["project_id"],
                "additionalProperties": False,
            },
            handler=status_handler,
            citation_policy="none",
        ),
    )
    register_governed_agent_tool(
        module,
        AgentToolSpec(
            name="hydrology_topology_scope",
            description="Read a fresh persisted hydrologic topology around an anchor node, optionally upstream or downstream. Read only.",
            schema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "minLength": 1, "maxLength": 256},
                    "topology_id": {"type": "string", "minLength": 1, "maxLength": 500},
                    "anchor_node_id": {"type": "string", "minLength": 1, "maxLength": 256},
                    "direction": {"type": "string", "enum": ["local", "upstream", "downstream"], "maxLength": 32},
                    "max_hops": {"type": "integer", "minimum": 1, "maximum": 100},
                    "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_TOPOLOGY},
                    "fingerprint": {"type": "string", "maxLength": 64},
                },
                "required": ["project_id", "topology_id", "anchor_node_id"],
                "additionalProperties": False,
            },
            handler=topology_handler,
            citation_policy="none",
        ),
    )
    register_governed_agent_tool(
        module,
        AgentToolSpec(
            name="hydrology_evidence_rows",
            description="Read fresh citation-neutral hydrology evidence rows from a persisted projection/report and publish bounded server citations to their original source identities. Read only.",
            schema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "minLength": 1, "maxLength": 256},
                    "kind": {"type": "string", "enum": list(_EVIDENCE_KINDS), "maxLength": 64},
                    "logical_id": {"type": "string", "minLength": 1, "maxLength": 500},
                    "fingerprint": {"type": "string", "maxLength": 64},
                    "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_EVIDENCE},
                },
                "required": ["project_id", "kind", "logical_id"],
                "additionalProperties": False,
            },
            handler=evidence_handler,
            citation_policy="server_only",
        ),
    )


__all__ = ["register_hydrology_agent_tools"]
