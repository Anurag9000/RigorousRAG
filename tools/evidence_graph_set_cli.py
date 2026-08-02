"""Read-only operator CLI for persisted cross-document graph sets."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.evidence_graph_runtime import get_evidence_graph_store
from tools.evidence_graph_set_runtime import get_evidence_graph_set_store
from tools.evidence_graph_set_store import (
    EvidenceGraphSetAuthorityError,
    assess_graph_set_authority,
)
from tools.evidence_graph_sets import (
    cross_document_neighbors,
    find_cross_document_paths,
)
from tools.security import normalize_owner_id
from tools.sparse_runtime import get_generation_store


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _summary(value: Any, report: Any) -> dict[str, Any]:
    return {
        "owner_id": value.owner_id,
        "graph_set_key": value.graph_set_key,
        "graph_set_id": value.graph_set_id,
        "graph_set_digest": value.graph_set_digest,
        "member_count": len(value.members),
        "edge_count": len(value.edges),
        "members": [
            {
                "doc_id": member.doc_id,
                "generation": member.generation,
                "content_sha256": member.content_sha256,
                "profile_fingerprint": member.profile_fingerprint,
                "graph_digest": member.graph_digest,
                "authority_digest": member.authority_digest,
            }
            for member in value.members
        ],
        "authoritative_current": report.authoritative_current,
        "stale_member_doc_ids": list(report.stale_member_doc_ids),
        "missing_member_doc_ids": list(report.missing_member_doc_ids),
        "authority_digest": report.authority_digest,
        "created_at": value.created_at,
        "mutation_performed": False,
        "semantic_inference_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_set_cli",
        description=(
            "Inspect explicit cross-document graph sets. Current logical-key reads "
            "fail closed when any member generation is stale."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    status.add_argument("--owner-id", required=True)
    status.add_argument("--graph-set-key", required=True)
    status.add_argument("--graph-set-id")
    history = commands.add_parser("history")
    history.add_argument("--owner-id", required=True)
    history.add_argument("--graph-set-key", required=True)
    history.add_argument("--limit", type=int, default=100)
    neighbors = commands.add_parser("neighbors")
    neighbors.add_argument("--owner-id", required=True)
    neighbors.add_argument("--graph-set-key", required=True)
    neighbors.add_argument("--doc-id", required=True)
    neighbors.add_argument("--node-id", required=True)
    neighbors.add_argument("--edge-type", action="append")
    paths = commands.add_parser("paths")
    paths.add_argument("--owner-id", required=True)
    paths.add_argument("--graph-set-key", required=True)
    paths.add_argument("--source-doc-id", required=True)
    paths.add_argument("--source-node-id", required=True)
    paths.add_argument("--target-doc-id", required=True)
    paths.add_argument("--target-node-id", required=True)
    paths.add_argument("--edge-type", action="append")
    paths.add_argument("--max-depth", type=int, default=6)
    paths.add_argument("--max-paths", type=int, default=20)
    return parser


def _current(owner_id: str, key: str) -> tuple[Any, Any]:
    return get_evidence_graph_set_store().resolve_current(
        owner_id=owner_id,
        graph_set_key=key,
        generations=get_generation_store(),
        graphs=get_evidence_graph_store(),
    )


def _status(args: argparse.Namespace) -> int:
    owner = normalize_owner_id(args.owner_id)
    store = get_evidence_graph_set_store()
    if args.graph_set_id:
        value = store.get(owner_id=owner, graph_set_id=args.graph_set_id)
        report = assess_graph_set_authority(
            value,
            generations=get_generation_store(),
            graphs=get_evidence_graph_store(),
        )
    else:
        value, report = _current(owner, args.graph_set_key)
    _print(_summary(value, report))
    return 0


def _history(args: argparse.Namespace) -> int:
    owner = normalize_owner_id(args.owner_id)
    store = get_evidence_graph_set_store()
    generations = get_generation_store()
    graphs = get_evidence_graph_store()
    values = store.history(
        owner_id=owner,
        graph_set_key=args.graph_set_key,
        limit=args.limit,
    )
    _print(
        {
            "owner_id": owner,
            "graph_set_key": args.graph_set_key,
            "count": len(values),
            "versions": [
                _summary(
                    value,
                    assess_graph_set_authority(
                        value, generations=generations, graphs=graphs
                    ),
                )
                for value in values
            ],
            "mutation_performed": False,
        }
    )
    return 0


def _neighbors(args: argparse.Namespace) -> int:
    value, report = _current(args.owner_id, args.graph_set_key)
    values = cross_document_neighbors(
        value,
        doc_id=args.doc_id,
        node_id=args.node_id,
        edge_types=args.edge_type,
    )
    _print(
        {
            **_summary(value, report),
            "result_count": len(values),
            "results": [
                {
                    "edge_id": edge.edge_id,
                    "edge_type": edge.edge_type,
                    "relation_key": edge.relation_key,
                    "weight": edge.weight,
                    "target": {
                        "doc_id": target.doc_id,
                        "generation": target.generation,
                        "node_id": target.node_id,
                        "node_type": target.node_type,
                        "label": target.label,
                        "page_number": target.page_number,
                        "section": target.section,
                        "provenance_digest": target.provenance_digest,
                    },
                }
                for edge, target in values
            ],
        }
    )
    return 0


def _paths(args: argparse.Namespace) -> int:
    value, report = _current(args.owner_id, args.graph_set_key)
    values = find_cross_document_paths(
        value,
        source_doc_id=args.source_doc_id,
        source_node_id=args.source_node_id,
        target_doc_id=args.target_doc_id,
        target_node_id=args.target_node_id,
        edge_types=args.edge_type,
        max_depth=args.max_depth,
        max_paths=args.max_paths,
    )
    _print(
        {
            **_summary(value, report),
            "path_count": len(values),
            "paths": [
                {
                    "path_digest": path.path_digest,
                    "nodes": [
                        {
                            "doc_id": node.doc_id,
                            "generation": node.generation,
                            "node_id": node.node_id,
                            "node_type": node.node_type,
                            "label": node.label,
                            "page_number": node.page_number,
                            "section": node.section,
                            "provenance_digest": node.provenance_digest,
                        }
                        for node in path.nodes
                    ],
                    "edges": [
                        {
                            "edge_id": edge.edge_id,
                            "edge_type": edge.edge_type,
                            "relation_key": edge.relation_key,
                            "weight": edge.weight,
                            "provenance_digest": edge.provenance_digest,
                        }
                        for edge in path.edges
                    ],
                }
                for path in values
            ],
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "status":
            return _status(args)
        if args.command == "history":
            return _history(args)
        if args.command == "neighbors":
            return _neighbors(args)
        if args.command == "paths":
            return _paths(args)
        raise ValueError("unsupported graph set command.")
    except EvidenceGraphSetAuthorityError:
        _print({"error": "stale_graph_set"}, stream=sys.stderr)
        return 1
    except KeyError:
        _print({"error": "not_found"}, stream=sys.stderr)
        return 1
    except (OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
