"""Read-only operator CLI for persisted evidence graphs."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.evidence_graph_analysis import analyze_evidence_graph
from tools.evidence_graph_authority import (
    EvidenceGraphAuthorityError,
    assess_graph_authority,
    resolve_evidence_graph,
)
from tools.evidence_graph_retrieval import find_paths, search_nodes
from tools.evidence_graph_runtime import get_evidence_graph_store
from tools.security import normalize_owner_id
from tools.sparse_runtime import get_generation_store


def _print(payload: Any, *, stream: Any = None) -> None:
    destination = stream if stream is not None else sys.stdout
    destination.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or any(
        ord(character) < 32 or ord(character) == 127 for character in cleaned
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _view(owner_id: str, doc_id: str, generation: int | None) -> Any:
    return resolve_evidence_graph(
        owner_id=owner_id,
        doc_id=doc_id,
        graphs=get_evidence_graph_store(),
        generations=get_generation_store(),
        generation=generation,
    )


def _summary(view: Any) -> dict[str, Any]:
    batch = view.batch
    return {
        "owner_id": batch.owner_id,
        "doc_id": batch.doc_id,
        "generation": batch.generation,
        "content_sha256": batch.content_sha256,
        "profile_fingerprint": batch.profile_fingerprint,
        "graph_digest": batch.graph_digest,
        "node_count": len(batch.nodes),
        "edge_count": len(batch.edges),
        "created_at": batch.created_at,
        "authoritative_sequence": view.authoritative_sequence,
        "authoritative_state": view.authoritative_state,
        "authoritative_current": view.authoritative_current,
        "authority_digest": view.authority_digest,
        "mutation_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_cli",
        description=(
            "Inspect persisted explicit evidence graphs. Current graph reads fail "
            "closed unless they match the authoritative generation."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("--owner-id", required=True)
        command.add_argument("--doc-id", required=True)
        command.add_argument("--generation", type=int)
    history = commands.add_parser("history")
    history.add_argument("--owner-id", required=True)
    history.add_argument("--doc-id", required=True)
    history.add_argument("--limit", type=int, default=100)
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--owner-id", required=True)
    search.add_argument("--doc-id", required=True)
    search.add_argument("--generation", type=int)
    search.add_argument("--node-type", action="append")
    search.add_argument("--limit", type=int, default=20)
    paths = commands.add_parser("paths")
    paths.add_argument("source_node_id")
    paths.add_argument("target_node_id")
    paths.add_argument("--owner-id", required=True)
    paths.add_argument("--doc-id", required=True)
    paths.add_argument("--generation", type=int)
    paths.add_argument("--edge-type", action="append")
    paths.add_argument("--max-depth", type=int, default=6)
    paths.add_argument("--max-paths", type=int, default=20)
    return parser


def _status(args: argparse.Namespace) -> int:
    _print(_summary(_view(args.owner_id, args.doc_id, args.generation)))
    return 0


def _history(args: argparse.Namespace) -> int:
    owner = normalize_owner_id(args.owner_id)
    doc_id = _identifier(args.doc_id, "doc_id", 200)
    graphs = get_evidence_graph_store()
    authoritative = get_generation_store().current(owner_id=owner, doc_id=doc_id)
    values = graphs.history(
        owner_id=owner,
        doc_id=doc_id,
        limit=_integer(args.limit, "limit", 1, 10_000),
    )
    _print(
        {
            "owner_id": owner,
            "doc_id": doc_id,
            "count": len(values),
            "generations": [
                _summary(assess_graph_authority(value, authoritative))
                for value in values
            ],
            "mutation_performed": False,
        }
    )
    return 0


def _search(args: argparse.Namespace) -> int:
    view = _view(args.owner_id, args.doc_id, args.generation)
    batch = view.batch
    values = search_nodes(
        batch,
        args.query,
        node_types=args.node_type,
        limit=_integer(args.limit, "limit", 1, 1_000),
    )
    _print(
        {
            **_summary(view),
            "query_term_matches_only": True,
            "result_count": len(values),
            "results": [
                {
                    "node_id": value.node.node_id,
                    "node_type": value.node.node_type,
                    "label": value.node.label,
                    "page_number": value.node.page_number,
                    "section": value.node.section,
                    "score": value.score,
                    "matched_terms": list(value.matched_terms),
                    "provenance_digest": value.node.provenance_digest,
                }
                for value in values
            ],
        }
    )
    return 0


def _paths(args: argparse.Namespace) -> int:
    view = _view(args.owner_id, args.doc_id, args.generation)
    batch = view.batch
    values = find_paths(
        batch,
        source_node_id=_identifier(args.source_node_id, "source_node_id", 64),
        target_node_id=_identifier(args.target_node_id, "target_node_id", 64),
        edge_types=args.edge_type,
        max_depth=_integer(args.max_depth, "max_depth", 1, 20),
        max_paths=_integer(args.max_paths, "max_paths", 1, 1_000),
    )
    _print(
        {
            **_summary(view),
            "path_count": len(values),
            "paths": [
                {
                    "path_digest": value.path_digest,
                    "nodes": [
                        {
                            "node_id": node.node_id,
                            "node_type": node.node_type,
                            "label": node.label,
                            "page_number": node.page_number,
                            "section": node.section,
                        }
                        for node in value.nodes
                    ],
                    "edges": [
                        {
                            "edge_id": edge.edge_id,
                            "edge_type": edge.edge_type,
                            "weight": edge.weight,
                            "provenance_digest": edge.provenance_digest,
                        }
                        for edge in value.edges
                    ],
                }
                for value in values
            ],
        }
    )
    return 0


def _analyze(args: argparse.Namespace) -> int:
    view = _view(args.owner_id, args.doc_id, args.generation)
    batch = view.batch
    report = analyze_evidence_graph(batch)
    _print(
        {
            **_summary(view),
            "analysis_digest": report.analysis_digest,
            "node_counts": report.node_counts,
            "edge_counts": report.edge_counts,
            "semantic_inference_performed": False,
            "claim_clusters": [
                {
                    "claim_node_id": cluster.claim.node_id,
                    "claim_label": cluster.claim.label,
                    "supporting_node_ids": [
                        node.node_id for node in cluster.supporting_nodes
                    ],
                    "contradicting_node_ids": [
                        node.node_id for node in cluster.contradicting_nodes
                    ],
                    "support_edge_ids": [
                        edge.edge_id for edge in cluster.support_edges
                    ],
                    "contradiction_edge_ids": [
                        edge.edge_id for edge in cluster.contradiction_edges
                    ],
                    "has_conflict": cluster.has_conflict,
                    "cluster_digest": cluster.cluster_digest,
                }
                for cluster in report.claim_clusters
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
        if args.command == "search":
            return _search(args)
        if args.command == "paths":
            return _paths(args)
        if args.command == "analyze":
            return _analyze(args)
        raise ValueError("unsupported evidence graph command.")
    except EvidenceGraphAuthorityError:
        _print({"error": "stale_graph"}, stream=sys.stderr)
        return 1
    except KeyError:
        _print({"error": "not_found"}, stream=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
