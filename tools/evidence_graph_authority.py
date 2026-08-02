"""Fail-closed authority checks for serving derived evidence graphs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from tools.security import normalize_owner_id


class EvidenceGraphAuthorityError(RuntimeError):
    """Raised when a derived current graph is stale or identity-mismatched."""


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or any(
        ord(character) < 32 or ord(character) == 127 for character in cleaned
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _digest(value: Any, label: str) -> str:
    cleaned = _identifier(value, label, 64).lower()
    if len(cleaned) != 64 or any(character not in "0123456789abcdef" for character in cleaned):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return cleaned


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class EvidenceGraphAuthorityView:
    batch: Any
    authoritative_sequence: int | None
    authoritative_state: str | None
    authoritative_current: bool
    authority_digest: str


def _deleted_tombstone(batch: Any) -> bool:
    nodes = tuple(getattr(batch, "nodes", ()))
    edges = tuple(getattr(batch, "edges", ()))
    if len(nodes) != 1 or edges:
        return False
    node = nodes[0]
    metadata = getattr(node, "metadata", {})
    return bool(
        getattr(node, "node_type", None) == "document"
        and isinstance(metadata, dict)
        and metadata.get("derived_tombstone") is True
        and metadata.get("authoritative_state") == "deleted"
    )


def assess_graph_authority(
    batch: Any,
    generation: Any | None,
) -> EvidenceGraphAuthorityView:
    """Classify one graph against the authoritative current generation."""

    graph_digest = _digest(getattr(batch, "graph_digest", None), "graph_digest")
    current = bool(
        generation is not None
        and getattr(batch, "owner_id", None) == getattr(generation, "owner_id", None)
        and getattr(batch, "doc_id", None) == getattr(generation, "doc_id", None)
        and getattr(batch, "generation", None) == getattr(generation, "sequence", None)
        and getattr(batch, "content_sha256", None)
        == getattr(generation, "content_sha256", None)
        and getattr(batch, "profile_fingerprint", None)
        == getattr(generation, "profile_fingerprint", None)
    )
    state = None if generation is None else getattr(generation, "state", None)
    if current and state == "deleted" and not _deleted_tombstone(batch):
        current = False
    sequence = None if generation is None else getattr(generation, "sequence", None)
    return EvidenceGraphAuthorityView(
        batch=batch,
        authoritative_sequence=(
            sequence if isinstance(sequence, int) and not isinstance(sequence, bool) else None
        ),
        authoritative_state=state if isinstance(state, str) else None,
        authoritative_current=current,
        authority_digest=_sha256(
            {
                "scope": "rigorousrag-evidence-graph-authority-v1",
                "graph_digest": graph_digest,
                "graph_generation": getattr(batch, "generation", None),
                "authoritative_sequence": sequence,
                "authoritative_state": state,
                "authoritative_content_sha256": (
                    None
                    if generation is None
                    else getattr(generation, "content_sha256", None)
                ),
                "authoritative_profile_fingerprint": (
                    None
                    if generation is None
                    else getattr(generation, "profile_fingerprint", None)
                ),
                "authoritative_current": current,
            }
        ),
    )


def resolve_evidence_graph(
    *,
    owner_id: str,
    doc_id: str,
    graphs: Any,
    generations: Any,
    generation: int | None = None,
) -> EvidenceGraphAuthorityView:
    """Resolve current graphs fail-closed; explicit historical graphs remain inspectable."""

    owner = normalize_owner_id(owner_id)
    document = _identifier(doc_id, "doc_id")
    authoritative = generations.current(owner_id=owner, doc_id=document)
    if generation is None:
        if authoritative is None:
            raise KeyError((owner, document))
        batch = graphs.current(owner_id=owner, doc_id=document)
        if batch is None:
            raise KeyError((owner, document))
        view = assess_graph_authority(batch, authoritative)
        if not view.authoritative_current:
            raise EvidenceGraphAuthorityError(
                "derived evidence graph does not match the authoritative current generation."
            )
        return view
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise ValueError("generation must be a positive integer.")
    batch = graphs.get(owner_id=owner, doc_id=document, generation=generation)
    return assess_graph_authority(batch, authoritative)


__all__ = [
    "EvidenceGraphAuthorityError",
    "EvidenceGraphAuthorityView",
    "assess_graph_authority",
    "resolve_evidence_graph",
]
