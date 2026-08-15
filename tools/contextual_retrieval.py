"""Hierarchical/contextual retrieval primitives with strict provenance preservation.

Chunks remain authoritative evidence units. Contextualization adds bounded parent and
neighbor context for scoring only; it does not change citation identity or pretend that
derived summaries are source text.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

_MAX_CHUNKS = 500_000
_MAX_TEXT = 100_000


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _score(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must lie in [0,1]")
    return result


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class HierarchicalChunk:
    chunk_id: str
    doc_id: str
    text: str
    source_sha256: str
    page_number: int | None = None
    section_path: tuple[str, ...] = ()
    parent_chunk_id: str = ""
    previous_chunk_id: str = ""
    next_chunk_id: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _text(self.chunk_id, "chunk_id", 500))
        object.__setattr__(self, "doc_id", _text(self.doc_id, "doc_id", 200))
        object.__setattr__(self, "text", _text(self.text, "chunk text", _MAX_TEXT))
        digest = _text(self.source_sha256, "source_sha256", 64).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("source_sha256 is invalid")
        object.__setattr__(self, "source_sha256", digest)
        if self.page_number is not None and (isinstance(self.page_number, bool) or not isinstance(self.page_number, int) or not 1 <= self.page_number <= 1_000_000):
            raise ValueError("page_number is invalid")
        if len(self.section_path) > 32:
            raise ValueError("section_path exceeds the limit")
        object.__setattr__(self, "section_path", tuple(_text(item, "section heading", 1000) for item in self.section_path))
        for name in ("parent_chunk_id", "previous_chunk_id", "next_chunk_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name, 500, allow_empty=True))
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 64:
            raise ValueError("metadata must be a bounded mapping")
        object.__setattr__(self, "metadata", {_text(str(k), "metadata key", 100): _text(str(v), "metadata value", 1000) for k, v in self.metadata.items()})

    @property
    def evidence_fingerprint(self) -> str:
        return hashlib.sha256(_canonical({"chunk_id": self.chunk_id, "doc_id": self.doc_id, "source_sha256": self.source_sha256, "page_number": self.page_number, "text_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest()})).hexdigest()


@dataclass(frozen=True)
class ContextWindow:
    target_chunk_id: str
    scoring_text: str
    evidence_fingerprint: str
    context_chunk_ids: tuple[str, ...]
    derived_context_sha256: str


def build_context_window(
    target: HierarchicalChunk,
    chunks: Mapping[str, HierarchicalChunk],
    *,
    include_parent: bool = True,
    neighbor_radius: int = 1,
    max_context_chars: int = 12_000,
) -> ContextWindow:
    if target.chunk_id not in chunks or chunks[target.chunk_id] != target:
        raise ValueError("target must exist in chunks mapping")
    if not 0 <= neighbor_radius <= 8 or not 100 <= max_context_chars <= 100_000:
        raise ValueError("context limits are invalid")
    context: list[HierarchicalChunk] = []
    if include_parent and target.parent_chunk_id and target.parent_chunk_id in chunks:
        parent = chunks[target.parent_chunk_id]
        if parent.doc_id == target.doc_id and parent.source_sha256 == target.source_sha256:
            context.append(parent)
    current_id = target.previous_chunk_id
    previous: list[HierarchicalChunk] = []
    for _ in range(neighbor_radius):
        if not current_id or current_id not in chunks:
            break
        item = chunks[current_id]
        if item.doc_id != target.doc_id or item.source_sha256 != target.source_sha256:
            break
        previous.append(item)
        current_id = item.previous_chunk_id
    context.extend(reversed(previous))
    context.append(target)
    current_id = target.next_chunk_id
    for _ in range(neighbor_radius):
        if not current_id or current_id not in chunks:
            break
        item = chunks[current_id]
        if item.doc_id != target.doc_id or item.source_sha256 != target.source_sha256:
            break
        context.append(item)
        current_id = item.next_chunk_id
    pieces: list[str] = []
    used: list[str] = []
    if target.section_path:
        pieces.append(" > ".join(target.section_path))
    for item in context:
        candidate = item.text
        prospective = "\n\n".join((*pieces, candidate))
        if len(prospective) > max_context_chars:
            if item.chunk_id == target.chunk_id:
                remaining = max(1, max_context_chars - len("\n\n".join(pieces)) - 2)
                pieces.append(candidate[:remaining])
                used.append(item.chunk_id)
            continue
        pieces.append(candidate)
        used.append(item.chunk_id)
    scoring_text = "\n\n".join(pieces)
    return ContextWindow(target.chunk_id, scoring_text, target.evidence_fingerprint, tuple(used), hashlib.sha256(scoring_text.encode("utf-8")).hexdigest())


@dataclass(frozen=True)
class HierarchicalScore:
    chunk_id: str
    chunk_score: float
    parent_score: float = 0.0
    section_score: float = 0.0
    neighbor_score: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _text(self.chunk_id, "chunk_id", 500))
        for name in ("chunk_score", "parent_score", "section_score", "neighbor_score"):
            object.__setattr__(self, name, _score(getattr(self, name), name))

    def combined(self, *, chunk_weight: float = 0.65, parent_weight: float = 0.15, section_weight: float = 0.1, neighbor_weight: float = 0.1) -> float:
        weights = (chunk_weight, parent_weight, section_weight, neighbor_weight)
        if any(not math.isfinite(float(value)) or value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("hierarchical weights are invalid")
        total = sum(weights)
        return (
            self.chunk_score * chunk_weight
            + self.parent_score * parent_weight
            + self.section_score * section_weight
            + self.neighbor_score * neighbor_weight
        ) / total


def select_hierarchical(
    scores: Sequence[HierarchicalScore],
    chunks: Mapping[str, HierarchicalChunk],
    *,
    top_k: int = 20,
    max_per_document: int = 4,
) -> tuple[HierarchicalChunk, ...]:
    if len(scores) > _MAX_CHUNKS or not 1 <= top_k <= 1000 or not 1 <= max_per_document <= 100:
        raise ValueError("hierarchical selection limits are invalid")
    ranked = sorted(scores, key=lambda item: (-item.combined(), item.chunk_id))
    counts: dict[str, int] = {}
    selected: list[HierarchicalChunk] = []
    for score in ranked:
        chunk = chunks.get(score.chunk_id)
        if chunk is None:
            continue
        if counts.get(chunk.doc_id, 0) >= max_per_document:
            continue
        counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        selected.append(chunk)
        if len(selected) >= top_k:
            break
    return tuple(selected)


__all__ = ["ContextWindow", "HierarchicalChunk", "HierarchicalScore", "build_context_window", "select_hierarchical"]
