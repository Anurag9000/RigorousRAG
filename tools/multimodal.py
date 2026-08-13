"""Modality-preserving chunks and cross-modal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Mapping, Optional, Sequence, Tuple


class Modality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FIGURE = "figure"
    AUDIO = "audio"
    VIDEO = "video"
    EQUATION = "equation"


@dataclass(frozen=True)
class MultiModalChunk:
    chunk_id: str
    modality: Modality
    content: str
    source_id: str
    page: Optional[int] = None
    region: Optional[Tuple[float, float, float, float]] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def citation_key(self) -> str:
        location = f":p{self.page}" if self.page is not None else ""
        return f"{self.source_id}{location}:{self.chunk_id}"


@dataclass(frozen=True)
class RankedChunk:
    chunk: MultiModalChunk
    score: float
    rank: int


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankedChunk]],
    *,
    k: float = 60.0,
    modality_weights: Optional[Mapping[Modality, float]] = None,
) -> List[RankedChunk]:
    if k <= 0:
        raise ValueError("k must be positive.")
    scores = {}
    chunks = {}
    for ranking in rankings:
        for index, item in enumerate(ranking, start=1):
            weight = 1.0
            if modality_weights is not None:
                weight = float(modality_weights.get(item.chunk.modality, 1.0))
                if weight < 0:
                    raise ValueError("modality weights must be non-negative.")
            chunks[item.chunk.chunk_id] = item.chunk
            scores[item.chunk.chunk_id] = scores.get(item.chunk.chunk_id, 0.0) + weight / (k + index)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return [
        RankedChunk(chunks[key], scores[key], rank)
        for rank, key in enumerate(ordered, start=1)
    ]
