"""Lazy Sequence facade over one or more authoritative dynamic-RAG JSONL shards.

The canonical dynamic v2 builder intentionally consumes an indexable ``Sequence`` so episode
spooling can revisit exact records deterministically.  Python tuples/lists would make that API
corpus-sized in memory.  This facade preserves the Sequence contract while delegating each item
read to ``ManifestBoundAuthoritativeJsonlDataset`` and retaining only shard metadata plus a
bounded prefix-length index.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from training.advanced_rag_authoritative_data import (
    LegalDynamicRagEpisodeStep,
    ManifestBoundAuthoritativeJsonlDataset,
)

_MAX_SHARDS = 100_000


@dataclass(frozen=True)
class DynamicJsonlShard:
    path: str
    content_sha256: str
    dataset_manifest_sha256: str
    split_name: str
    expected_record_count: int | None = None

    def open(self) -> ManifestBoundAuthoritativeJsonlDataset:
        return ManifestBoundAuthoritativeJsonlDataset(
            self.path,
            expected_sha256=self.content_sha256,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            split_name=self.split_name,
            record_kind="dynamic_rag_episode",
            expected_record_count=self.expected_record_count,
        )


class IndexedDynamicStepSequence(Sequence[LegalDynamicRagEpisodeStep]):
    """Deterministic lazy concatenation of validated dynamic JSONL shards."""

    def __init__(self, shards: Sequence[DynamicJsonlShard]) -> None:
        selected = tuple(shards)
        if not selected or len(selected) > _MAX_SHARDS:
            raise ValueError(f"dynamic source requires 1..{_MAX_SHARDS} shards")
        if any(not isinstance(item, DynamicJsonlShard) for item in selected):
            raise ValueError("dynamic source shards must be DynamicJsonlShard values")
        datasets = tuple(item.open() for item in selected)
        if any(len(dataset) <= 0 for dataset in datasets):
            raise ValueError("dynamic source shards may not be empty")
        prefix = []
        running = 0
        for dataset in datasets:
            running += len(dataset)
            prefix.append(running)
        self._datasets = datasets
        self._prefix = tuple(prefix)
        self._length = running

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int | slice) -> LegalDynamicRagEpisodeStep | list[LegalDynamicRagEpisodeStep]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._length)
            return [self[position] for position in range(start, stop, step)]
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dynamic source index must be an integer or slice")
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError("dynamic source index out of range")
        shard_index = bisect_right(self._prefix, index)
        previous = 0 if shard_index == 0 else self._prefix[shard_index - 1]
        value = self._datasets[shard_index][index - previous]
        if not isinstance(value, LegalDynamicRagEpisodeStep):
            raise ValueError("authoritative dynamic shard returned a non-legal step")
        return value


__all__ = ["DynamicJsonlShard", "IndexedDynamicStepSequence"]
