"""Lossless compatibility planner layered above the canonical cohort runtime.

A *dataset group* is the user-facing parent group: every model whose training input
belongs to the same dataset family is counted together, and parent groups are
ordered by descending distinct model-family count. Multi-dataset jobs form the
final overlap parent group.

A dataset group may contain multiple *compatibility subcohorts*. This is required
to preserve the authored experiment exactly. Models may share one physical raw
training batch only when they have the same sample stream (split/fold, shuffle or
sampler law and seed semantics) and the same optimizer-facing physical-batch
contract. Tokenizers, collators, augmentations and tensor layouts may differ: those
are model ``view_key`` values and are cached independently above the one raw batch.

This prevents an unsafe interpretation of "same dataset" from silently changing
CV folds, weighted-vs-balanced sampling, tail-batch behavior, gradient accumulation
or optimizer-step counts merely to obtain more parallelism.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "opf-dataset-cohort-compatibility/v1"
OVERLAP = "__overlap__"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclasses.dataclass(frozen=True)
class Candidate:
    job_id: str
    model_family: str
    datasets: tuple[str, ...]
    sample_stream_key: str
    batch_contract_key: str
    physical_batch_size: int
    gradient_accumulation_steps: int
    view_key: str
    adapter: str
    cpu_capable: bool = True
    gpu_capable: bool = True
    exact_resume: bool = True
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("job_id", "model_family", "sample_stream_key", "batch_contract_key", "view_key", "adapter"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if not self.datasets or any(not str(value).strip() for value in self.datasets):
            raise ValueError(f"{self.job_id}: datasets must be non-empty")
        if self.physical_batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError(f"{self.job_id}: physical batch and accumulation must be positive")
        if not self.cpu_capable:
            raise ValueError(f"{self.job_id}: CPU variant is mandatory")
        if not self.exact_resume:
            raise ValueError(f"{self.job_id}: exact resume is mandatory")

    @property
    def normalized_datasets(self) -> tuple[str, ...]:
        return tuple(sorted(dict.fromkeys(str(value).strip() for value in self.datasets)))

    @property
    def compatibility_key(self) -> tuple[str, str, int, int]:
        return (
            self.sample_stream_key,
            self.batch_contract_key,
            self.physical_batch_size,
            self.gradient_accumulation_steps,
        )


@dataclasses.dataclass(frozen=True)
class Subcohort:
    index: int
    parent_index: int
    parent_dataset_key: str
    sample_stream_key: str
    batch_contract_key: str
    physical_batch_size: int
    gradient_accumulation_steps: int
    models: tuple[Candidate, ...]

    @property
    def distinct_model_families(self) -> int:
        return len({row.model_family for row in self.models})

    @property
    def gpu_capable(self) -> bool:
        return all(row.gpu_capable for row in self.models)


@dataclasses.dataclass(frozen=True)
class DatasetGroup:
    index: int
    dataset_key: str
    datasets: tuple[str, ...]
    overlap: bool
    distinct_model_families: int
    model_count: int
    subcohorts: tuple[Subcohort, ...]


@dataclasses.dataclass(frozen=True)
class CompatibilityPlan:
    groups: tuple[DatasetGroup, ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "digest": self.digest,
            "groups": [
                {
                    "index": group.index,
                    "dataset_key": group.dataset_key,
                    "datasets": list(group.datasets),
                    "overlap": group.overlap,
                    "distinct_model_families": group.distinct_model_families,
                    "model_count": group.model_count,
                    "subcohorts": [
                        {
                            "index": sub.index,
                            "parent_index": sub.parent_index,
                            "sample_stream_key": sub.sample_stream_key,
                            "batch_contract_key": sub.batch_contract_key,
                            "physical_batch_size": sub.physical_batch_size,
                            "gradient_accumulation_steps": sub.gradient_accumulation_steps,
                            "distinct_model_families": sub.distinct_model_families,
                            "gpu_capable": sub.gpu_capable,
                            "models": [dataclasses.asdict(model) for model in sub.models],
                        }
                        for sub in group.subcohorts
                    ],
                }
                for group in self.groups
            ],
        }


def build_plan(candidates: Sequence[Candidate]) -> CompatibilityPlan:
    seen: set[str] = set()
    single: dict[str, list[Candidate]] = defaultdict(list)
    overlap: list[Candidate] = []
    for row in candidates:
        if row.job_id in seen:
            raise ValueError(f"duplicate job_id: {row.job_id}")
        seen.add(row.job_id)
        datasets = row.normalized_datasets
        if len(datasets) == 1:
            single[datasets[0]].append(row)
        else:
            overlap.append(row)

    def parent_rank(item: tuple[str, list[Candidate]]) -> tuple[int, int, str]:
        key, rows = item
        return (-len({row.model_family for row in rows}), -len(rows), key)

    parents: list[tuple[str, tuple[str, ...], bool, list[Candidate]]] = [
        (key, (key,), False, rows)
        for key, rows in sorted(single.items(), key=parent_rank)
    ]
    if overlap:
        parents.append((OVERLAP, tuple(sorted({dataset for row in overlap for dataset in row.normalized_datasets})), True, overlap))

    groups: list[DatasetGroup] = []
    global_sub_index = 0
    for parent_index, (key, datasets, is_overlap, rows) in enumerate(parents, start=1):
        buckets: dict[tuple[str, str, int, int], list[Candidate]] = defaultdict(list)
        for row in rows:
            buckets[row.compatibility_key].append(row)
        ordered_buckets = sorted(
            buckets.items(),
            key=lambda item: (
                -len({row.model_family for row in item[1]}),
                -len(item[1]),
                item[0][0], item[0][1], item[0][2], item[0][3],
            ),
        )
        subcohorts: list[Subcohort] = []
        for compatibility, members in ordered_buckets:
            global_sub_index += 1
            sample_stream, batch_contract, batch_size, accumulation = compatibility
            ordered = tuple(sorted(members, key=lambda row: (row.model_family, row.job_id)))
            subcohorts.append(
                Subcohort(
                    index=global_sub_index,
                    parent_index=parent_index,
                    parent_dataset_key=key,
                    sample_stream_key=sample_stream,
                    batch_contract_key=batch_contract,
                    physical_batch_size=batch_size,
                    gradient_accumulation_steps=accumulation,
                    models=ordered,
                )
            )
        groups.append(
            DatasetGroup(
                index=parent_index,
                dataset_key=key,
                datasets=datasets,
                overlap=is_overlap,
                distinct_model_families=len({row.model_family for row in rows}),
                model_count=len(rows),
                subcohorts=tuple(subcohorts),
            )
        )

    serial = [
        {
            "job_id": row.job_id,
            "model_family": row.model_family,
            "datasets": list(row.normalized_datasets),
            "sample_stream_key": row.sample_stream_key,
            "batch_contract_key": row.batch_contract_key,
            "physical_batch_size": row.physical_batch_size,
            "gradient_accumulation_steps": row.gradient_accumulation_steps,
            "view_key": row.view_key,
            "adapter": row.adapter,
            "cpu_capable": row.cpu_capable,
            "gpu_capable": row.gpu_capable,
            "exact_resume": row.exact_resume,
            "metadata": dict(row.metadata),
        }
        for row in sorted(candidates, key=lambda item: item.job_id)
    ]
    plan = CompatibilityPlan(tuple(groups), _digest(serial))
    # Structural assertions make ordering regressions fail immediately.
    non_overlap = [group for group in plan.groups if not group.overlap]
    counts = [group.distinct_model_families for group in non_overlap]
    if counts != sorted(counts, reverse=True):
        raise RuntimeError("dataset groups are not ordered by descending distinct model-family count")
    if any(group.overlap for group in plan.groups[:-1]):
        raise RuntimeError("overlap dataset group must be last")
    return plan


def explain_split(a: Candidate, b: Candidate) -> tuple[str, ...]:
    reasons: list[str] = []
    if a.normalized_datasets != b.normalized_datasets:
        reasons.append("dataset identity differs")
    if a.sample_stream_key != b.sample_stream_key:
        reasons.append("sample stream/split/fold/sampler identity differs")
    if a.batch_contract_key != b.batch_contract_key:
        reasons.append("optimizer-facing batch contract differs")
    if a.physical_batch_size != b.physical_batch_size:
        reasons.append("physical batch size differs")
    if a.gradient_accumulation_steps != b.gradient_accumulation_steps:
        reasons.append("gradient accumulation differs")
    return tuple(reasons)


__all__ = ["Candidate", "CompatibilityPlan", "DatasetGroup", "OVERLAP", "SCHEMA", "Subcohort", "build_plan", "explain_split"]
