#!/usr/bin/env python3
"""Physical dataset-cohort compilation of the closed-world RigorousRAG suite.

The v1 catalog remains the logical scientific inventory.  This authority compiles
that inventory into physical OPF jobs.  Learned retrieval recipes sharing the
same governed dataset are represented by one GPU-first/CPU-capable synchronized
cohort transaction; their individual validation/testing/metrics jobs remain
separate and depend on the cohort transaction.  Classical and advanced jobs are
left byte-for-byte semantically unchanged until their dedicated cohort adapters
are admitted.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from training import authoritative_training_suite_catalog as logical

SCHEMA = "rigorousrag-authoritative-training-suite/v2-dataset-cohorts"
RETRIEVAL_COHORT_ID = "train:retrieval:dataset-cohort"
RETRIEVAL_COHORT_SOURCE = "training/run_rigorousrag_retrieval_cohort.py"
RETRIEVAL_POSTPROCESS_SOURCE = "training/authoritative_training_suite_postprocess_v2.py"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _source(relative: str) -> None:
    path = (_REPO_ROOT / relative).resolve(strict=True)
    path.relative_to(_REPO_ROOT.resolve())
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"physical cohort source is not a regular repository file: {relative}")


def _validate_graph(records: Sequence[Mapping[str, Any]]) -> None:
    ids = [str(row["id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("physical dataset-cohort catalog contains duplicate job IDs")
    commands = [tuple(str(value) for value in row.get("command", ())) for row in records]
    if len(commands) != len(set(commands)):
        raise ValueError("physical dataset-cohort catalog contains duplicate commands")
    known = set(ids)
    deps: dict[str, set[str]] = {}
    for row in records:
        raw = row.get("depends_on", ()) or ()
        raw = [raw] if isinstance(raw, str) else raw
        selected = {str(value) for value in raw}
        unknown = selected - known
        if unknown or str(row["id"]) in selected:
            raise ValueError(f"physical job {row['id']} has invalid dependencies: {sorted(unknown)}")
        deps[str(row["id"])] = selected
    indegree = {key: len(value) for key, value in deps.items()}
    children: dict[str, list[str]] = {key: [] for key in ids}
    for child, parents in deps.items():
        for parent in parents:
            children[parent].append(child)
    ready = deque(sorted(key for key, degree in indegree.items() if degree == 0))
    visited = 0
    while ready:
        parent = ready.popleft(); visited += 1
        for child in sorted(children[parent]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(ids):
        raise ValueError("physical dataset-cohort catalog contains a dependency cycle")


def _compile(logical_records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    retrieval_training = [
        dict(row) for row in logical_records
        if row.get("is_training_job") is True and str(row.get("family")) == "retrieval"
    ]
    if len(retrieval_training) != 9:
        raise ValueError(f"authoritative retrieval cohort requires exactly 9 logical members, got {len(retrieval_training)}")
    logical_ids = {str(row["id"]) for row in retrieval_training}
    expected = {
        "train:retrieval:dense-base", "train:retrieval:dense-distilled",
        "train:retrieval:splade-base", "train:retrieval:splade-distilled",
        "train:retrieval:unicoil-base", "train:retrieval:unicoil-distilled",
        "train:retrieval:colbert-base", "train:retrieval:colbert-distilled",
        "train:retrieval:cross-encoder-listwise",
    }
    if logical_ids != expected:
        raise ValueError(f"logical retrieval member drift: missing={sorted(expected-logical_ids)}, unexpected={sorted(logical_ids-expected)}")
    setup_dependencies = sorted({str(dep) for row in retrieval_training for dep in (row.get("depends_on") or ())})
    _source(RETRIEVAL_COHORT_SOURCE)
    _source(RETRIEVAL_POSTPROCESS_SOURCE)
    cohort = {
        "id": RETRIEVAL_COHORT_ID,
        "command": [RETRIEVAL_COHORT_SOURCE, "--backend", "auto"],
        "entrypoint_source": RETRIEVAL_COHORT_SOURCE,
        "phase": "training",
        "family": "retrieval",
        "model_family": "retrieval-dataset-cohort",
        "architecture": "shared-batch-multi-family",
        "model": "dense+splade+unicoil+colbert+cross-encoder",
        "dataset": "retrieval",
        "datasets": ["retrieval"],
        "task": "synchronized-multi-model-retrieval-training",
        "recipe": "dataset-cohort-v1",
        "recipe_config": "config/training_suite.example.json",
        "repeat_index": 0,
        "device_capable": True,
        "cpu_capable": True,
        "gpu_capable": True,
        "is_training_job": True,
        "depends_on": setup_dependencies,
        "early_stopping": True,
        "resume_strategy": "exact_checkpoint",
        "checkpoint_contract": {
            "exact_resume": True,
            "cohort_shared_batch_cursor": True,
            "per_model_optimizer_scheduler_scaler_rng": True,
            "atomic_outputs": True,
        },
        "dataset_cohort": True,
        "cohort_members": sorted(logical_ids),
        "cohort_member_count": len(logical_ids),
        "cohort_grouping_rule": "single-dataset groups descending by distinct model families; overlap group last",
        "cohort_uniform_batch": True,
        "cohort_shared_raw_batch": True,
        "cohort_shared_device_views": True,
        "cohort_gpu_first_cpu_fallback": True,
        "scientific_surface": "authoritative-repository-training-dataset-cohort",
    }
    output: list[dict[str, Any]] = [
        dict(row) for row in logical_records if str(row.get("id")) not in logical_ids
    ]
    output.append(cohort)
    for row in output:
        raw = row.get("depends_on", ()) or ()
        raw = [raw] if isinstance(raw, str) else list(raw)
        if any(str(value) in logical_ids for value in raw):
            row["depends_on"] = [
                RETRIEVAL_COHORT_ID if str(value) in logical_ids else str(value)
                for value in raw
            ]
            row["depends_on"] = list(dict.fromkeys(row["depends_on"]))
        if str(row.get("family")) == "retrieval" and str(row.get("phase")) in {"validation", "testing", "metrics"}:
            command = list(row.get("command") or ())
            if not command:
                raise ValueError(f"retrieval post job {row['id']} has no command")
            command[0] = RETRIEVAL_POSTPROCESS_SOURCE
            row["command"] = command
            row["entrypoint_source"] = RETRIEVAL_POSTPROCESS_SOURCE
            row["physical_training_job_id"] = RETRIEVAL_COHORT_ID
    output.sort(key=lambda row: (str(row.get("phase")), str(row.get("id"))))
    _validate_graph(output)
    return tuple(output)


def iter_jobs(mode: str = "exhaustive") -> Iterable[dict[str, Any]]:
    logical_records = tuple(dict(row) for row in logical.iter_jobs(mode))
    records = list(_compile(logical_records))
    digest = _sha({
        "schema": SCHEMA,
        "logical_schema": logical.SCHEMA,
        "logical_training_ids": sorted(str(row["id"]) for row in logical_records if row.get("is_training_job") is True),
        "physical_jobs": [
            {"id": row["id"], "phase": row["phase"], "command": row["command"], "depends_on": row.get("depends_on", [])}
            for row in records
        ],
    })
    for row in records:
        row["authoritative_suite_schema"] = SCHEMA
        row["authoritative_suite_digest"] = digest
        row["logical_authoritative_suite_schema"] = logical.SCHEMA
    return tuple(records)


__all__ = ["RETRIEVAL_COHORT_ID", "SCHEMA", "iter_jobs"]
