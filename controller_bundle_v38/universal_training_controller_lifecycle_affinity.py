#!/usr/bin/env python3
"""Semantic lifecycle dependency refinement for repository-wide orchestration.

The earlier lifecycle layer intentionally used a conservative fallback: when it
could not identify a matching predecessor it depended on every job in the prior
phase. That is safe but can create unnecessary barriers across unrelated datasets,
models or tasks. This layer preserves every explicit repository dependency and
rebuilds only inferred edges using model/dataset/task/config/family/source affinity.

No resource scheduling is implemented here. The resulting dependency graph is
still executed by the ordinary DAG layer, whose ready jobs are handed to the
literal pinned OPF_ADP scheduler.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import universal_training_controller_current as current
import universal_training_controller_lifecycle as lifecycle

AFFINITY_SCHEMA = 1
_CAPTURED = None

DIMENSIONS = {
    "model": ("model", "models", "architecture", "architectures", "arch"),
    "dataset": ("dataset", "datasets", "data", "corpus", "corpora"),
    "task": ("task", "tasks", "objective", "objectives", "benchmark", "benchmarks"),
    "config": ("recipe_config", "config", "configuration", "config_path", "config_file"),
    "domain": ("domain", "domains", "environment", "environments", "env"),
    "method": ("method", "methods", "algorithm", "algorithms"),
}
STOP = {
    "train", "training", "run", "main", "script", "scripts", "model", "models",
    "dataset", "datasets", "task", "tasks", "config", "configs", "configuration",
    "validation", "validate", "evaluation", "evaluate", "eval", "testing", "test",
    "inference", "infer", "prediction", "predict", "metrics", "metric", "report",
    "aggregate", "aggregation", "prepare", "preprocess", "preprocessing", "download",
    "setup", "build", "src", "tools", "tool", "python", "py", "json", "yaml", "yml",
    "toml", "generic", "lifecycle", "entrypoint", "job", "jobs",
}


def _flatten(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        out: list[str] = []
        for key, item in value.items():
            out.extend(_flatten(key)); out.extend(_flatten(item))
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        out = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [str(value)]


def _tokens(value: Any) -> set[str]:
    result: set[str] = set()
    for item in _flatten(value):
        for token in re.findall(r"[A-Za-z0-9]+", item.lower()):
            if len(token) >= 2 and token not in STOP and not token.isdigit():
                result.add(token)
    return result


def _dimension(record: Mapping[str, Any], name: str) -> set[str]:
    values: list[Any] = []
    for key in DIMENSIONS[name]:
        if key in record:
            values.append(record.get(key))
    return _tokens(values)


def _source_tokens(record: Mapping[str, Any]) -> set[str]:
    source = str(record.get("entrypoint_source") or record.get("id") or "")
    return _tokens(Path(source).with_suffix("").as_posix())


def _command_tokens(record: Mapping[str, Any]) -> set[str]:
    command = [str(x) for x in (record.get("command") or [])]
    interesting: list[str] = []
    for index, token in enumerate(command):
        if token.startswith("--"):
            if "=" in token:
                key, value = token.split("=", 1)
                if any(word in key.lower() for word in ("model", "dataset", "task", "config", "recipe", "domain", "env", "method", "algorithm")):
                    interesting.append(value)
            elif any(word in token.lower() for word in ("model", "dataset", "task", "config", "recipe", "domain", "env", "method", "algorithm")) and index + 1 < len(command):
                interesting.append(command[index + 1])
        elif Path(token).suffix.lower() in {".yaml", ".yml", ".json", ".toml"}:
            interesting.append(token)
    return _tokens(interesting)


def _family(record: Mapping[str, Any]) -> set[str]:
    value = str(record.get("family") or "")
    if value.lower().startswith("lifecycle-"):
        return set()
    return _tokens(value)


def _score(target: Mapping[str, Any], candidate: Mapping[str, Any]) -> int:
    score = 0
    weights = {"model": 120, "dataset": 120, "task": 110, "config": 90, "domain": 80, "method": 80}
    for dimension, weight in weights.items():
        left = _dimension(target, dimension)
        right = _dimension(candidate, dimension)
        if left and right and left & right:
            score += weight + 10 * min(3, len(left & right))
    tf = _family(target); cf = _family(candidate)
    if tf and cf and tf & cf:
        score += 55
    ts = _source_tokens(target); cs = _source_tokens(candidate)
    if ts and cs:
        overlap = ts & cs
        if overlap:
            score += min(45, 9 * len(overlap))
    tc = _command_tokens(target); cc = _command_tokens(candidate)
    if tc and cc:
        score += min(50, 10 * len(tc & cc))
    # Cross-match explicit dimensions against predecessor/source tokens. This is
    # especially useful for dataset preparation scripts named after a corpus.
    target_identity = set().union(*(_dimension(target, name) for name in ("model", "dataset", "task", "domain", "method")))
    candidate_identity = set().union(*(_dimension(candidate, name) for name in ("model", "dataset", "task", "domain", "method")))
    if target_identity & (cs | cc):
        score += min(60, 12 * len(target_identity & (cs | cc)))
    if candidate_identity & (ts | tc):
        score += min(60, 12 * len(candidate_identity & (ts | tc)))
    return score


def _matches(target: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> tuple[list[str], str, Dict[str, int]]:
    if not candidates:
        return [], "none", {}
    scores = {str(row.get("id")): _score(target, row) for row in candidates}
    best = max(scores.values()) if scores else 0
    if best >= 55:
        # Keep every strongly tied predecessor; multi-dataset/model ensembles may
        # legitimately need more than one upstream job.
        threshold = max(55, best - 15)
        chosen = [str(row.get("id")) for row in candidates if scores[str(row.get("id"))] >= threshold]
        return chosen, "semantic-affinity", scores
    if len(candidates) == 1:
        return [str(candidates[0].get("id"))], "single-predecessor", scores
    # Safety fallback: if the repository gives no identity evidence, preserve
    # the earlier all-predecessor barrier rather than guessing an unsafe edge.
    return [str(row.get("id")) for row in candidates], "conservative-all", scores


def _explicit_dependency_index(profile: Mapping[str, Any]) -> Dict[str, list[str]]:
    result: Dict[str, list[str]] = {}
    for collection_name in ("jobs", "extra_jobs", "validation_jobs", "testing_jobs", "metrics_jobs"):
        collection = profile.get(collection_name, []) or []
        if not isinstance(collection, list):
            continue
        for index, item in enumerate(collection):
            if not isinstance(item, Mapping):
                continue
            raw = item.get("depends_on")
            if raw is None:
                continue
            if isinstance(raw, str):
                raw = [raw]
            job_id = str(item.get("id") or item.get("name") or item.get("entrypoint") or f"{collection_name}-{index:04d}")
            result[job_id] = [str(x) for x in (raw or [])]
    return result


def _records(root: Path, profile: Dict[str, Any]):
    assert _CAPTURED is not None
    records = [dict(row) for row in _CAPTURED(root, profile)]
    explicit = _explicit_dependency_index(profile)
    by_phase: Dict[str, list[Dict[str, Any]]] = {}
    for row in records:
        by_phase.setdefault(lifecycle._normalized_phase(row), []).append(row)

    setup = by_phase.get("setup", [])
    preprocess = by_phase.get("preprocess", [])
    training = by_phase.get("training", [])
    validation = by_phase.get("validation", [])
    testing = by_phase.get("testing", [])
    metrics = by_phase.get("metrics", [])
    decisions: Dict[str, Dict[str, Any]] = {}

    def apply(row: Dict[str, Any], candidates: Sequence[Mapping[str, Any]]) -> None:
        job_id = str(row.get("id"))
        declared = explicit.get(job_id)
        if declared is not None:
            row["depends_on"] = sorted(set(declared))
            decisions[job_id] = {"strategy": "explicit", "selected": row["depends_on"], "scores": {}}
            return
        selected, strategy, scores = _matches(row, candidates)
        row["depends_on"] = sorted(set(selected))
        decisions[job_id] = {"strategy": strategy, "selected": row["depends_on"], "scores": scores}

    for row in setup:
        if str(row.get("id")) in explicit:
            apply(row, [])
        else:
            row["depends_on"] = []
            decisions[str(row.get("id"))] = {"strategy": "root", "selected": [], "scores": {}}
    for row in preprocess:
        apply(row, setup)
    for row in training:
        # A matched preprocess job carries its own setup dependency. If no
        # preprocessing phase exists, training matches setup directly.
        apply(row, preprocess or setup)
    for row in validation:
        apply(row, training or preprocess or setup)
    for row in testing:
        apply(row, validation or training or preprocess or setup)
    for row in metrics:
        apply(row, testing or validation or training or preprocess or setup)

    profile["lifecycle_affinity"] = {
        "schema": AFFINITY_SCHEMA,
        "enabled": True,
        "decisions": decisions,
        "semantic_jobs": sum(1 for row in decisions.values() if row.get("strategy") == "semantic-affinity"),
        "conservative_fallback_jobs": sum(1 for row in decisions.values() if row.get("strategy") == "conservative-all"),
    }
    return records


def install() -> None:
    global _CAPTURED
    if getattr(current._ORIGINAL_JOB_RECORDS, "_training_control_lifecycle_affinity", False):
        return
    _CAPTURED = current._ORIGINAL_JOB_RECORDS
    _records._training_control_lifecycle_affinity = True  # type: ignore[attr-defined]
    current._ORIGINAL_JOB_RECORDS = _records


__all__ = ["AFFINITY_SCHEMA", "install"]
