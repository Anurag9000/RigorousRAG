#!/usr/bin/env python3
"""Closed-world RigorousRAG scientific workload catalog for the universal OPF controller.

The catalog contains no scheduler.  It enumerates the repository's authoritative trainable
families and post-training lifecycle, validates every local source/config target, validates the
scientific identity encoded by every recipe, rejects duplicate IDs/commands and invalid DAGs,
and returns ordinary job dictionaries to the shared v23 controller.  The literal pinned OPF_ADP
runner remains the sole CPU/GPU/RAM/VRAM/concurrency/process-control implementation.

The built-in scientific matrix is intentionally closed rather than a Cartesian-product guess:

* four classical learners: fusion weight, ListNet fusion, domain classifier, plan ranker;
* nine retrieval recipes: dense/SPLADE/uniCOIL/ColBERT base+distilled where implemented, plus
  the listwise cross-encoder;
* four advanced-RAG recipes: grounded generation and dynamic policy, each with causal-LM and
  seq2seq-LM generator families;
* seven deterministic calibration/materialization algorithms.

Each training job is followed by artifact-bound validation, artifact/readiness testing, and
metrics materialization.  Those post jobs do not invent a held-out scientific test dataset.
External governed import/materialization/benchmark/release operations are accepted through the
suite manifest as explicit command jobs, because their source paths, SHA-256 identities,
licenses, benchmark cohorts and promotion policies are operator data rather than constants this
repository can truthfully fabricate.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "rigorousrag-authoritative-training-suite/v1"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SUITE = "config/training_suite.example.json"
_ALLOWED_PHASES = frozenset({"setup", "preprocess", "training", "validation", "testing", "metrics"})

CLASSICAL_SOURCE = "training/authoritative_classical_training_cli_v3.py"
RETRIEVAL_SOURCE = "training/authoritative_retrieval_training_cli_v2.py"
ADVANCED_SOURCE = "training/authoritative_advanced_training_cli.py"
POSTPROCESS_SOURCE = "training/authoritative_training_suite_postprocess.py"
CALIBRATION_SOURCE = "training/authoritative_calibration_cli.py"


@dataclass(frozen=True)
class TrainingRecipe:
    job_id: str
    family: str
    architecture: str
    model: str
    dataset: str
    task: str
    config: str
    source: str
    expected_kind: str | None = None
    expected_variant: str | None = None
    expected_generator_kind: str | None = None
    device_capable: bool = True


TRAINING_RECIPES: tuple[TrainingRecipe, ...] = (
    TrainingRecipe("train:classical:fusion-weight", "classical", "fusion_weight", "weighted-score-fusion", "classical-fusion", "score-fusion", "config/classical_fusion_training.example.json", CLASSICAL_SOURCE, expected_kind="fusion_weight", device_capable=False),
    TrainingRecipe("train:classical:listwise-fusion", "classical", "listwise_fusion", "listnet-fusion", "classical-fusion", "listwise-score-fusion", "config/classical_listwise_fusion_training.example.json", CLASSICAL_SOURCE, expected_kind="listwise_fusion", device_capable=False),
    TrainingRecipe("train:classical:domain-classifier", "classical", "domain_classifier", "domain-classifier", "classical-domain", "domain-classification", "config/classical_domain_training.example.json", CLASSICAL_SOURCE, expected_kind="domain_classifier", device_capable=False),
    TrainingRecipe("train:classical:plan-ranker", "classical", "plan_ranker", "plan-ranker", "classical-plan-ranking", "plan-ranking", "config/classical_plan_ranker_training.example.json", CLASSICAL_SOURCE, expected_kind="plan_ranker", device_capable=False),
    TrainingRecipe("train:retrieval:dense-base", "retrieval", "dense", "dense-bi-encoder", "retrieval", "retrieval", "config/retrieval_dense_base_training.example.json", RETRIEVAL_SOURCE, expected_variant="base"),
    TrainingRecipe("train:retrieval:dense-distilled", "retrieval", "dense", "dense-bi-encoder", "retrieval", "retrieval-distillation", "config/retrieval_dense_distilled_training.example.json", RETRIEVAL_SOURCE, expected_variant="distilled"),
    TrainingRecipe("train:retrieval:splade-base", "retrieval", "splade", "splade", "retrieval", "sparse-retrieval", "config/retrieval_splade_base_training.example.json", RETRIEVAL_SOURCE, expected_variant="base"),
    TrainingRecipe("train:retrieval:splade-distilled", "retrieval", "splade", "splade", "retrieval", "sparse-retrieval-distillation", "config/retrieval_splade_distilled_training.example.json", RETRIEVAL_SOURCE, expected_variant="distilled"),
    TrainingRecipe("train:retrieval:unicoil-base", "retrieval", "unicoil", "unicoil", "retrieval", "sparse-retrieval", "config/retrieval_unicoil_training.example.json", RETRIEVAL_SOURCE, expected_variant="base"),
    TrainingRecipe("train:retrieval:unicoil-distilled", "retrieval", "unicoil", "unicoil", "retrieval", "sparse-retrieval-distillation", "config/retrieval_unicoil_distilled_training.example.json", RETRIEVAL_SOURCE, expected_variant="distilled"),
    TrainingRecipe("train:retrieval:colbert-base", "retrieval", "colbert", "colbert", "retrieval", "late-interaction-retrieval", "config/retrieval_colbert_base_training.example.json", RETRIEVAL_SOURCE, expected_variant="base"),
    TrainingRecipe("train:retrieval:colbert-distilled", "retrieval", "colbert", "colbert", "retrieval", "late-interaction-retrieval-distillation", "config/retrieval_colbert_distilled_training.example.json", RETRIEVAL_SOURCE, expected_variant="distilled"),
    TrainingRecipe("train:retrieval:cross-encoder-listwise", "retrieval", "cross_encoder", "listwise-cross-encoder", "retrieval", "listwise-reranking", "config/retrieval_cross_encoder_training.example.json", RETRIEVAL_SOURCE, expected_variant="listwise"),
    TrainingRecipe("train:advanced:grounded-causal", "advanced", "grounded_generation", "grounded-causal-lm", "grounded-canonical-v2", "grounded-generation", "config/advanced_grounded_training.example.json", ADVANCED_SOURCE, expected_kind="grounded_generation", expected_generator_kind="causal_lm"),
    TrainingRecipe("train:advanced:grounded-seq2seq", "advanced", "grounded_generation", "grounded-seq2seq-lm", "grounded-canonical-v2", "grounded-generation", "config/advanced_grounded_seq2seq_training.example.json", ADVANCED_SOURCE, expected_kind="grounded_generation", expected_generator_kind="seq2seq_lm"),
    TrainingRecipe("train:advanced:dynamic-causal", "advanced", "dynamic_rag_policy", "dynamic-rag-policy-causal-generator", "dynamic-canonical-v2", "dynamic-retrieval-policy", "config/advanced_dynamic_rag_training.example.json", ADVANCED_SOURCE, expected_kind="dynamic_rag_policy", expected_generator_kind="causal_lm"),
    TrainingRecipe("train:advanced:dynamic-seq2seq", "advanced", "dynamic_rag_policy", "dynamic-rag-policy-seq2seq-generator", "dynamic-canonical-v2", "dynamic-retrieval-policy", "config/advanced_dynamic_rag_seq2seq_training.example.json", ADVANCED_SOURCE, expected_kind="dynamic_rag_policy", expected_generator_kind="seq2seq_lm"),
)

CALIBRATION_CONFIGS: Mapping[str, str] = {
    "evaluation_histogram": "config/calibration_histogram.example.json",
    "evaluation_conformal_nonconformity": "config/calibration_conformal_nonconformity.example.json",
    "evaluation_conformal_retrieval": "config/calibration_conformal_retrieval.example.json",
    "retrieval_split_conformal": "config/calibration_retrieval_split_conformal.example.json",
    "threshold_decision": "config/calibration_threshold_decision.example.json",
    "confidence_isotonic": "config/calibration_confidence_isotonic.example.json",
    "cross_profile_isotonic": "config/calibration_cross_profile_isotonic.example.json",
}

_EXPECTED_TRAINING_IDS = frozenset(recipe.job_id for recipe in TRAINING_RECIPES)
_EXPECTED_CALIBRATIONS = frozenset(CALIBRATION_CONFIGS)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _safe_local(relative: str, label: str, *, must_exist: bool = True) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    raw = Path(relative)
    if raw.is_absolute():
        raise ValueError(f"{label} must stay repository-relative")
    candidate = (_REPO_ROOT / raw).absolute()
    try:
        candidate.relative_to(_REPO_ROOT.absolute())
    except Exception as exc:
        raise ValueError(f"{label} escapes repository root: {relative}") from exc
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"{label} traverses a symlink: {current}")
    resolved = candidate.resolve(strict=must_exist)
    try:
        resolved.relative_to(_REPO_ROOT.resolve())
    except Exception as exc:
        raise ValueError(f"{label} resolves outside repository root: {relative}") from exc
    if must_exist and not resolved.is_file():
        raise ValueError(f"{label} must resolve to an existing regular file: {relative}")
    return resolved


def _read_json(relative: str, label: str) -> Mapping[str, Any]:
    path = _safe_local(relative, label)
    if path.stat().st_size <= 0 or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(f"{label} exceeds the JSON safety bound")
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON: {relative}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _suite_path() -> str:
    requested = os.environ.get("RIGOROUSRAG_TRAINING_SUITE_CONFIG", "").strip()
    return requested or _DEFAULT_SUITE


def _load_suite() -> Mapping[str, Any]:
    path = _suite_path()
    raw = _read_json(path, "training suite config")
    allowed = {"schema", "training_config_overrides", "calibration_config_overrides", "lifecycle_jobs"}
    if set(raw) - allowed:
        raise ValueError(f"training suite config contains unsupported fields: {sorted(set(raw)-allowed)}")
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"training suite config schema must be {SCHEMA!r}")
    for key in ("training_config_overrides", "calibration_config_overrides"):
        if not isinstance(raw.get(key, {}), Mapping):
            raise ValueError(f"{key} must be an object")
    if not isinstance(raw.get("lifecycle_jobs", []), list):
        raise ValueError("lifecycle_jobs must be an array")
    return raw


def _training_config(recipe: TrainingRecipe, suite: Mapping[str, Any]) -> str:
    overrides = suite.get("training_config_overrides", {})
    assert isinstance(overrides, Mapping)
    unknown = set(str(key) for key in overrides) - _EXPECTED_TRAINING_IDS
    if unknown:
        raise ValueError(f"training_config_overrides names unknown jobs: {sorted(unknown)}")
    selected = str(overrides.get(recipe.job_id, recipe.config))
    raw = _read_json(selected, f"training config for {recipe.job_id}")
    if recipe.family == "classical":
        if str(raw.get("kind") or "") != recipe.expected_kind:
            raise ValueError(f"{recipe.job_id} config kind differs from catalog identity")
    elif recipe.family == "retrieval":
        architecture = str(raw.get("architecture") or "").lower()
        variant = str(raw.get("step_variant", "base")).lower()
        if architecture != recipe.architecture or variant != recipe.expected_variant:
            raise ValueError(f"{recipe.job_id} retrieval architecture/variant differs from catalog identity")
    else:
        if str(raw.get("kind") or "") != recipe.expected_kind:
            raise ValueError(f"{recipe.job_id} advanced kind differs from catalog identity")
        binding_name = "base_model" if recipe.expected_kind == "grounded_generation" else "generator"
        binding = raw.get(binding_name)
        if not isinstance(binding, Mapping) or str(binding.get("kind") or "") != recipe.expected_generator_kind:
            raise ValueError(f"{recipe.job_id} generator family differs from catalog identity")
    return selected


def _training_record(recipe: TrainingRecipe, config: str, setup_dependencies: Sequence[str]) -> dict[str, Any]:
    _safe_local(recipe.source, f"training source for {recipe.job_id}")
    return {
        "id": recipe.job_id,
        "command": [recipe.source, "train", "--config", config],
        "entrypoint_source": recipe.source,
        "phase": "training",
        "family": recipe.family,
        "repeat_index": 0,
        "device_capable": recipe.device_capable,
        "is_training_job": True,
        "model": recipe.model,
        "architecture": recipe.architecture,
        "dataset": recipe.dataset,
        "task": recipe.task,
        "recipe": recipe.job_id,
        "recipe_config": config,
        "depends_on": list(setup_dependencies),
        "early_stopping": True,
        "resume_strategy": "native_transactional_state" if recipe.family == "classical" else "exact_checkpoint",
        "scientific_surface": "authoritative-repository-training",
    }


def _post_records(recipe: TrainingRecipe, config: str) -> list[dict[str, Any]]:
    _safe_local(POSTPROCESS_SOURCE, "postprocess source")
    slug = recipe.job_id.replace(":", "__")
    validation_id = f"validate:{recipe.job_id.removeprefix('train:')}"
    test_id = f"test:{recipe.job_id.removeprefix('train:')}"
    metrics_id = f"metrics:{recipe.job_id.removeprefix('train:')}"
    common = {
        "entrypoint_source": POSTPROCESS_SOURCE,
        "family": recipe.family,
        "repeat_index": 0,
        "device_capable": False,
        "is_training_job": False,
        "model": recipe.model,
        "architecture": recipe.architecture,
        "dataset": recipe.dataset,
        "task": recipe.task,
        "recipe": recipe.job_id,
        "recipe_config": config,
        "resume_strategy": "restart_exact",
        "checkpoint_contract": {"exact_resume": True, "deterministic": True, "idempotent": True, "atomic_outputs": True},
        "early_stopping_applicable": False,
        "early_stopping_exception_reason": "post-training non-optimization lifecycle phase",
    }
    return [
        {
            **common,
            "id": validation_id,
            "phase": "validation",
            "command": [POSTPROCESS_SOURCE, "validate", "--family", recipe.family, "--config", config, "--training-job-id", recipe.job_id, "--output", f"artifacts/training_suite/validation/{slug}.json"],
            "depends_on": [recipe.job_id],
        },
        {
            **common,
            "id": test_id,
            "phase": "testing",
            "command": [POSTPROCESS_SOURCE, "test", "--family", recipe.family, "--config", config, "--training-job-id", recipe.job_id, "--output", f"artifacts/training_suite/testing/{slug}.json"],
            "depends_on": [validation_id],
            "testing_semantics": "artifact-readiness; no undeclared held-out dataset fabricated",
        },
        {
            **common,
            "id": metrics_id,
            "phase": "metrics",
            "command": [POSTPROCESS_SOURCE, "metrics", "--family", recipe.family, "--config", config, "--training-job-id", recipe.job_id, "--output", f"artifacts/training_suite/metrics/{slug}.json"],
            "depends_on": [test_id],
        },
    ]


def _external_lifecycle_jobs(suite: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(suite.get("lifecycle_jobs", [])):
        if not isinstance(raw, Mapping):
            raise ValueError(f"lifecycle_jobs[{index}] must be an object")
        allowed = {"id", "enabled", "phase", "family", "command", "entrypoint_source", "device_capable", "depends_on", "dataset", "task", "model", "architecture", "description"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"lifecycle_jobs[{index}] has unsupported fields: {sorted(unknown)}")
        if raw.get("enabled", False) is not True:
            continue
        job_id = str(raw.get("id") or "").strip()
        phase = str(raw.get("phase") or "").strip().lower()
        command = raw.get("command")
        source = str(raw.get("entrypoint_source") or "").strip()
        if not job_id or phase not in _ALLOWED_PHASES or phase == "training":
            raise ValueError(f"lifecycle_jobs[{index}] must have id and a non-training supported phase")
        if not isinstance(command, list) or not command or any(not isinstance(value, str) or not value for value in command):
            raise ValueError(f"lifecycle_jobs[{index}].command must be a non-empty string array")
        if not source:
            source = command[0]
        _safe_local(source, f"lifecycle source for {job_id}")
        dependencies = raw.get("depends_on", []) or []
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        if not isinstance(dependencies, list):
            raise ValueError(f"lifecycle_jobs[{index}].depends_on must be an array")
        record: dict[str, Any] = {
            "id": job_id,
            "command": list(command),
            "entrypoint_source": source,
            "phase": phase,
            "family": str(raw.get("family") or f"rigorousrag-{phase}"),
            "repeat_index": 0,
            "device_capable": bool(raw.get("device_capable", phase in {"validation", "testing"})),
            "depends_on": [str(value) for value in dependencies],
            "is_training_job": False,
            "resume_strategy": "restart_exact",
            "checkpoint_contract": {"exact_resume": True, "deterministic": True, "idempotent": True, "atomic_outputs": True},
            "early_stopping_applicable": False,
            "early_stopping_exception_reason": "non-training configured lifecycle phase",
            "configured_lifecycle_job": True,
        }
        for key in ("dataset", "task", "model", "architecture", "description"):
            if key in raw:
                record[key] = raw[key]
        rows.append(record)
    return rows


def _calibration_records(suite: Mapping[str, Any], all_metrics: Sequence[str], retrieval_metrics: Sequence[str]) -> list[dict[str, Any]]:
    _safe_local(CALIBRATION_SOURCE, "calibration source")
    overrides = suite.get("calibration_config_overrides", {})
    assert isinstance(overrides, Mapping)
    unknown = set(str(key) for key in overrides) - _EXPECTED_CALIBRATIONS
    if unknown:
        raise ValueError(f"calibration_config_overrides names unknown kinds: {sorted(unknown)}")
    rows: list[dict[str, Any]] = []
    retrieval_specific = {"evaluation_conformal_retrieval", "retrieval_split_conformal", "cross_profile_isotonic"}
    for kind, default_config in CALIBRATION_CONFIGS.items():
        config = str(overrides.get(kind, default_config))
        raw = _read_json(config, f"calibration config {kind}")
        if raw.get("schema") != "rigorousrag-authoritative-calibration-job/v1" or raw.get("kind") != kind:
            raise ValueError(f"calibration config for {kind} has the wrong schema/kind")
        dependencies = retrieval_metrics if kind in retrieval_specific else all_metrics
        rows.append(
            {
                "id": f"calibration:{kind}",
                "command": [CALIBRATION_SOURCE, "fit", "--config", config],
                "entrypoint_source": CALIBRATION_SOURCE,
                "phase": "metrics",
                "family": "calibration",
                "repeat_index": 0,
                "device_capable": False,
                "is_training_job": False,
                "dataset": "held-out-calibration-scores",
                "task": kind,
                "recipe": kind,
                "recipe_config": config,
                "depends_on": list(dependencies),
                "resume_strategy": "restart_exact",
                "checkpoint_contract": {"exact_resume": True, "deterministic": True, "idempotent": True, "atomic_outputs": True},
                "early_stopping_applicable": False,
                "early_stopping_exception_reason": "deterministic calibration/materialization, not iterative training",
            }
        )
    return rows


def _validate_graph(records: Sequence[Mapping[str, Any]]) -> None:
    ids = [str(row.get("id")) for row in records]
    if len(ids) != len(set(ids)):
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        raise ValueError(f"training suite contains duplicate job IDs: {duplicates}")
    commands = [tuple(str(value) for value in row.get("command", [])) for row in records]
    if len(commands) != len(set(commands)):
        raise ValueError("training suite contains duplicate executable commands")
    known = set(ids)
    dependencies: dict[str, set[str]] = {}
    for row in records:
        job_id = str(row["id"])
        raw = row.get("depends_on", []) or []
        if isinstance(raw, str):
            raw = [raw]
        deps = {str(value) for value in raw}
        unknown = sorted(deps - known)
        if job_id in deps:
            unknown.append(job_id)
        if unknown:
            raise ValueError(f"job {job_id} has dangling/self dependencies: {sorted(set(unknown))}")
        dependencies[job_id] = deps
    indegree = {job_id: len(deps) for job_id, deps in dependencies.items()}
    children: dict[str, list[str]] = {job_id: [] for job_id in ids}
    for job_id, deps in dependencies.items():
        for dependency in deps:
            children[dependency].append(job_id)
    ready = deque(sorted(job_id for job_id, degree in indegree.items() if degree == 0))
    visited: list[str] = []
    while ready:
        job_id = ready.popleft()
        visited.append(job_id)
        for child in sorted(children[job_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(visited) != len(ids):
        cyclic = sorted(job_id for job_id, degree in indegree.items() if degree > 0)
        raise ValueError(f"training suite DAG contains a cycle: {cyclic}")


def _validate_closed_world(records: Sequence[Mapping[str, Any]]) -> None:
    training_ids = {str(row["id"]) for row in records if str(row.get("phase")) == "training"}
    if training_ids != _EXPECTED_TRAINING_IDS:
        raise ValueError(
            "authoritative training matrix drifted; "
            f"missing={sorted(_EXPECTED_TRAINING_IDS-training_ids)}, unexpected={sorted(training_ids-_EXPECTED_TRAINING_IDS)}"
        )
    calibration_ids = {
        str(row["id"]).removeprefix("calibration:")
        for row in records
        if str(row["id"]).startswith("calibration:")
    }
    if calibration_ids != _EXPECTED_CALIBRATIONS:
        raise ValueError(
            "authoritative calibration matrix drifted; "
            f"missing={sorted(_EXPECTED_CALIBRATIONS-calibration_ids)}, unexpected={sorted(calibration_ids-_EXPECTED_CALIBRATIONS)}"
        )


def iter_jobs(mode: str = "exhaustive") -> Iterable[dict[str, Any]]:
    selected_mode = str(mode).strip().lower()
    if selected_mode != "exhaustive":
        raise ValueError("RigorousRAG authoritative suite supports only mode='exhaustive'")
    suite = _load_suite()
    external = _external_lifecycle_jobs(suite)
    setup_dependencies = [
        str(row["id"]) for row in external if str(row.get("phase")) in {"setup", "preprocess"}
    ]
    records: list[dict[str, Any]] = list(external)
    training_rows: list[dict[str, Any]] = []
    post_rows: list[dict[str, Any]] = []
    for recipe in TRAINING_RECIPES:
        config = _training_config(recipe, suite)
        training = _training_record(recipe, config, setup_dependencies)
        training_rows.append(training)
        post_rows.extend(_post_records(recipe, config))
    records.extend(training_rows)
    records.extend(post_rows)
    all_metrics = [str(row["id"]) for row in post_rows if str(row.get("phase")) == "metrics"]
    retrieval_metrics = [str(row["id"]) for row in post_rows if str(row.get("phase")) == "metrics" and str(row.get("family")) == "retrieval"]
    records.extend(_calibration_records(suite, all_metrics, retrieval_metrics))
    _validate_closed_world(records)
    _validate_graph(records)
    suite_digest = _sha(
        {
            "schema": SCHEMA,
            "suite_config": _suite_path(),
            "training_ids": sorted(_EXPECTED_TRAINING_IDS),
            "calibration_kinds": sorted(_EXPECTED_CALIBRATIONS),
            "jobs": [{"id": row["id"], "phase": row["phase"], "command": row["command"], "depends_on": row.get("depends_on", [])} for row in records],
        }
    )
    for row in records:
        row["authoritative_suite_schema"] = SCHEMA
        row["authoritative_suite_digest"] = suite_digest
    return tuple(records)


__all__ = ["CALIBRATION_CONFIGS", "SCHEMA", "TRAINING_RECIPES", "TrainingRecipe", "iter_jobs"]
