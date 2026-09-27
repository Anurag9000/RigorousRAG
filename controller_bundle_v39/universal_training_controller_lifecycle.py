#!/usr/bin/env python3
"""Full repository lifecycle orchestration above the literal OPF_ADP scheduler.

This module deliberately contains *no* resource scheduler.  It expands a
repository's concrete executable lifecycle surfaces into ordinary universal
controller jobs so the unchanged, byte-pinned OPF_ADP scheduler owns admission,
pressure handling, pause/resume, retries, GPU selection, logging, persistent
state and concurrency for every phase.

The lifecycle is:

    dataset/setup -> preprocess -> training -> validation/evaluation
                  -> testing/inference -> metrics/aggregation

Explicit ``profile.jobs`` and ``profile.job_catalog`` remain authoritative for
model/task matrices.  The discovery path is a fail-safe fallback for simpler
repositories and for executable lifecycle scripts not already represented by an
explicit catalog.  Hidden model classes are not guessed into synthetic commands;
they remain covered through the trainer/registry that imports them, as enforced
by the existing semantic source-accounting audit.
"""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import universal_training_controller as base
import universal_training_controller_current as current

LIFECYCLE_SCHEMA = 1
SCRIPT_SUFFIXES = {".py", ".sh", ".ps1", ".bat", ".cmd", ".r", ".jl"}
SKIP_PARTS = {
    ".git", ".training_control", "__pycache__", ".venv", "venv", "env",
    "node_modules", "results", "result", "outputs", "output", "checkpoints",
    "artifacts", "dist", "build", "docs", "doc", "tests", "test",
}

PHASE_ORDER = {
    "setup": 0,
    "dataset": 0,
    "download": 0,
    "materialize": 0,
    "preprocess": 1,
    "preprocessing": 1,
    "training": 2,
    "train": 2,
    "validation": 3,
    "validate": 3,
    "evaluation": 3,
    "evaluate": 3,
    "eval": 3,
    "testing": 4,
    "test": 4,
    "inference": 4,
    "infer": 4,
    "prediction": 4,
    "predict": 4,
    "metrics": 5,
    "metric": 5,
    "aggregation": 5,
    "aggregate": 5,
    "report": 5,
}

# Filename/stem discovery is intentionally conservative.  Unit-test trees are
# excluded above, while explicit catalogs can represent any non-conventional
# command without relying on these markers.
PHASE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("setup", re.compile(
        r"(?:^|[_\-.])(download|fetch|materiali[sz]e|dataset[_-]?setup|prepare[_-]?data|"
        r"prepare[_-]?dataset|build[_-]?dataset|dataset[_-]?ingest(?:ion)?|ingest[_-]?data)(?:[_\-.]|$)", re.I)),
    ("preprocess", re.compile(
        r"(?:^|[_\-.])(preprocess|preprocessing|prepare[_-]?(?:audio|documents|features|manifests)|"
        r"build[_-]?(?:features|folds|manifest|index)|tokeni[sz]e)(?:[_\-.]|$)", re.I)),
    ("training", re.compile(
        r"(?:^|[_\-.])(train|training|pretrain|finetune|fine[_-]?tune|fit|experiment|experiments|"
        r"sweep|search|optimi[sz]e|continual|adapt|ppo|dqn|reinforce|policy|rl)(?:[_\-.]|$)", re.I)),
    ("validation", re.compile(
        r"(?:^|[_\-.])(validate|validation|eval|evaluate|evaluation|benchmark)(?:[_\-.]|$)", re.I)),
    ("testing", re.compile(
        r"(?:^|[_\-.])(test[_-]?(?:model|checkpoint|policy|agent|pipeline)?|inference|infer|predict|"
        r"prediction|rollout)(?:[_\-.]|$)", re.I)),
    ("metrics", re.compile(
        r"(?:^|[_\-.])(metric|metrics|aggregate|aggregation|summari[sz]e|score|scoring|"
        r"collect[_-]?metrics|report[_-]?(?:results|metrics)?)(?:[_\-.]|$)", re.I)),
)

# Body markers make unconventional executable trainers/evaluators discoverable
# without treating ordinary imported model modules as executable jobs.
TRAIN_BODY = re.compile(
    r"torch\.optim|\.backward\s*\(|\bTrainer\s*\(|\.fit\s*\(|training_step|"
    r"\b(?:PPO|DQN|A2C|SAC|TD3)\s*\(|\.learn\s*\(", re.I,
)
EVAL_BODY = re.compile(
    r"model\.eval\s*\(|evaluate\s*\(|evaluation_loop|validation_step|val_dataloader|"
    r"classification_report\s*\(|mean_squared_error\s*\(|roc_auc_score\s*\(", re.I,
)
INFER_BODY = re.compile(
    r"torch\.no_grad\s*\(|inference_mode\s*\(|\.predict\s*\(|generate\s*\(", re.I,
)
METRIC_BODY = re.compile(
    r"accuracy_score\s*\(|f1_score\s*\(|precision_score\s*\(|recall_score\s*\(|"
    r"bleu|rouge|wer\s*\(|cer\s*\(|metrics?\s*=", re.I,
)
EARLY_STOP_BODY = re.compile(r"early[_ -]?stopp|EarlyStopping|stopping_rounds|patience", re.I)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _is_executable(path: Path, text: str) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "__name__" in text and "__main__" in text
    return suffix in SCRIPT_SUFFIXES


def _iter_scripts(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SCRIPT_SUFFIXES:
            continue
        rel = path.relative_to(root)
        if any(part.lower() in SKIP_PARTS for part in rel.parts):
            continue
        if rel.as_posix() == "run_all_training.py":
            continue
        yield path


def _phase_for(path: Path, text: str) -> str | None:
    stem = path.stem.lower()
    for phase, pattern in PHASE_PATTERNS:
        if pattern.search(stem):
            return phase
    # Body-based fallback is only used for executable scripts.
    if TRAIN_BODY.search(text):
        return "training"
    if EVAL_BODY.search(text):
        return "validation"
    if INFER_BODY.search(text):
        return "testing"
    if METRIC_BODY.search(text):
        return "metrics"
    return None


def _record_source(root: Path, record: Mapping[str, Any]) -> str | None:
    explicit = str(record.get("entrypoint_source") or "").replace("\\", "/").strip()
    if explicit and (root / explicit).is_file():
        return explicit
    for token in record.get("command", []) or []:
        try:
            candidate = Path(str(token))
            if candidate.is_absolute():
                return candidate.resolve().relative_to(root.resolve()).as_posix()
            local = root / candidate
            if local.is_file():
                return candidate.as_posix()
        except Exception:
            continue
    return None


def _command_key(record: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(str(x) for x in (record.get("command") or []))


def _source_key(root: Path, record: Mapping[str, Any]) -> str:
    return _record_source(root, record) or ""


def _normalized_phase(record: Mapping[str, Any]) -> str:
    phase = str(record.get("phase") or "training").strip().lower()
    aliases = {
        "train": "training", "pretrain": "training", "pretraining": "training",
        "finetune": "training", "fine-tune": "training", "fine_tune": "training",
        "adapt": "training", "fit": "training", "rl": "training",
        "dataset": "setup", "download": "setup", "materialize": "setup",
        "preprocessing": "preprocess", "validate": "validation",
        "evaluation": "validation", "evaluate": "validation", "eval": "validation",
        "test": "testing", "inference": "testing", "infer": "testing",
        "prediction": "testing", "predict": "testing",
        "metric": "metrics", "aggregation": "metrics", "aggregate": "metrics",
        "report": "metrics",
    }
    return aliases.get(phase, phase)


def _restart_exact_metadata() -> Dict[str, Any]:
    return {
        "is_training_job": False,
        "resume_strategy": "restart_exact",
        "checkpoint_contract": {
            "exact_resume": True,
            "deterministic": True,
            "idempotent": True,
            "atomic_outputs": True,
        },
        "deterministic": True,
        "idempotent": True,
        "atomic_outputs": True,
        "early_stopping_applicable": False,
        "early_stopping_exception_reason": "non-training lifecycle phase",
    }


def _discovered_record(root: Path, path: Path, phase: str, text: str) -> Dict[str, Any]:
    rel = path.relative_to(root).as_posix()
    record: Dict[str, Any] = {
        "id": f"lifecycle:{phase}:{rel}",
        "command": base._command_for_path(root, rel),
        "entrypoint_source": rel,
        "phase": phase,
        "family": f"lifecycle-{phase}",
        "repeat_index": 0,
        "device_capable": phase in {"training", "validation", "testing"},
        "lifecycle_discovered": True,
        "lifecycle_schema": LIFECYCLE_SCHEMA,
    }
    if phase == "training":
        record["is_training_job"] = True
        # Never fabricate early stopping.  We expose native source evidence and
        # let the existing strict training contract fail closed when a trainer
        # lacks it.  Custom catalogs may provide a richer explicit contract.
        if EARLY_STOP_BODY.search(text):
            record["early_stopping"] = True
    else:
        record.update(_restart_exact_metadata())
    return record


def _profile_declared_jobs(root: Path, profile: Mapping[str, Any], key: str, phase: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    values = profile.get(key, []) or []
    if not isinstance(values, list):
        raise SystemExit(f"{key} must be a list")
    for index, value in enumerate(values):
        if isinstance(value, str):
            if not (root / value).is_file():
                continue
            text = _read(root / value)
            rows.append(_discovered_record(root, root / value, phase, text))
            rows[-1]["id"] = f"profile:{phase}:{value}"
            rows[-1]["lifecycle_discovered"] = False
            continue
        if not isinstance(value, Mapping):
            raise SystemExit(f"{key}[{index}] must be a string or object")
        command_value = value.get("command") or value.get("entrypoint")
        if not command_value:
            raise SystemExit(f"{key}[{index}] has no command/entrypoint")
        record = {
            "id": str(value.get("id") or value.get("name") or f"{phase}-{index:04d}"),
            "command": base._normalize_command(root, command_value),
            "phase": phase,
            "family": str(value.get("family") or f"lifecycle-{phase}"),
            "repeat_index": int(value.get("repeat_index", 0)),
            "device_capable": bool(value.get("device_capable", phase in {"training", "validation", "testing"})),
            "lifecycle_discovered": False,
            "lifecycle_schema": LIFECYCLE_SCHEMA,
        }
        record.update({
            k: v for k, v in value.items()
            if k not in {"id", "name", "command", "entrypoint", "phase", "family", "repeat_index", "device_capable"}
        })
        if phase != "training":
            defaults = _restart_exact_metadata()
            for k, v in defaults.items():
                record.setdefault(k, v)
        rows.append(record)
    return rows


def _setup_records(root: Path, profile: MutableMapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, value in enumerate(profile.get("setup_commands", []) or []):
        record = {
            "id": f"setup-command-{index:04d}",
            "command": base._normalize_command(root, value),
            "phase": "setup",
            "family": "lifecycle-setup",
            "repeat_index": 0,
            "device_capable": False,
            "lifecycle_discovered": False,
            "lifecycle_schema": LIFECYCLE_SCHEMA,
        }
        record.update(_restart_exact_metadata())
        rows.append(record)
    for relative in profile.get("preferred_dataset_entrypoints", []) or []:
        rel = str(relative)
        path = root / rel
        if not path.is_file():
            continue
        phase = _phase_for(path, _read(path)) or "setup"
        if phase not in {"setup", "preprocess"}:
            phase = "setup"
        record = _discovered_record(root, path, phase, _read(path))
        record["id"] = f"dataset:{rel}"
        record["lifecycle_discovered"] = False
        rows.append(record)
    if rows:
        # Prevent base/dag from executing these again outside OPF scheduling.
        profile["_training_control_lifecycle_setup_scheduled"] = True
    return rows


def _cohort_name(record: Mapping[str, Any]) -> str:
    source = str(record.get("entrypoint_source") or record.get("id") or "").lower()
    stem = Path(source).stem
    stem = re.sub(
        r"(?:^|[_-])(?:train(?:ing)?|pretrain|finetune|fine[_-]?tune|validate|validation|"
        r"eval(?:uate|uation)?|test|testing|inference|infer|predict|prediction|metrics?|"
        r"aggregate|aggregation|report|run)(?:[_-]|$)",
        "_", stem,
    )
    stem = re.sub(r"[_-]+", "_", stem).strip("_")
    return stem


def _add_dependencies(records: List[Dict[str, Any]]) -> None:
    by_phase: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        by_phase.setdefault(_normalized_phase(record), []).append(record)

    setup_ids = [str(r["id"]) for p in ("setup", "preprocess") for r in by_phase.get(p, [])]
    training = by_phase.get("training", [])
    validation = by_phase.get("validation", [])
    testing = by_phase.get("testing", [])
    metrics = by_phase.get("metrics", [])

    def merge(record: Dict[str, Any], deps: Iterable[str]) -> None:
        existing = record.get("depends_on", []) or []
        if isinstance(existing, str):
            existing = [existing]
        record["depends_on"] = sorted(set(str(x) for x in [*existing, *deps] if str(x) != str(record["id"])))

    for record in training:
        merge(record, setup_ids)

    def matching_dependencies(target: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> List[str]:
        if not candidates:
            return []
        cohort = _cohort_name(target)
        matches = [str(r["id"]) for r in candidates if cohort and _cohort_name(r) == cohort]
        if matches:
            return matches
        if len(candidates) == 1:
            return [str(candidates[0]["id"])]
        # A generic evaluator/tester/aggregator commonly consumes the whole
        # experiment set; depending on every prior job is safer than racing a
        # partially produced result tree. Explicit catalogs can express finer
        # parallelism and always take precedence over these inferred edges.
        return [str(r["id"]) for r in candidates]

    for record in validation:
        merge(record, matching_dependencies(record, training) or setup_ids)
    for record in testing:
        merge(record, matching_dependencies(record, validation) or matching_dependencies(record, training) or setup_ids)
    for record in metrics:
        predecessors: List[Dict[str, Any]] = testing or validation or training
        merge(record, matching_dependencies(record, predecessors) or setup_ids)


def _dedupe(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    answer: List[Dict[str, Any]] = []
    seen_commands: set[Tuple[str, ...]] = set()
    seen_ids: set[str] = set()
    for record in records:
        command = _command_key(record)
        job_id = str(record.get("id"))
        if not command or command in seen_commands:
            continue
        if job_id in seen_ids:
            suffix = 2
            base_id = job_id
            while f"{base_id}:{suffix}" in seen_ids:
                suffix += 1
            record = dict(record)
            record["id"] = f"{base_id}:{suffix}"
            job_id = str(record["id"])
        seen_commands.add(command)
        seen_ids.add(job_id)
        answer.append(record)
    return answer


def _lifecycle_records(root: Path, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = [dict(row) for row in _CAPTURED_ORIGINAL(root, profile)]
    if profile.get("disable_lifecycle_orchestration") is True:
        return records

    authoritative = bool(profile.get("jobs")) or isinstance(profile.get("job_catalog"), Mapping)
    # Always centralize declared dataset/setup work even when the model catalog
    # itself is authoritative.
    records.extend(_setup_records(root, profile))

    declared_phase_keys = (
        ("preferred_validation_entrypoints", "validation"),
        ("preferred_testing_entrypoints", "testing"),
        ("preferred_metrics_entrypoints", "metrics"),
        ("validation_jobs", "validation"),
        ("testing_jobs", "testing"),
        ("metrics_jobs", "metrics"),
    )
    for key, phase in declared_phase_keys:
        records.extend(_profile_declared_jobs(root, profile, key, phase))

    auto_discover = bool(profile.get("auto_lifecycle_discovery", not authoritative))
    if auto_discover:
        represented_sources = {_source_key(root, row) for row in records}
        ignored = [str(x).replace("\\", "/") for x in (profile.get("ignore_entrypoints", []) or [])]
        registry_covers = [str(x).replace("\\", "/") for x in (profile.get("dynamic_registry_covers", []) or [])]
        for path in _iter_scripts(root):
            rel = path.relative_to(root).as_posix()
            if rel in represented_sources:
                continue
            if any(rel == pattern or fnmatch.fnmatch(rel, pattern) for pattern in ignored):
                continue
            # A registry-covered source is reached through its authoritative
            # wrapper/catalog; do not accidentally schedule helpers as jobs.
            if any(rel == pattern or fnmatch.fnmatch(rel, pattern) for pattern in registry_covers):
                continue
            text = _read(path)
            if not _is_executable(path, text):
                continue
            phase = _phase_for(path, text)
            if phase is None:
                continue
            records.append(_discovered_record(root, path, phase, text))

    records = _dedupe(records)
    _add_dependencies(records)
    profile["lifecycle_orchestration"] = {
        "schema": LIFECYCLE_SCHEMA,
        "enabled": True,
        "authoritative_model_catalog": authoritative,
        "auto_discovery": auto_discover,
        "job_count": len(records),
        "phase_counts": {
            phase: sum(1 for row in records if _normalized_phase(row) == phase)
            for phase in ("setup", "preprocess", "training", "validation", "testing", "metrics")
        },
    }
    return records


def _run_setup(root: Path, profile: Dict[str, Any]) -> None:
    if profile.get("_training_control_lifecycle_setup_scheduled"):
        return
    _ORIGINAL_RUN_SETUP(root, profile)


def install() -> None:
    global _CAPTURED_ORIGINAL, _ORIGINAL_RUN_SETUP
    if getattr(current._ORIGINAL_JOB_RECORDS, "_training_control_lifecycle", False):
        return
    _CAPTURED_ORIGINAL = current._ORIGINAL_JOB_RECORDS
    _ORIGINAL_RUN_SETUP = base._run_setup
    _lifecycle_records._training_control_lifecycle = True  # type: ignore[attr-defined]
    current._ORIGINAL_JOB_RECORDS = _lifecycle_records
    base._run_setup = _run_setup


_CAPTURED_ORIGINAL = current._ORIGINAL_JOB_RECORDS
_ORIGINAL_RUN_SETUP = base._run_setup

__all__ = ["LIFECYCLE_SCHEMA", "install"]
