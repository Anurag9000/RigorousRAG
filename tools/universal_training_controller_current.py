#!/usr/bin/env python3
"""Current cross-repository training controller.

This module deliberately does not fork the OPF_ADP scheduler. It imports the
existing universal adapter and repins its scheduler cache to an exact OPF_ADP
commit/blob set, then strengthens repository coverage auditing.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import universal_training_controller as base

OPF_REFERENCE_COMMIT = "a34c31259bd5d5f58081e3766918f9df63017455"
OPF_RUNTIME_BLOBS = {
    "utils/opf_massive_suite_runner.py": "b97d47499c83bc6ed3a5753f7f3009b624c94868",
    "utils/runtime_tuning.py": "f1cbfc44e009701a5540a046f2cd6b9f41f16b74",
    "utils/ml_backends.py": "2fe2b24e530cab3d747c983c4457f4080703512f",
    "utils/logging_utils.py": "482ba94643aa921f49eebb835f29cf4930bb2498",
    "utils/opf_shared_defaults.py": "76ad434ecef1f708c835210d4bc86e0717999d99",
    "DNN/VANILLA/Dyn_DNN4OPF/utils/run_defaults.py": "dacb9a2c44d611c045fbb7512ba5327343f79a85",
}
AUDIT_SCHEMA = 4
SOURCE_SUFFIXES = {
    ".py", ".sh", ".ps1", ".bat", ".cmd", ".ipynb", ".r", ".jl"
}
SKIP = {
    ".git", ".training_control", "__pycache__", ".venv", "venv", "env",
    "node_modules", "results", "result", "outputs", "output",
    "checkpoints", "artifacts", "dist", "build",
}
TRAIN_PATTERNS = (
    re.compile(r"\btorch\.optim\b|\bloss\.backward\s*\(|\.backward\s*\(", re.I),
    re.compile(r"\b(?:model|estimator|pipeline|clf|regressor)\.fit\s*\(", re.I),
    re.compile(r"\bTrainer\s*\(|\btraining_step\s*\(|\btrain_one_epoch\b", re.I),
    re.compile(r"\b(?:PPO|DQN|A2C|SAC|TD3|REINFORCE)\s*\(|\.learn\s*\(", re.I),
    re.compile(r"\boptuna\b|\bGridSearchCV\b|\bRandomizedSearchCV\b", re.I),
)
MODEL_PATTERNS = (
    re.compile(r"class\s+\w+\s*\([^)]*(?:nn\.Module|torch\.nn\.Module|LightningModule)", re.I),
    re.compile(r"\b(?:Sequential|Functional|Model)\s*\(", re.I),
    re.compile(r"\b(?:RandomForest|XGB|LGBM|CatBoost|SVC|SVR|LogisticRegression|LinearRegression)\w*\s*\(", re.I),
)
CHECKPOINT_WRITE = re.compile(
    r"torch\.save\s*\(|save_checkpoint\s*\(|ModelCheckpoint|save_pretrained\s*\(|"
    r"checkpoint.*write|joblib\.dump\s*\(|pickle\.dump\s*\(",
    re.I,
)
CHECKPOINT_READ = re.compile(
    r"torch\.load\s*\(|load_state_dict\s*\(|load_checkpoint\s*\(|"
    r"resume_from_checkpoint|from_pretrained\s*\(|joblib\.load\s*\(|pickle\.load\s*\(",
    re.I,
)
RESUME_TOKEN = re.compile(r"\bresume\b|start_epoch|initial_epoch|checkpoint_last|last_checkpoint", re.I)
EARLY_STOP_TOKEN = re.compile(r"early[_ -]?stopp|patience|stopping_rounds|EarlyStopping", re.I)


def _configure_reference() -> None:
    base.OPF_REFERENCE_COMMIT = OPF_REFERENCE_COMMIT
    base.OPF_RAW_ROOT = (
        f"https://raw.githubusercontent.com/{base.OPF_REFERENCE_REPOSITORY}/"
        f"{OPF_REFERENCE_COMMIT}"
    )
    base.OPF_RUNTIME_BLOBS = dict(OPF_RUNTIME_BLOBS)
    base.OPF_RUNTIME_FILES = tuple(OPF_RUNTIME_BLOBS)


def _iter_sources(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        rel = path.relative_to(root)
        if any(part.lower() in SKIP for part in rel.parts):
            continue
        yield path


def _read_text(path: Path) -> str:
    try:
        if path.suffix.lower() == ".ipynb":
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            cells = payload.get("cells", []) if isinstance(payload, dict) else []
            return "\n".join(
                "".join(cell.get("source", []))
                for cell in cells
                if isinstance(cell, dict) and cell.get("cell_type") == "code"
            )
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _is_executable_script(path: Path, text: str) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "__name__" in text and "__main__" in text
    if suffix == ".ipynb":
        return False
    return suffix in {".sh", ".ps1", ".bat", ".cmd", ".r", ".jl"}


def _matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _training_inventory(root: Path) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    executable_candidates: List[str] = []
    model_surfaces: List[str] = []
    training_logic_surfaces: List[str] = []
    for path in _iter_sources(root):
        rel = path.relative_to(root).as_posix()
        if rel in {"run_all_training.py", "tools/universal_training_controller.py",
                   "tools/universal_training_controller_current.py"}:
            continue
        text = _read_text(path)
        train_hit = _matches_any(text, TRAIN_PATTERNS)
        model_hit = _matches_any(text, MODEL_PATTERNS)
        executable = _is_executable_script(path, text)
        if train_hit:
            training_logic_surfaces.append(rel)
            if executable:
                executable_candidates.append(rel)
        if model_hit:
            model_surfaces.append(rel)
        if train_hit or model_hit:
            files.append({
                "path": rel,
                "training_logic": bool(train_hit),
                "model_surface": bool(model_hit),
                "executable": bool(executable),
                "checkpoint_write": bool(CHECKPOINT_WRITE.search(text)),
                "checkpoint_read": bool(CHECKPOINT_READ.search(text)),
                "resume_token": bool(RESUME_TOKEN.search(text)),
                "early_stopping": bool(EARLY_STOP_TOKEN.search(text)),
            })
    return {
        "source_files_scanned": sum(1 for _ in _iter_sources(root)),
        "training_files": sorted(files, key=lambda x: x["path"]),
        "training_logic_surfaces": sorted(set(training_logic_surfaces)),
        "model_surfaces": sorted(set(model_surfaces)),
        "executable_training_candidates": sorted(set(executable_candidates)),
    }


def _covered_by_patterns(path: str, patterns: Iterable[str]) -> bool:
    normalized = path.replace("\\", "/")
    for pattern in patterns:
        p = str(pattern).replace("\\", "/")
        if normalized == p or fnmatch.fnmatch(normalized, p):
            return True
    return False


def _command_repo_paths(root: Path, jobs: Sequence[Dict[str, Any]]) -> Set[str]:
    covered: Set[str] = set()
    for job in jobs:
        for part in job.get("command", []):
            try:
                candidate = Path(str(part))
                if candidate.is_absolute():
                    covered.add(candidate.resolve().relative_to(root).as_posix())
                elif (root / candidate).is_file():
                    covered.add(candidate.as_posix())
            except Exception:
                continue
    return covered


def _job_resume_evidence(root: Path, job: Dict[str, Any]) -> Dict[str, Any]:
    paths = _command_repo_paths(root, [job])
    source_evidence: List[Dict[str, Any]] = []
    native = False
    early = False
    for rel in sorted(paths):
        path = root / rel
        text = _read_text(path)
        if not text:
            continue
        write = bool(CHECKPOINT_WRITE.search(text))
        read = bool(CHECKPOINT_READ.search(text))
        resume = bool(RESUME_TOKEN.search(text))
        early_stop = bool(EARLY_STOP_TOKEN.search(text))
        native = native or (write and read and resume)
        early = early or early_stop
        source_evidence.append({
            "path": rel,
            "checkpoint_write": write,
            "checkpoint_read": read,
            "resume_token": resume,
            "early_stopping": early_stop,
        })
    declared = str(job.get("resume_strategy") or "").strip().lower()
    if declared in {"native_checkpoint", "framework_checkpoint"}:
        native = True
    return {
        "job_id": str(job.get("id")),
        "resume_strategy": declared or ("native_checkpoint_detected" if native else "unproven"),
        "native_resume_proven": bool(native),
        "early_stopping_present": bool(early or job.get("early_stopping") is True),
        "source_evidence": source_evidence,
    }


def _enhanced_coverage_report(
    root: Path, profile: Dict[str, Any], jobs: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    original = _ORIGINAL_COVERAGE_REPORT(root, profile, jobs)
    inventory = _training_inventory(root)
    covered = set(_command_repo_paths(root, jobs))
    cover_patterns = list(profile.get("dynamic_registry_covers", []) or [])
    cover_patterns += list(profile.get("ignore_entrypoints", []) or [])
    setup_paths = [
        str(item).replace("\\", "/")
        for item in (profile.get("preferred_dataset_entrypoints", []) or [])
        if (root / str(item)).is_file()
    ]
    covered.update(setup_paths)

    executable_uncovered = [
        path for path in inventory["executable_training_candidates"]
        if path not in covered and not _covered_by_patterns(path, cover_patterns)
    ]
    model_unaccounted = [
        path for path in inventory["model_surfaces"]
        if path not in covered and not _covered_by_patterns(path, cover_patterns)
    ]
    logic_unaccounted = [
        path for path in inventory["training_logic_surfaces"]
        if path not in covered and not _covered_by_patterns(path, cover_patterns)
    ]

    resume = [_job_resume_evidence(root, job) for job in jobs]
    unresolved_resume = [r["job_id"] for r in resume if not r["native_resume_proven"]]
    missing_early = [r["job_id"] for r in resume if not r["early_stopping_present"]]

    require_resume = bool(profile.get("require_native_resume", False)) or (
        os.environ.get("TRAINING_CONTROL_REQUIRE_NATIVE_RESUME") == "1"
    )
    require_early = bool(profile.get("require_early_stopping", False)) or (
        os.environ.get("TRAINING_CONTROL_REQUIRE_EARLY_STOPPING") == "1"
    )
    require_model_accounting = bool(profile.get("require_model_surface_accounting", True))
    strict_missing = list(executable_uncovered)
    if require_model_accounting:
        strict_missing.extend(model_unaccounted)
        strict_missing.extend(logic_unaccounted)

    original.update({
        "schema": AUDIT_SCHEMA,
        "opf_reference_commit": OPF_REFERENCE_COMMIT,
        "opf_runtime_blobs": OPF_RUNTIME_BLOBS,
        "inventory": inventory,
        "uncovered_executable_training_candidates": sorted(set(executable_uncovered)),
        "unaccounted_model_surfaces": sorted(set(model_unaccounted)),
        "unaccounted_training_logic_surfaces": sorted(set(logic_unaccounted)),
        "resume_audit": resume,
        "unresolved_native_resume_jobs": unresolved_resume,
        "jobs_without_early_stopping_evidence": missing_early,
        "require_native_resume": require_resume,
        "require_early_stopping": require_early,
        "require_model_surface_accounting": require_model_accounting,
    })
    original["coverage_ok"] = (
        bool(original.get("coverage_ok", True))
        and not strict_missing
        and (not require_resume or not unresolved_resume)
        and (not require_early or not missing_early)
    )
    return original


def _install_enhancements() -> None:
    _configure_reference()
    base._coverage_report = _enhanced_coverage_report


_ORIGINAL_COVERAGE_REPORT = base._coverage_report


def main(argv: Sequence[str] | None = None) -> int:
    _install_enhancements()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
