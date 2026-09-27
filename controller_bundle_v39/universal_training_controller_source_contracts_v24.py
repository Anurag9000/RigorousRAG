#!/usr/bin/env python3
"""Fail-closed source contracts for the account-wide training controller v24.

This layer does not schedule resources and does not change OPF_ADP.  It closes
three static-accounting loopholes that can otherwise make an exhaustive catalog
look healthier than its executable source:

1. every repository-local script/config explicitly named by a compiled job must
   exist at controller construction time;
2. interruption-exact training resume must be proven from reachable source, not
   merely from ``resume_strategy``/``checkpoint_contract`` metadata;
3. semantic early stopping must be supported by reachable trainer source, not
   merely by an ``early_stopping=True`` catalog flag.

The proof is deliberately conservative.  Framework-native EarlyStopping
callbacks qualify.  Hand-written loops qualify only when source contains an
actual patience/improvement state, a validation/evaluation monitor and a stop
transition.  Repositories may declare an early-stopping exemption only by using
the existing explicit ``early_stopping_applicable=False`` + reason contract.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import universal_training_controller_current as current

SOURCE_CONTRACT_SCHEMA = 1
CONFIG_SUFFIXES = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
TRAINING_PHASES = {
    "training", "train", "pretrain", "pretraining", "finetune", "fine-tune",
    "fine_tune", "adapt", "fit", "rl", "reinforcement-learning",
    "reinforcement_learning", "policy-training", "policy_training",
}

_FRAMEWORK_EARLY = re.compile(
    r"\bEarlyStopping\s*\(|\bEarlyStoppingCallback\s*\(|"
    r"callbacks\s*=.*EarlyStopping|early_stopping_rounds\s*=|"
    r"load_best_model_at_end\s*=\s*True",
    re.I | re.S,
)
_PATIENCE = re.compile(
    r"\bpatience\b|\bmin_delta\b|\bstale_epochs\b|\bbad_epochs\b|"
    r"\bno_improv(?:ement|e)?\w*\b|\bearly_stop(?:ping)?_counter\b",
    re.I,
)
_MONITOR = re.compile(
    r"\bval(?:idation)?[_ .-]?(?:loss|metric|score|reward|accuracy|f1|auc|mrr|ndcg|wer|cer)?\b|"
    r"\beval(?:uation)?[_ .-]?(?:loss|metric|score|reward)?\b|"
    r"\bdev[_ .-]?(?:loss|metric|score)?\b",
    re.I,
)
_BEST = re.compile(r"\bbest[_ .-]?(?:loss|metric|score|reward|state|checkpoint|model)\b", re.I)
_STOP = re.compile(
    r"\bshould_stop\b|\bearly_stop(?:ping)?\b|\bbreak\b|"
    r"raise\s+StopIteration|return\s+.*best",
    re.I,
)
_IMPROVEMENT = re.compile(
    r"(?:<|>)\s*best|best\s*=|min_delta|improv(?:e|ed|ement)|stale_epochs|bad_epochs",
    re.I,
)


def _is_training(job: Mapping[str, Any]) -> bool:
    explicit = job.get("is_training_job")
    if explicit is not None:
        return explicit is True
    phase = str(job.get("phase") or "").strip().lower()
    family = str(job.get("family") or "").strip().lower()
    return phase in TRAINING_PHASES or family in TRAINING_PHASES


def _combined_text(root: Path, paths: Iterable[str]) -> str:
    parts = []
    for rel in sorted(set(str(x) for x in paths)):
        path = root / rel
        if path.is_file() and path.suffix.lower() in current.SOURCE_SUFFIXES:
            text = current._read_text(path)
            if text:
                parts.append(text)
    return "\n".join(parts)


def _semantic_early_stop(text: str) -> Dict[str, bool]:
    framework = bool(_FRAMEWORK_EARLY.search(text))
    patience = bool(_PATIENCE.search(text))
    monitor = bool(_MONITOR.search(text))
    best = bool(_BEST.search(text))
    stop = bool(_STOP.search(text))
    improvement = bool(_IMPROVEMENT.search(text))
    handwritten = bool(patience and monitor and best and stop and improvement)
    return {
        "framework_native": framework,
        "patience_or_counter": patience,
        "validation_monitor": monitor,
        "best_state": best,
        "stop_transition": stop,
        "improvement_rule": improvement,
        "proven": bool(framework or handwritten),
    }


def _pathish(token: str) -> bool:
    value = token.strip().strip("'\"")
    if not value or value.startswith(("http://", "https://")):
        return False
    suffix = Path(value).suffix.lower()
    if suffix in current.EXEC_SCRIPT_SUFFIXES | CONFIG_SUFFIXES:
        return "/" in value or "\\" in value or suffix in current.EXEC_SCRIPT_SUFFIXES or value.startswith(("config", "."))
    return False


def _candidate_path(root: Path, token: str) -> tuple[str, bool] | None:
    value = token.strip().strip("'\"")
    if not _pathish(value):
        return None
    raw = Path(value)
    if raw.is_absolute():
        try:
            rel = raw.resolve().relative_to(root.resolve()).as_posix()
        except Exception:
            return None
        return rel, raw.is_file()
    candidate = root / raw
    try:
        rel = candidate.resolve(strict=False).relative_to(root.resolve()).as_posix()
    except Exception:
        return value.replace("\\", "/"), False
    return rel, candidate.is_file()


def _command_contract(root: Path, job: Mapping[str, Any]) -> Dict[str, Any]:
    command = [str(x) for x in (job.get("command") or [])]
    missing: list[str] = []
    present: list[str] = []
    for token in command:
        candidate = _candidate_path(root, token)
        if candidate is None:
            continue
        rel, exists = candidate
        (present if exists else missing).append(rel)
    for key in ("entrypoint_source", "recipe_config", "config", "catalog_source", "canonical_path"):
        value = job.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = _candidate_path(root, value)
        if candidate is None:
            # Explicit source/config metadata is path-bearing even without a slash.
            raw = root / value
            rel = value.replace("\\", "/")
            exists = raw.is_file()
        else:
            rel, exists = candidate
        (present if exists else missing).append(rel)
    return {
        "job_id": str(job.get("id")),
        "present_local_targets": sorted(set(present)),
        "missing_local_targets": sorted(set(missing)),
        "valid": not missing,
    }


def install() -> None:
    original_report = current._enhanced_coverage_report

    def coverage_report(root: Path, profile: Dict[str, Any], jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        report = original_report(root, profile, jobs)
        reach = current._reachability(root, jobs)
        command_rows = [_command_contract(root, job) for job in jobs]
        dangling = [row for row in command_rows if not row["valid"]]

        source_rows = []
        early_missing: list[str] = []
        exact_missing: list[str] = []
        for job in jobs:
            if not _is_training(job):
                continue
            job_id = str(job.get("id"))
            paths = (reach.get("per_job", {}).get(job_id, {}) or {}).get("reachable", [])
            checkpoint = current._checkpoint_contract_for_paths(root, paths)
            early = _semantic_early_stop(_combined_text(root, paths))
            applicable = job.get("early_stopping_applicable")
            reason = str(job.get("early_stopping_exception_reason") or "").strip()
            exempt = applicable is False and bool(reason)
            exact = bool(checkpoint.get("exact_resume_detected"))
            if not exact:
                exact_missing.append(job_id)
            if not early["proven"] and not exempt:
                early_missing.append(job_id)
            source_rows.append({
                "job_id": job_id,
                "reachable_source_count": len(paths),
                "source_exact_resume_proven": exact,
                "source_semantic_early_stopping": early,
                "early_stopping_exempt": exempt,
                "early_stopping_exemption_reason": reason if exempt else "",
            })

        require_targets = bool(profile.get("require_existing_job_targets", True))
        require_source_exact = bool(profile.get("require_source_proven_training_exact_resume", True))
        require_source_early = bool(profile.get("require_source_proven_training_early_stopping", True))
        if os.environ.get("TRAINING_CONTROL_ALLOW_METADATA_ONLY_CONTRACTS", "").strip().lower() in {"1", "true", "yes", "on"}:
            require_source_exact = False
            require_source_early = False

        ok = (
            (not require_targets or not dangling)
            and (not require_source_exact or not exact_missing)
            and (not require_source_early or not early_missing)
        )
        controls = dict(report.get("strict_controls") or {})
        controls.update({
            "require_existing_job_targets": require_targets,
            "require_source_proven_training_exact_resume": require_source_exact,
            "require_source_proven_training_early_stopping": require_source_early,
        })
        report.update({
            "source_contract_schema": SOURCE_CONTRACT_SCHEMA,
            "job_target_contracts": command_rows,
            "jobs_with_missing_local_targets": dangling,
            "training_source_contracts": source_rows,
            "training_jobs_without_source_exact_resume": sorted(set(exact_missing)),
            "training_jobs_without_source_semantic_early_stopping": sorted(set(early_missing)),
            "strict_source_contracts_pass": ok,
            "strict_controls": controls,
        })
        report["coverage_ok"] = bool(report.get("coverage_ok", False)) and ok
        return report

    current._enhanced_coverage_report = coverage_report


__all__ = ["SOURCE_CONTRACT_SCHEMA", "install"]
