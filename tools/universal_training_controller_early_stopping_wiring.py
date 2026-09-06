#!/usr/bin/env python3
"""Activate repository-native early stopping from the central runner when possible.

The controller never fabricates model-specific stopping semantics.  Instead it
inspects the concrete training entrypoint already selected by the repository. If
that entrypoint contains operational early-stopping evidence and explicitly
exposes CLI controls for it, absent controls are filled from conservative central
defaults. Config-driven early-stopping declarations are recorded as evidence.

Training code with no early-stopping implementation remains unmodified and is
left for the strict training-contract layer to reject when the repository requires
it. This makes central early stopping real where the trainer supports it without
pretending an arbitrary process kill is equivalent to a validation-aware stop.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import universal_training_controller_current as current
import universal_training_controller_lifecycle as lifecycle

EARLY_STOP_WIRING_SCHEMA = 1
_CAPTURED = None

EARLY_SOURCE = re.compile(
    r"EarlyStopping|early[_ -]?stopp|early_stopping_rounds|stopping_rounds|"
    r"bad_epochs|no_improv|patience", re.I,
)
BREAK_EVIDENCE = re.compile(r"\bbreak\b", re.I)
ARG_DECL = re.compile(
    r"add_argument\(\s*['\"](?P<flag>--[A-Za-z0-9_-]+)['\"](?P<body>[^)]*)\)",
    re.S,
)
CONFIG_SUFFIXES = {".yaml", ".yml", ".json", ".toml"}
CONFIG_FLAGS = {
    "--config", "--config-file", "--config_file", "--config-path", "--config_path",
    "--cfg", "--recipe", "--recipe-config", "--recipe_config",
}


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _entrypoint(root: Path, record: Mapping[str, Any]) -> Path | None:
    rel = lifecycle._record_source(root, record)
    if not rel:
        return None
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except Exception:
        return None
    return path if path.is_file() else None


def _declared_args(text: str) -> Dict[str, str]:
    return {match.group("flag"): match.group("body") for match in ARG_DECL.finditer(text)}


def _has_flag(command: Sequence[str], flag: str) -> bool:
    return any(str(token) == flag or str(token).startswith(flag + "=") for token in command)


def _config_path(root: Path, command: Sequence[str]) -> Path | None:
    for index, token in enumerate(command):
        text = str(token)
        if text in CONFIG_FLAGS and index + 1 < len(command):
            candidate = Path(str(command[index + 1]))
            candidate = candidate if candidate.is_absolute() else root / candidate
            if candidate.is_file() and candidate.suffix.lower() in CONFIG_SUFFIXES:
                return candidate.resolve()
        for flag in CONFIG_FLAGS:
            prefix = flag + "="
            if text.startswith(prefix):
                candidate = Path(text[len(prefix):])
                candidate = candidate if candidate.is_absolute() else root / candidate
                if candidate.is_file() and candidate.suffix.lower() in CONFIG_SUFFIXES:
                    return candidate.resolve()
    return None


def _config_evidence(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    text = _read(path)
    low = text.lower()
    evidence: Dict[str, Any] = {}
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            for key in ("early_stopping", "early_stop", "patience", "early_stopping_patience", "min_delta"):
                if key in payload:
                    evidence[key] = payload.get(key)
    if not evidence:
        for key in ("early_stopping", "early_stop", "patience", "early_stopping_patience", "min_delta"):
            match = re.search(rf"(?mi)^\s*{re.escape(key)}\s*[:=]\s*([^#\n\r]+)", text)
            if match:
                evidence[key] = match.group(1).strip().strip("'\" ,")
    if evidence:
        evidence["config_path"] = str(path)
    return evidence


def _default_int(profile: Mapping[str, Any], key: str, env: str, fallback: int) -> int:
    raw = os.environ.get(env)
    if raw is None:
        raw = (profile.get("early_stopping_defaults") or {}).get(key) if isinstance(profile.get("early_stopping_defaults"), Mapping) else None
    try:
        return int(raw) if raw is not None else int(fallback)
    except Exception:
        return int(fallback)


def _default_float(profile: Mapping[str, Any], key: str, env: str, fallback: float) -> float:
    raw = os.environ.get(env)
    if raw is None:
        raw = (profile.get("early_stopping_defaults") or {}).get(key) if isinstance(profile.get("early_stopping_defaults"), Mapping) else None
    try:
        return float(raw) if raw is not None else float(fallback)
    except Exception:
        return float(fallback)


def _wire_one(root: Path, profile: Mapping[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    if lifecycle._normalized_phase(record) != "training":
        return record
    path = _entrypoint(root, record)
    text = _read(path) if path is not None else ""
    source_has_early = bool(EARLY_SOURCE.search(text))
    config = _config_evidence(_config_path(root, record.get("command") or []))
    args = _declared_args(text)
    command = [str(x) for x in (record.get("command") or [])]
    injected: list[str] = []

    # Only activate controls when the trainer source itself contains early-stop
    # semantics. A generic unrelated --patience option is never guessed.
    if source_has_early:
        boolean_flags = (
            "--early-stopping", "--early_stopping", "--enable-early-stopping",
            "--enable_early_stopping",
        )
        for flag in boolean_flags:
            body = args.get(flag)
            if body is None or _has_flag(command, flag):
                continue
            if "store_true" in body or "BooleanOptionalAction" in body:
                command.append(flag)
                injected.append(flag)
                break

        patience = _default_int(profile, "patience", "TRAINING_CONTROL_EARLY_STOPPING_PATIENCE", 10)
        for flag in ("--early-stopping-patience", "--early_stopping_patience", "--patience"):
            if flag in args and not _has_flag(command, flag):
                command.extend([flag, str(patience)])
                injected.extend([flag, str(patience)])
                break

        min_delta = _default_float(profile, "min_delta", "TRAINING_CONTROL_EARLY_STOPPING_MIN_DELTA", 0.0)
        for flag in ("--early-stopping-min-delta", "--early_stopping_min_delta", "--min-delta", "--min_delta"):
            if flag in args and not _has_flag(command, flag):
                command.extend([flag, str(min_delta)])
                injected.extend([flag, str(min_delta)])
                break

    evidence: list[str] = []
    if source_has_early:
        evidence.append("entrypoint_source")
    if source_has_early and BREAK_EVIDENCE.search(text):
        evidence.append("loop_break")
    if config:
        evidence.append("training_config")
    if injected:
        evidence.append("central_cli_activation")

    result = dict(record)
    result["command"] = command
    if evidence:
        result["early_stopping"] = True
        result["early_stopping_contract"] = {
            "schema": EARLY_STOP_WIRING_SCHEMA,
            "evidence": evidence,
            "entrypoint": str(path.relative_to(root.resolve())) if path is not None else None,
            "config": config,
            "injected_cli": injected,
            "source_native": source_has_early,
            "controller_invented_semantics": False,
        }
    return result


def _records(root: Path, profile: Dict[str, Any]):
    assert _CAPTURED is not None
    records = [_wire_one(root, profile, dict(row)) for row in _CAPTURED(root, profile)]
    wired = [
        str(row.get("id")) for row in records
        if isinstance(row.get("early_stopping_contract"), Mapping)
    ]
    activated = [
        str(row.get("id")) for row in records
        if isinstance(row.get("early_stopping_contract"), Mapping)
        and bool((row.get("early_stopping_contract") or {}).get("injected_cli"))
    ]
    profile["early_stopping_wiring"] = {
        "schema": EARLY_STOP_WIRING_SCHEMA,
        "jobs_with_operational_evidence": wired,
        "jobs_with_central_cli_activation": activated,
    }
    return records


def install() -> None:
    global _CAPTURED
    if getattr(current._ORIGINAL_JOB_RECORDS, "_training_control_early_stop_wiring", False):
        return
    _CAPTURED = current._ORIGINAL_JOB_RECORDS
    _records._training_control_early_stop_wiring = True  # type: ignore[attr-defined]
    current._ORIGINAL_JOB_RECORDS = _records


__all__ = ["EARLY_STOP_WIRING_SCHEMA", "install"]
