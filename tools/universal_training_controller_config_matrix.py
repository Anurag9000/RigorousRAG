#!/usr/bin/env python3
"""Conservative config-matrix expansion for exhaustive repository training control.

This layer does not invent model/dataset/task Cartesian products.  It only fills a
very specific omission class: a training command already consumes a config file
through an explicit config flag, sibling config files in the same directory have
strong training semantics, and that directory maps to exactly one trainer command
shape.  In that case each previously unrepresented sibling config is an explicitly
repo-authored experiment recipe and is cloned into a concrete central-runner job.

Ambiguous config directories are never guessed.  They are reported in profile
metadata for the workload-closure layer to fail closed if strict coverage asks for
all repository-authored experiment surfaces.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import universal_training_controller_current as current
import universal_training_controller_lifecycle as lifecycle

CONFIG_MATRIX_SCHEMA = 1
CONFIG_SUFFIXES = {".yaml", ".yml", ".json", ".toml"}
CONFIG_FLAGS = {
    "--config", "--config-file", "--config_file", "--config-path", "--config_path",
    "--cfg", "--recipe", "--recipe-config", "--recipe_config",
}
TRAINING_MARKERS = (
    "model", "architecture", "dataset", "task", "training", "trainer", "optimizer",
    "learning_rate", "learning-rate", "epochs", "max_epochs", "batch_size", "loss",
    "algorithm", "method", "scheduler", "checkpoint",
)
IDENTITY_KEYS = {
    "model": ("model", "model_name", "architecture", "arch"),
    "dataset": ("dataset", "dataset_name", "data_name"),
    "task": ("task", "task_name", "objective", "benchmark"),
}
_CAPTURED = None


def _repo_path(root: Path, token: str) -> Path | None:
    try:
        path = Path(str(token))
        candidate = path if path.is_absolute() else root / path
        candidate = candidate.resolve()
        candidate.relative_to(root.resolve())
        return candidate if candidate.is_file() else None
    except Exception:
        return None


def _config_slot(root: Path, command: Sequence[str]) -> tuple[int, Path, str] | None:
    for index, token in enumerate(command):
        text = str(token)
        if text in CONFIG_FLAGS and index + 1 < len(command):
            path = _repo_path(root, str(command[index + 1]))
            if path is not None and path.suffix.lower() in CONFIG_SUFFIXES:
                return index + 1, path, text
        for flag in CONFIG_FLAGS:
            prefix = flag + "="
            if text.startswith(prefix):
                value = text[len(prefix):]
                path = _repo_path(root, value)
                if path is not None and path.suffix.lower() in CONFIG_SUFFIXES:
                    return index, path, prefix
    return None


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _looks_training_config(path: Path) -> bool:
    text = _read(path).lower()
    if not text:
        return False
    hits = sum(1 for marker in TRAINING_MARKERS if marker in text)
    identity = any(marker in text for marker in ("model", "architecture", "dataset", "task", "algorithm", "method"))
    training = any(marker in text for marker in ("training", "trainer", "optimizer", "learning_rate", "epochs", "max_epochs", "batch_size", "loss"))
    return hits >= 3 and identity and training


def _trainer_signature(command: Sequence[str], slot: int, flag_mode: str) -> Tuple[str, ...]:
    values = [str(x) for x in command]
    if flag_mode.endswith("="):
        values[slot] = flag_mode + "<CONFIG>"
    else:
        values[slot] = "<CONFIG>"
    return tuple(values)


def _simple_scalars(path: Path) -> Dict[str, str]:
    text = _read(path)
    result: Dict[str, str] = {}
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            for target, aliases in IDENTITY_KEYS.items():
                for alias in aliases:
                    value = payload.get(alias)
                    if isinstance(value, (str, int, float)):
                        result[target] = str(value)
                        break
        return result
    for target, aliases in IDENTITY_KEYS.items():
        for alias in aliases:
            match = re.search(
                rf"(?mi)^\s*{re.escape(alias)}\s*[:=]\s*['\"]?([^#\n\r'\"]+)", text
            )
            if match:
                value = match.group(1).strip().strip(",")
                if value and len(value) <= 160:
                    result[target] = value
                    break
    return result


def _records(root: Path, profile: Dict[str, Any]):
    assert _CAPTURED is not None
    records = [dict(row) for row in _CAPTURED(root, profile)]
    if profile.get("disable_config_matrix_autofill") is True:
        return records

    represented: set[Path] = set()
    templates_by_parent: Dict[Path, list[tuple[Dict[str, Any], int, str, Tuple[str, ...]]]] = {}
    for record in records:
        if lifecycle._normalized_phase(record) != "training":
            continue
        command = [str(x) for x in (record.get("command") or [])]
        slot = _config_slot(root, command)
        if slot is None:
            continue
        index, config_path, flag_mode = slot
        represented.add(config_path.resolve())
        signature = _trainer_signature(command, index, flag_mode)
        templates_by_parent.setdefault(config_path.parent.resolve(), []).append(
            (record, index, flag_mode, signature)
        )

    added: list[str] = []
    ambiguous: Dict[str, list[str]] = {}
    seen_ids = {str(row.get("id")) for row in records}
    seen_commands = {tuple(str(x) for x in (row.get("command") or [])) for row in records}

    for parent, templates in sorted(templates_by_parent.items(), key=lambda item: str(item[0])):
        signatures = {entry[3] for entry in templates}
        candidates = sorted(
            path.resolve()
            for path in parent.iterdir()
            if path.is_file() and path.suffix.lower() in CONFIG_SUFFIXES and _looks_training_config(path)
        )
        missing = [path for path in candidates if path not in represented]
        if not missing:
            continue
        if len(signatures) != 1:
            ambiguous[parent.relative_to(root.resolve()).as_posix()] = [
                path.relative_to(root.resolve()).as_posix() for path in missing
            ]
            continue

        template, slot_index, flag_mode, _ = templates[0]
        for config_path in missing:
            rel = config_path.relative_to(root.resolve()).as_posix()
            clone = dict(template)
            command = [str(x) for x in (template.get("command") or [])]
            if flag_mode.endswith("="):
                command[slot_index] = flag_mode + str(config_path)
            else:
                command[slot_index] = str(config_path)
            key = tuple(command)
            if key in seen_commands:
                continue
            base_id = f"config:auto:{rel}"
            job_id = base_id
            suffix = 2
            while job_id in seen_ids:
                job_id = f"{base_id}:{suffix}"
                suffix += 1
            clone["id"] = job_id
            clone["command"] = command
            clone["recipe_config"] = rel
            clone["config_matrix_autofilled"] = True
            clone["config_matrix_schema"] = CONFIG_MATRIX_SCHEMA
            for key_name, value in _simple_scalars(config_path).items():
                clone.setdefault(key_name, value)
            records.append(clone)
            seen_ids.add(job_id)
            seen_commands.add(key)
            represented.add(config_path)
            added.append(job_id)

    profile["config_matrix_autofill"] = {
        "schema": CONFIG_MATRIX_SCHEMA,
        "enabled": True,
        "added_job_count": len(added),
        "added_job_ids": added,
        "ambiguous_config_directories": ambiguous,
    }
    return records


def install() -> None:
    global _CAPTURED
    if getattr(current._ORIGINAL_JOB_RECORDS, "_training_control_config_matrix", False):
        return
    _CAPTURED = current._ORIGINAL_JOB_RECORDS
    _records._training_control_config_matrix = True  # type: ignore[attr-defined]
    current._ORIGINAL_JOB_RECORDS = _records


__all__ = ["CONFIG_MATRIX_SCHEMA", "install"]
