#!/usr/bin/env python3
"""Safely satisfy repeated required console-script options from profile defaults."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import universal_training_controller_console as console


def _defaults(profile: Mapping[str, Any], item: Mapping[str, Any]) -> list[str] | None:
    if item.get("required_positionals"):
        return None
    mapping = profile.get("console_argument_defaults", {})
    if not isinstance(mapping, Mapping):
        return None
    args: list[str] = []
    for option in item.get("required_options", []) or []:
        if option not in mapping:
            return None
        value = mapping[option]
        if isinstance(value, bool):
            if value:
                args.append(str(option))
            continue
        if value is None:
            return None
        args.extend([str(option), str(value)])
    return args


def _inventory(root: Path, profile: Mapping[str, Any]) -> Dict[str, Any]:
    report = _ORIGINAL_INVENTORY(root, profile)
    entries = report.get("entries", [])
    for item in entries:
        if not isinstance(item, dict):
            continue
        default_args = None
        if item.get("training_surface") and not item.get("ignored") and item.get("explicit_args") is None:
            default_args = _defaults(profile, item)
        item["default_args"] = default_args
        if default_args is not None:
            item["configured"] = True
    training = [x for x in entries if isinstance(x, dict) and x.get("training_surface")]
    report["training_entries"] = training
    report["unconfigured_training_entrypoints"] = [
        x["name"] for x in training if not x.get("configured") and not x.get("ignored")
    ]
    return report


def _jobs(root: Path, profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    augmented = dict(profile)
    explicit = dict(profile.get("console_script_args", {}) or {}) if isinstance(profile.get("console_script_args", {}), Mapping) else {}
    for item in _inventory(root, profile).get("training_entries", []):
        if item.get("ignored") or item.get("explicit_args") is not None:
            continue
        defaults = item.get("default_args")
        if defaults is not None:
            explicit[str(item["name"])] = list(defaults)
    augmented["console_script_args"] = explicit
    return _ORIGINAL_JOBS(root, augmented)


def install() -> None:
    console._inventory = _inventory
    console._jobs = _jobs


_ORIGINAL_INVENTORY = console._inventory
_ORIGINAL_JOBS = console._jobs
