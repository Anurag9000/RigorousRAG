#!/usr/bin/env python3
"""Large-profile transport for the universal training controller.

The original environment-JSON interface remains supported.  Repositories with
very large explicit job catalogs can instead set ``TRAINING_CONTROL_PROFILE_FILE``
to a JSON file.  This avoids OS environment-size limits while changing no OPF
scheduler behavior.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import universal_training_controller as base


def _load_profile() -> Dict[str, Any]:
    path_text = os.environ.get("TRAINING_CONTROL_PROFILE_FILE", "").strip()
    inline = os.environ.get("TRAINING_CONTROL_PROFILE", "").strip()
    if path_text and inline:
        raise SystemExit(
            "Set only one of TRAINING_CONTROL_PROFILE_FILE or TRAINING_CONTROL_PROFILE"
        )
    if not path_text:
        return _ORIGINAL_LOAD_PROFILE()
    path = Path(path_text).expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Cannot load training-control profile file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("TRAINING_CONTROL_PROFILE_FILE must contain one JSON object")
    return value


def install() -> None:
    base._load_profile = _load_profile


_ORIGINAL_LOAD_PROFILE = base._load_profile
