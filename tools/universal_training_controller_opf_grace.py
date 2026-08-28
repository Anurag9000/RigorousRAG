#!/usr/bin/env python3
"""Configurable cooperative-termination grace for the pinned OPF scheduler.

The OPF resource/admission algorithm remains unchanged. This module wraps only
the process-tree termination helper after the literal pinned scheduler is
imported, allowing exact-checkpoint trainers enough time to honor SIGTERM before
OPF escalates to a hard kill.
"""
from __future__ import annotations

import os
from typing import Any

import universal_training_controller as base

DEFAULT_GRACE_SECONDS = 30.0
MAX_GRACE_SECONDS = 300.0


def _configured_grace() -> float:
    raw = os.environ.get("TRAINING_CONTROL_TERMINATION_GRACE_SEC", str(DEFAULT_GRACE_SECONDS))
    try:
        value = float(raw)
    except Exception:
        value = DEFAULT_GRACE_SECONDS
    return min(MAX_GRACE_SECONDS, max(0.1, value))


def _import_opf_scheduler(cache: Any):
    opf = _ORIGINAL_IMPORT(cache)
    original = getattr(opf, "_terminate_process_tree", None)
    if original is None or getattr(original, "_training_control_grace_wrapped", False):
        return opf

    def terminate_process_tree(proc: Any, timeout_sec: float = 10.0) -> None:
        # Never shorten an explicit OPF timeout. The repository/controller may
        # only extend the cooperative checkpoint opportunity up to a bounded cap.
        return original(proc, timeout_sec=max(float(timeout_sec), _configured_grace()))

    terminate_process_tree._training_control_grace_wrapped = True  # type: ignore[attr-defined]
    terminate_process_tree._training_control_original = original  # type: ignore[attr-defined]
    opf._terminate_process_tree = terminate_process_tree
    opf.TRAINING_CONTROL_TERMINATION_GRACE_SEC = _configured_grace()
    return opf


def install() -> None:
    base._import_opf_scheduler = _import_opf_scheduler


_ORIGINAL_IMPORT = base._import_opf_scheduler
