#!/usr/bin/env python3
"""Synchronize the universal controller to one current literal OPF_ADP reference.

This module changes no scheduling behavior.  It only makes the OPF reference used
by the base adapter, the enhanced audit layer, and the mechanism certificate
identical and explicit.  The scheduler source and helper modules are still loaded
byte-for-byte from the pinned OPF_ADP commit.
"""
from __future__ import annotations

from typing import Dict

import universal_training_controller as base
import universal_training_controller_current as current

OPF_REFERENCE_REPOSITORY = "Anurag9000/OPF_ADP"
OPF_REFERENCE_COMMIT = "2dfe664af88b95981da2b84b60f228a37156749f"
OPF_RUNTIME_BLOBS: Dict[str, str] = {
    "utils/opf_massive_suite_runner.py": "b2ae3d04f9398df5c18c7c13f4c939bce46b930d",
    "utils/runtime_tuning.py": "f1cbfc44e009701a5540a046f2cd6b9f41f16b74",
    "utils/ml_backends.py": "c4cd5eaf783cd7ffbb92ab01ec743ef7cbd13d84",
    "utils/logging_utils.py": "482ba94643aa921f49eebb835f29cf4930bb2498",
    "utils/opf_shared_defaults.py": "bd76baa134b07567015d0151d5f14ba81dc667df",
    "DNN/VANILLA/Dyn_DNN4OPF/utils/run_defaults.py": "ff79e8c51f1fb21a11e4687989198ef0abb07491",
}


def _set_base_reference() -> None:
    base.OPF_REFERENCE_REPOSITORY = OPF_REFERENCE_REPOSITORY
    base.OPF_REFERENCE_COMMIT = OPF_REFERENCE_COMMIT
    base.OPF_RAW_ROOT = (
        f"https://raw.githubusercontent.com/{OPF_REFERENCE_REPOSITORY}/"
        f"{OPF_REFERENCE_COMMIT}"
    )
    base.OPF_RUNTIME_BLOBS = dict(OPF_RUNTIME_BLOBS)
    base.OPF_RUNTIME_FILES = tuple(OPF_RUNTIME_BLOBS)


def _set_current_reference() -> None:
    # ``current`` historically owned a second OPF pin and its own helper that
    # re-applies that pin to ``base``.  Synchronize both so a later call to
    # current._configure_reference() cannot revert the selected scheduler.
    current.OPF_REFERENCE_REPOSITORY = OPF_REFERENCE_REPOSITORY
    current.OPF_REFERENCE_COMMIT = OPF_REFERENCE_COMMIT
    current.OPF_RUNTIME_BLOBS = dict(OPF_RUNTIME_BLOBS)


def install() -> None:
    _set_current_reference()
    _set_base_reference()
    # Exercise the historical repin path deliberately; after synchronization it
    # must be an idempotent reapplication of the exact same reference.
    current._configure_reference()
    assert base.OPF_REFERENCE_REPOSITORY == OPF_REFERENCE_REPOSITORY
    assert base.OPF_REFERENCE_COMMIT == OPF_REFERENCE_COMMIT
    assert dict(base.OPF_RUNTIME_BLOBS) == OPF_RUNTIME_BLOBS
    assert tuple(base.OPF_RUNTIME_FILES) == tuple(OPF_RUNTIME_BLOBS)
    assert current.OPF_REFERENCE_COMMIT == OPF_REFERENCE_COMMIT
    assert dict(current.OPF_RUNTIME_BLOBS) == OPF_RUNTIME_BLOBS


def certificate() -> dict:
    return {
        "repository": OPF_REFERENCE_REPOSITORY,
        "commit": OPF_REFERENCE_COMMIT,
        "runtime_blobs": dict(OPF_RUNTIME_BLOBS),
        "base_commit": base.OPF_REFERENCE_COMMIT,
        "base_runtime_blobs": dict(base.OPF_RUNTIME_BLOBS),
        "current_commit": current.OPF_REFERENCE_COMMIT,
        "current_runtime_blobs": dict(current.OPF_RUNTIME_BLOBS),
        "synchronized": (
            base.OPF_REFERENCE_COMMIT == current.OPF_REFERENCE_COMMIT == OPF_REFERENCE_COMMIT
            and dict(base.OPF_RUNTIME_BLOBS) == dict(current.OPF_RUNTIME_BLOBS) == OPF_RUNTIME_BLOBS
        ),
    }
