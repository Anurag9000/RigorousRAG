#!/usr/bin/env python3
"""Synchronize the universal controller to the current literal OPF_ADP reference.

This v44 overlay pins the audited shared CUDA admission and backend-isolation source. It synchronizes the OPF reference used
by the base adapter, the enhanced audit layer, and the mechanism certificate
identical and explicit. The scheduler source and helper modules are still loaded
byte-for-byte from the pinned OPF_ADP commit.

The CUDA-first shared scheduler preserves the pressure/admission algorithms while
adding strict child device isolation and import-time accelerator-policy ordering. Both the scheduler and backend tests are
pinned independently, and every downloaded source is verified by Git blob SHA.
"""
from __future__ import annotations

from typing import Dict

import universal_training_controller as base
import universal_training_controller_current as current

OPF_REFERENCE_REPOSITORY = "Anurag9000/OPF_ADP"
OPF_REFERENCE_COMMIT = "3d27019c17b2e8a48ef03068535d5ed5fdadca03"
OPF_RUNTIME_BLOBS: Dict[str, str] = {
    "utils/opf_massive_suite_runner.py": "9269e5c28a93e979c81b439547b4c9fe7d5a3c52",
    "utils/runtime_tuning.py": "f1cbfc44e009701a5540a046f2cd6b9f41f16b74",
    "utils/ml_backends.py": "ef23f05e32e5e4927c9aaaef7c7286a4bda92453",
    "utils/logging_utils.py": "482ba94643aa921f49eebb835f29cf4930bb2498",
    "utils/opf_shared_defaults.py": "85f1a948dca2f67a845ad129a6223484b47a0ad7",
    "DNN/VANILLA/Dyn_DNN4OPF/utils/run_defaults.py": "58b6c1633393b193d4d82daa952ad5e6857206fb",
}
OPF_OPERATIONAL_CONTRACT_BLOBS: Dict[str, str] = {
    "tests/test_massive_scheduler_operational_contract.py": "06b63ce1f6cdb446276fc0fdb934a8dfec323c74",
    "tests/test_ml_backends_gpu_first.py": "6ed2f7c4c51027e7f66ad59d9b5d8ded86074ecc",
    "tests/test_ml_backend_admission_isolation.py": "7f78297b066fbb0870f44de65dcaa80eca9f2564",
    "tests/test_opf_shared_defaults_cuda_admission.py": "0aa1aff969605643c455f70ed3383f0ddb6e6a14",
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
    # re-applies that pin to ``base``. Synchronize both so a later call to
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
        "operational_contract_blobs": dict(OPF_OPERATIONAL_CONTRACT_BLOBS),
        "base_commit": base.OPF_REFERENCE_COMMIT,
        "base_runtime_blobs": dict(base.OPF_RUNTIME_BLOBS),
        "current_commit": current.OPF_REFERENCE_COMMIT,
        "current_runtime_blobs": dict(current.OPF_RUNTIME_BLOBS),
        "synchronized": (
            base.OPF_REFERENCE_COMMIT == current.OPF_REFERENCE_COMMIT == OPF_REFERENCE_COMMIT
            and dict(base.OPF_RUNTIME_BLOBS) == dict(current.OPF_RUNTIME_BLOBS) == OPF_RUNTIME_BLOBS
        ),
    }
