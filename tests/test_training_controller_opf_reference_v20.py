from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller as base
import universal_training_controller_current as current
import universal_training_controller_opf_reference_v2 as reference


def _snapshot() -> dict:
    return {
        "base_repo": base.OPF_REFERENCE_REPOSITORY,
        "base_commit": base.OPF_REFERENCE_COMMIT,
        "base_root": base.OPF_RAW_ROOT,
        "base_blobs": dict(base.OPF_RUNTIME_BLOBS),
        "base_files": tuple(base.OPF_RUNTIME_FILES),
        "current_repo": getattr(current, "OPF_REFERENCE_REPOSITORY", None),
        "current_commit": current.OPF_REFERENCE_COMMIT,
        "current_blobs": dict(current.OPF_RUNTIME_BLOBS),
    }


def _restore(state: dict) -> None:
    base.OPF_REFERENCE_REPOSITORY = state["base_repo"]
    base.OPF_REFERENCE_COMMIT = state["base_commit"]
    base.OPF_RAW_ROOT = state["base_root"]
    base.OPF_RUNTIME_BLOBS = dict(state["base_blobs"])
    base.OPF_RUNTIME_FILES = tuple(state["base_files"])
    if state["current_repo"] is None:
        try:
            delattr(current, "OPF_REFERENCE_REPOSITORY")
        except AttributeError:
            pass
    else:
        current.OPF_REFERENCE_REPOSITORY = state["current_repo"]
    current.OPF_REFERENCE_COMMIT = state["current_commit"]
    current.OPF_RUNTIME_BLOBS = dict(state["current_blobs"])


def test_reference_install_synchronizes_base_and_current() -> None:
    state = _snapshot()
    try:
        base.OPF_REFERENCE_COMMIT = "base-old"
        base.OPF_RUNTIME_BLOBS = {"old-base": "deadbeef"}
        base.OPF_RUNTIME_FILES = tuple(base.OPF_RUNTIME_BLOBS)
        current.OPF_REFERENCE_COMMIT = "current-old"
        current.OPF_RUNTIME_BLOBS = {"old-current": "cafebabe"}

        reference.install()

        assert base.OPF_REFERENCE_REPOSITORY == reference.OPF_REFERENCE_REPOSITORY
        assert current.OPF_REFERENCE_REPOSITORY == reference.OPF_REFERENCE_REPOSITORY
        assert base.OPF_REFERENCE_COMMIT == reference.OPF_REFERENCE_COMMIT
        assert current.OPF_REFERENCE_COMMIT == reference.OPF_REFERENCE_COMMIT
        assert base.OPF_RUNTIME_BLOBS == reference.OPF_RUNTIME_BLOBS
        assert current.OPF_RUNTIME_BLOBS == reference.OPF_RUNTIME_BLOBS
        assert base.OPF_RUNTIME_FILES == tuple(reference.OPF_RUNTIME_BLOBS)
        assert base.OPF_RAW_ROOT.endswith("/" + reference.OPF_REFERENCE_COMMIT)
        assert reference.certificate()["synchronized"] is True
    finally:
        _restore(state)


def test_reference_install_survives_historical_current_reconfigure() -> None:
    state = _snapshot()
    try:
        reference.install()
        # The historical helper is called by older controller layers.  After the
        # synchronization install it must be idempotent rather than reverting to
        # an older scheduler pin.
        current._configure_reference()
        assert base.OPF_REFERENCE_COMMIT == reference.OPF_REFERENCE_COMMIT
        assert base.OPF_RUNTIME_BLOBS == reference.OPF_RUNTIME_BLOBS
        assert current.OPF_REFERENCE_COMMIT == reference.OPF_REFERENCE_COMMIT
        assert current.OPF_RUNTIME_BLOBS == reference.OPF_RUNTIME_BLOBS
    finally:
        _restore(state)


def test_reference_install_does_not_import_scheduler() -> None:
    state = _snapshot()
    try:
        sys.modules.pop("opf_reference_massive_suite_runner", None)
        reference.install()
        assert "opf_reference_massive_suite_runner" not in sys.modules
    finally:
        _restore(state)


def test_current_literal_runtime_blob_set_is_complete() -> None:
    assert reference.OPF_RUNTIME_BLOBS == {
        "utils/opf_massive_suite_runner.py": "b2ae3d04f9398df5c18c7c13f4c939bce46b930d",
        "utils/runtime_tuning.py": "f1cbfc44e009701a5540a046f2cd6b9f41f16b74",
        "utils/ml_backends.py": "c4cd5eaf783cd7ffbb92ab01ec743ef7cbd13d84",
        "utils/logging_utils.py": "482ba94643aa921f49eebb835f29cf4930bb2498",
        "utils/opf_shared_defaults.py": "bd76baa134b07567015d0151d5f14ba81dc667df",
        "DNN/VANILLA/Dyn_DNN4OPF/utils/run_defaults.py": "ff79e8c51f1fb21a11e4687989198ef0abb07491",
    }
