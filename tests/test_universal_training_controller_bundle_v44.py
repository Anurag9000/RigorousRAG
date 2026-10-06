"""Static release-delta checks for staged v44 OPF admission correction.

v44 must preserve the v43 scientific/controller stack and change only the
versioned OPF reference overlay while pinning the newer private OPF source.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "tools/universal_training_controller_entry_v43_bundle.py"
NEW = ROOT / "tools/universal_training_controller_entry_v44_bundle.py"
OUTER = ROOT / "tools/universal_training_controller_entry_v44.py"
OVERLAY = ROOT / "controller_bundle_v44/universal_training_controller_opf_reference_v2.py"

EXPECTED_OPF_COMMIT = "3d27019c17b2e8a48ef03068535d5ed5fdadca03"
EXPECTED_OUTER_BLOB = "b9f79168e6d2af3406c2bc7625e0c7c46a452038"
EXPECTED_INNER_BLOB = "5400e1042f737f1fc0bb129e1087f76df2442e20"
EXPECTED_OVERLAY_BLOB = "79c39805d18ff6299daf81355e4dec3d4bb0b6fa"


def _blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _constants(path: Path, names: set[str]) -> dict:
    found = {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign):
            target, value = statement.target, statement.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id in names:
            found[target.id] = ast.literal_eval(value)
    assert set(found) == names, (path, names - set(found))
    return found


def test_v44_changes_only_controller_opf_overlay():
    old = _constants(OLD, {"CONTROLLER_FILES", "BUNDLE_VERSION"})
    new = _constants(NEW, {"CONTROLLER_FILES", "BUNDLE_VERSION"})
    assert old["BUNDLE_VERSION"] == "v43"
    assert new["BUNDLE_VERSION"] == "v44"
    assert len(old["CONTROLLER_FILES"]) == len(new["CONTROLLER_FILES"]) == 60
    changed = {
        key for key in old["CONTROLLER_FILES"]
        if old["CONTROLLER_FILES"][key] != new["CONTROLLER_FILES"][key]
    }
    assert changed == {"tools/universal_training_controller_opf_reference_v2.py"}
    assert new["CONTROLLER_FILES"]["tools/universal_training_controller_opf_reference_v2.py"] == (
        EXPECTED_OVERLAY_BLOB
    )


def test_v44_pins_exact_new_opf_delta():
    old = _constants(OLD, {"OPF_FILES", "OPF_COMMIT"})
    new = _constants(NEW, {"OPF_FILES", "OPF_COMMIT"})
    assert new["OPF_COMMIT"] == EXPECTED_OPF_COMMIT
    assert set(old["OPF_FILES"]) == set(new["OPF_FILES"])
    changed = {
        key for key in old["OPF_FILES"]
        if old["OPF_FILES"][key] != new["OPF_FILES"][key]
    }
    assert changed == {
        "utils/opf_massive_suite_runner.py",
        "utils/ml_backends.py",
        "utils/opf_shared_defaults.py",
        "DNN/VANILLA/Dyn_DNN4OPF/utils/run_defaults.py",
        "tests/test_massive_scheduler_operational_contract.py",
        "tests/test_ml_backend_admission_isolation.py",
        "tests/test_opf_shared_defaults_cuda_admission.py",
    }
    assert new["OPF_FILES"]["utils/opf_shared_defaults.py"] == (
        "85f1a948dca2f67a845ad129a6223484b47a0ad7"
    )
    assert new["OPF_FILES"]["tests/test_opf_shared_defaults_cuda_admission.py"] == (
        "0aa1aff969605643c455f70ed3383f0ddb6e6a14"
    )
    assert new["OPF_FILES"]["utils/runtime_tuning.py"] == old["OPF_FILES"]["utils/runtime_tuning.py"]
    assert new["OPF_FILES"]["utils/logging_utils.py"] == old["OPF_FILES"]["utils/logging_utils.py"]
    assert new["OPF_FILES"]["tests/test_ml_backends_gpu_first.py"] == (
        old["OPF_FILES"]["tests/test_ml_backends_gpu_first.py"]
    )


def test_v44_overlay_and_inner_manifest_are_identical():
    overlay = _constants(
        OVERLAY,
        {"OPF_REFERENCE_COMMIT", "OPF_RUNTIME_BLOBS", "OPF_OPERATIONAL_CONTRACT_BLOBS"},
    )
    inner = _constants(NEW, {"OPF_COMMIT", "OPF_FILES"})
    assert overlay["OPF_REFERENCE_COMMIT"] == inner["OPF_COMMIT"] == EXPECTED_OPF_COMMIT
    assert {
        **overlay["OPF_RUNTIME_BLOBS"],
        **overlay["OPF_OPERATIONAL_CONTRACT_BLOBS"],
    } == inner["OPF_FILES"]
    assert len(inner["OPF_FILES"]) == 10



def test_v44_outer_fetches_each_overlay_from_its_real_immutable_bundle_directory():
    values = _constants(OUTER, {"CONTROLLER_OVERLAY_LOCATIONS"})
    locations = values["CONTROLLER_OVERLAY_LOCATIONS"]
    assert locations["tools/universal_training_controller_opf_reference_v2.py"] == (
        "controller_bundle_v44"
    )
    inherited = {
        "tools/universal_training_controller_job_catalog_v2.py",
        "tools/universal_training_controller_workload_closure.py",
        "tools/universal_training_controller_scientific_surface_v25.py",
        "tools/universal_training_controller_selector_closure_v26.py",
        "tools/universal_training_controller_declaration_closure_v30.py",
    }
    assert {locations[key] for key in inherited} == {"controller_bundle_v42"}


def test_v44_outer_pins_exact_inner_and_release_files():
    values = _constants(OUTER, {"BUNDLE_VERSION", "V36_COMMIT", "V36_BLOB"})
    assert values["BUNDLE_VERSION"] == "v44"
    assert values["V36_COMMIT"] == "6a8db4d2f085ffe49a81943e31a6c7d9db09a51e"
    assert values["V36_BLOB"] == EXPECTED_INNER_BLOB
    assert _blob(NEW.read_bytes()) == EXPECTED_INNER_BLOB
    assert _blob(OVERLAY.read_bytes()) == EXPECTED_OVERLAY_BLOB
    assert _blob(OUTER.read_bytes()) == EXPECTED_OUTER_BLOB
