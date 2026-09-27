"""Immutable v39 controller release pins CUDA-isolated OPF v21 without rewriting v38."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + bytes((0,)) + data).hexdigest()


def _values(path: Path, keys: set[str]):
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in keys:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    return values


def test_v39_bundle_exact_source_and_opf_pins():
    source = ROOT / "tools" / "universal_training_controller_entry_v39_bundle.py"
    manifest = _values(source, {"CONTROLLER_FILES", "OPF_FILES", "OPF_COMMIT", "BUNDLE_VERSION"})
    assert manifest["BUNDLE_VERSION"] == "v39"
    assert len(manifest["CONTROLLER_FILES"]) == 60
    assert manifest["OPF_COMMIT"] == "0e9e4eef90903dd1769b54b18941a8bb6e1716f8"
    assert manifest["OPF_FILES"]["utils/opf_massive_suite_runner.py"] == "08cd45ec97626986f72cc1e614ad1b27b5288980"
    assert manifest["OPF_FILES"]["tests/test_massive_scheduler_operational_contract.py"] == "ad315baee8dcda194b72f5f0032ae72e5cf75044"
    assert manifest["OPF_FILES"]["utils/ml_backends.py"] == "33108a3e20e982188ebc089399c682b11f202c4c"
    for relative, expected in manifest["CONTROLLER_FILES"].items():
        path = ROOT / "controller_bundle_v39" / Path(relative).name
        assert path.is_file(), relative
        assert _blob(path.read_bytes()) == expected, relative


def test_outer_entry_pins_immutable_v39_stage_and_v38_snapshot():
    outer = _values(ROOT / "tools" / "universal_training_controller_entry_v39.py", {
        "BUNDLE_VERSION", "HOST_ARCHIVE_COMMIT", "HOST_BUNDLE_DIR", "V36_COMMIT", "V36_BLOB",
    })
    assert outer == {
        "BUNDLE_VERSION": "v39",
        "HOST_ARCHIVE_COMMIT": "ef4f9f336f78c27a6238ff6e91b8926eb9157dbe",
        "HOST_BUNDLE_DIR": "controller_bundle_v39",
        "V36_COMMIT": "ef4f9f336f78c27a6238ff6e91b8926eb9157dbe",
        "V36_BLOB": "1807ec85762aca16d27ce70344944927fc26689a",
    }
    assert _blob((ROOT / "tools" / "universal_training_controller_entry_v39_bundle.py").read_bytes()) == outer["V36_BLOB"]
    assert _blob((ROOT / "tools" / "universal_training_controller_entry_v38.py").read_bytes()) == "a341f0039139a9e9984476c370123df92da896b3"
