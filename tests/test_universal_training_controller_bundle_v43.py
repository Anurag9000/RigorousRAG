"""Offline invariants for staged controller v43.

v43 must preserve every v42 scientific/scanner overlay byte-for-byte and change
only the OPF reference to the isolated CUDA/RNG compatibility snapshot.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import types
import zipfile
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
INNER = ROOT / "tools/universal_training_controller_entry_v43_bundle.py"
OUTER = ROOT / "tools/universal_training_controller_entry_v43.py"
V42_INNER = ROOT / "tools/universal_training_controller_entry_v42_bundle.py"
ACTIVE = ROOT / "tools/universal_training_controller_entry.py"
OVERLAY_DIR = ROOT / "controller_bundle_v43"
V42_OVERLAY_DIR = ROOT / "controller_bundle_v42"

SCIENTIFIC_OVERLAYS = {
    "tools/universal_training_controller_job_catalog_v2.py",
    "tools/universal_training_controller_workload_closure.py",
    "tools/universal_training_controller_scientific_surface_v25.py",
    "tools/universal_training_controller_selector_closure_v26.py",
    "tools/universal_training_controller_declaration_closure_v30.py",
}
OPF_OVERLAY = "tools/universal_training_controller_opf_reference_v2.py"
EXPECTED_OPF_COMMIT = "d6cb36ae9c526b5d2c7d7869e7d0cb380584b0c2"
EXPECTED_OPF_CHANGES = {
    "utils/ml_backends.py",
    "utils/opf_shared_defaults.py",
    "tests/test_ml_backend_admission_isolation.py",
}
EXPECTED_OPF_ADDITIONS = {
    "tests/test_opf_shared_defaults_cuda_admission.py",
}


def _blob(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def _constants(path: Path, names: set[str]) -> dict:
    found = {}
    for statement in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign):
            target, value = statement.target, statement.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id in names:
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "frozenset"
                and len(value.args) == 1
            ):
                found[target.id] = frozenset(ast.literal_eval(value.args[0]))
            else:
                found[target.id] = ast.literal_eval(value)
    assert set(found) == names, (path, names - set(found))
    return found


def _outer_module():
    spec = importlib.util.spec_from_file_location("_v43_outer_test", OUTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v43_changes_only_opf_overlay_relative_to_v42():
    keys = {"CONTROLLER_FILES", "OPF_FILES", "OPF_COMMIT", "BUNDLE_VERSION"}
    old = _constants(V42_INNER, keys)
    new = _constants(INNER, keys)

    assert old["BUNDLE_VERSION"] == "v42"
    assert new["BUNDLE_VERSION"] == "v43"
    assert len(old["CONTROLLER_FILES"]) == len(new["CONTROLLER_FILES"]) == 60
    changed_controller = {
        key for key in old["CONTROLLER_FILES"]
        if old["CONTROLLER_FILES"][key] != new["CONTROLLER_FILES"][key]
    }
    assert changed_controller == {OPF_OVERLAY}

    changed_opf = {
        key for key in old["OPF_FILES"]
        if old["OPF_FILES"][key] != new["OPF_FILES"][key]
    }
    added_opf = set(new["OPF_FILES"]) - set(old["OPF_FILES"])
    removed_opf = set(old["OPF_FILES"]) - set(new["OPF_FILES"])
    assert changed_opf == EXPECTED_OPF_CHANGES
    assert added_opf == EXPECTED_OPF_ADDITIONS
    assert removed_opf == set()
    assert old["OPF_COMMIT"] == "a361b49ac24ffc7de87440538c88994faed017c4"
    assert new["OPF_COMMIT"] == EXPECTED_OPF_COMMIT
    assert new["OPF_FILES"]["utils/ml_backends.py"] == (
        "d8645afd18294b88b419bcb5335169e83afefcb4"
    )
    assert new["OPF_FILES"]["utils/opf_shared_defaults.py"] == (
        "08cb452d16c1a8c1955267d37294e7eca6674750"
    )
    assert new["OPF_FILES"]["tests/test_ml_backend_admission_isolation.py"] == (
        "a2c90f121413b34ac23b8b6baae5b6be04d914ef"
    )
    assert new["OPF_FILES"]["tests/test_opf_shared_defaults_cuda_admission.py"] == (
        "892c3da87837cc8157eed00efc5af1c24281b2ba"
    )


def test_v43_scientific_overlays_are_byte_identical_to_v42():
    manifest = _constants(INNER, {"CONTROLLER_FILES"})["CONTROLLER_FILES"]
    for relative in sorted(SCIENTIFIC_OVERLAYS):
        name = Path(relative).name
        v42 = V42_OVERLAY_DIR / name
        v43 = OVERLAY_DIR / name
        assert v42.is_file() and v43.is_file(), relative
        assert v43.read_bytes() == v42.read_bytes(), relative
        assert _blob(v43.read_bytes()) == manifest[relative]

    opf = OVERLAY_DIR / Path(OPF_OVERLAY).name
    assert opf.is_file()
    assert _blob(opf.read_bytes()) == manifest[OPF_OVERLAY] == (
        "f90151356bc031a3d1d9e54f4e040340f7469349"
    )
    overlay = _constants(opf, {
        "OPF_REFERENCE_COMMIT", "OPF_RUNTIME_BLOBS",
        "OPF_OPERATIONAL_CONTRACT_BLOBS",
    })
    assert overlay["OPF_REFERENCE_COMMIT"] == EXPECTED_OPF_COMMIT
    assert {
        **overlay["OPF_RUNTIME_BLOBS"],
        **overlay["OPF_OPERATIONAL_CONTRACT_BLOBS"],
    } == _constants(INNER, {"OPF_FILES"})["OPF_FILES"]


def test_v43_outer_pins_exact_inner_and_preserves_active_v41():
    values = _constants(OUTER, {
        "BUNDLE_VERSION", "V36_COMMIT", "V36_BLOB", "CONTROLLER_OVERLAYS",
        "HOST_ARCHIVE_COMMIT", "HOST_BUNDLE_DIR",
    })
    assert values["BUNDLE_VERSION"] == "v43"
    assert values["V36_COMMIT"] == "ff1f408b2c4538f6520dcb6154a299f736a01ab3"
    assert values["V36_BLOB"] == _blob(INNER.read_bytes()) == (
        "72634296a3598eee81c75fa0acc75a843865cdf6"
    )
    assert set(values["CONTROLLER_OVERLAYS"]) == SCIENTIFIC_OVERLAYS | {OPF_OVERLAY}
    assert values["HOST_BUNDLE_DIR"] == "controller_bundle_v39"
    # Staging v43 must not silently activate it before validation.
    assert _blob(ACTIVE.read_bytes()) == "2f34cf0a1319c00d04bcfa99972f6e209534a096"


def _archive_for(relative: str, source: Path) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as handle:
        handle.writestr(
            "rigorousrag/controller_bundle_v39/" + Path(relative).name,
            source.read_bytes(),
        )
    return payload.getvalue()


def test_v43_overlay_materialization_verifies_hash_before_bundle_marker(tmp_path):
    outer = _outer_module()
    opf_key = OPF_OVERLAY
    archived_key = "tools/universal_training_controller.py"
    opf_source = OVERLAY_DIR / Path(opf_key).name
    archived = ROOT / "controller_bundle_v39" / Path(archived_key).name
    source_map = {
        opf_key: _blob(opf_source.read_bytes()),
        archived_key: _blob(archived.read_bytes()),
    }
    inner = types.SimpleNamespace(
        HOST_REPO="Anurag9000/RigorousRAG",
        BUNDLE_VERSION="v43",
        CONTROLLER_FILES=source_map,
    )
    archive = _archive_for(archived_key, archived)

    def fetched(url):
        if "controller_bundle_v43/" in url:
            return opf_source.read_bytes()
        assert url == outer.ARCHIVE_URL
        return archive

    with mock.patch.object(outer, "_fetch", side_effect=fetched) as download:
        cache = outer._materialize_bundle(tmp_path, inner)
        assert _blob((cache / opf_source.name).read_bytes()) == source_map[opf_key]
        assert _blob((cache / archived.name).read_bytes()) == source_map[archived_key]
        assert json.loads((cache / "BUNDLE.json").read_text())["files"] == source_map
        assert download.call_count == 2


def test_corrupted_v43_opf_overlay_is_rejected_before_cache_certificate(tmp_path):
    outer = _outer_module()
    opf = OVERLAY_DIR / Path(OPF_OVERLAY).name
    inner = types.SimpleNamespace(
        HOST_REPO="Anurag9000/RigorousRAG",
        BUNDLE_VERSION="v43",
        CONTROLLER_FILES={OPF_OVERLAY: _blob(opf.read_bytes())},
    )
    with mock.patch.object(outer, "_fetch", return_value=b"corrupted-overlay"):
        with pytest.raises(RuntimeError, match="v43 controller overlay blob mismatch"):
            outer._materialize_bundle(tmp_path, inner)
    assert not any(tmp_path.rglob("BUNDLE.json"))
