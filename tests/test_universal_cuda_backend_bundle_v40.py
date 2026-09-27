"""Offline source and materialization checks for the staged v40 CUDA reference.

The active v39 root and archived scientific blobs remain unchanged. This test
does not claim successful GitHub bootstrap or GPU execution on an actual worker.
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
INNER = ROOT / "tools/universal_training_controller_entry_v40_bundle.py"
OUTER = ROOT / "tools/universal_training_controller_entry_v40.py"
OVERLAY = ROOT / "controller_bundle_v40/universal_training_controller_opf_reference_v2.py"
OLD = ROOT / "tools/universal_training_controller_entry_v39_bundle.py"


def _blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _constants(path: Path, names: set[str]) -> dict:
    found = {}
    for statement in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = ast.literal_eval(statement.value)
    assert set(found) == names, (path, names - set(found))
    return found


def _outer_module():
    spec = importlib.util.spec_from_file_location("_v40_staged_outer", OUTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v40_scientific_sources_are_identical_to_v39_except_new_opf_overlay():
    keys = {"CONTROLLER_FILES", "OPF_FILES", "OPF_COMMIT", "BUNDLE_VERSION"}
    old = _constants(OLD, keys)
    new = _constants(INNER, keys)
    assert old["BUNDLE_VERSION"] == "v39"
    assert new["BUNDLE_VERSION"] == "v40"
    assert len(new["CONTROLLER_FILES"]) == len(old["CONTROLLER_FILES"]) == 60
    changed = {k for k in old["CONTROLLER_FILES"]
               if old["CONTROLLER_FILES"][k] != new["CONTROLLER_FILES"][k]}
    assert changed == {"tools/universal_training_controller_opf_reference_v2.py"}
    for relative, expected in new["CONTROLLER_FILES"].items():
        actual = OVERLAY if relative in changed else ROOT / "controller_bundle_v39" / Path(relative).name
        assert actual.is_file(), relative
        assert _blob(actual.read_bytes()) == expected, relative

    overlay = _constants(OVERLAY, {
        "OPF_REFERENCE_COMMIT", "OPF_RUNTIME_BLOBS", "OPF_OPERATIONAL_CONTRACT_BLOBS",
    })
    assert new["OPF_COMMIT"] == overlay["OPF_REFERENCE_COMMIT"] == (
        "d3687061296a4cfc063bce149096dcc1d973716b"
    )
    assert new["OPF_FILES"] == {
        **overlay["OPF_RUNTIME_BLOBS"],
        **overlay["OPF_OPERATIONAL_CONTRACT_BLOBS"],
    }
    assert new["OPF_FILES"]["utils/ml_backends.py"] == "d8f808be88c19b32d82d66e31da1a45dd7b56e27"
    assert new["OPF_FILES"]["utils/opf_massive_suite_runner.py"] == "30067d5aeb6c1cc3debad8be086055c110c0899c"
    assert "tests/test_ml_backend_admission_isolation.py" in new["OPF_FILES"]


def test_v40_outer_pins_new_inner_and_unchanged_scientific_archive():
    old = _constants(ROOT / "tools/universal_training_controller_entry.py", {
        "BUNDLE_VERSION", "HOST_ARCHIVE_COMMIT", "HOST_BUNDLE_DIR", "V36_BLOB",
    })
    new = _constants(OUTER, {
        "BUNDLE_VERSION", "HOST_ARCHIVE_COMMIT", "HOST_BUNDLE_DIR", "V36_BLOB",
        "V36_COMMIT",
    })
    assert old["BUNDLE_VERSION"] == "v39"
    assert new["BUNDLE_VERSION"] == "v40"
    assert new["HOST_ARCHIVE_COMMIT"] == old["HOST_ARCHIVE_COMMIT"]
    assert new["HOST_BUNDLE_DIR"] == old["HOST_BUNDLE_DIR"] == "controller_bundle_v39"
    assert new["V36_BLOB"] == _blob(INNER.read_bytes())
    assert new["V36_COMMIT"] == "7ed16bade494d2191962756a1bf1ca87d8693245"


def _archive_for(source: Path) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as handle:
        handle.writestr("rigorousrag/controller_bundle_v39/" + source.name, source.read_bytes())
    return payload.getvalue()


def test_v40_overlay_materialization_is_byte_verified_without_network(tmp_path):
    outer = _outer_module()
    archived = ROOT / "controller_bundle_v39/universal_training_controller.py"
    overlay_key = "tools/universal_training_controller_opf_reference_v2.py"
    archived_key = "tools/universal_training_controller.py"
    source_map = {overlay_key: _blob(OVERLAY.read_bytes()),
                  archived_key: _blob(archived.read_bytes())}
    inner = types.SimpleNamespace(
        HOST_REPO="Anurag9000/RigorousRAG",
        BUNDLE_VERSION="v40",
        CONTROLLER_FILES=source_map,
    )
    archive = _archive_for(archived)
    def fetched(url):
        if "controller_bundle_v40/" in url:
            return OVERLAY.read_bytes()
        assert url == outer.ARCHIVE_URL
        return archive

    with mock.patch.object(outer, "_fetch", side_effect=fetched) as download:
        cache = outer._materialize_bundle(tmp_path, inner)
        assert _blob((cache / OVERLAY.name).read_bytes()) == source_map[overlay_key]
        assert _blob((cache / archived.name).read_bytes()) == source_map[archived_key]
        assert json.loads((cache / "BUNDLE.json").read_text())["files"] == source_map
        assert download.call_count == 2
        assert outer._materialize_bundle(tmp_path, inner) == cache
        assert download.call_count == 2


def test_corrupted_v40_overlay_is_rejected_before_cache_certificate(tmp_path):
    outer = _outer_module()
    key = "tools/universal_training_controller_opf_reference_v2.py"
    inner = types.SimpleNamespace(HOST_REPO="Anurag9000/RigorousRAG",
                                  BUNDLE_VERSION="v40",
                                  CONTROLLER_FILES={key: _blob(OVERLAY.read_bytes())})
    with mock.patch.object(outer, "_fetch", return_value=b"incorrect-overlay"):
        with pytest.raises(RuntimeError, match="overlay blob mismatch"):
            outer._materialize_bundle(tmp_path, inner)
    assert not any(tmp_path.rglob("BUNDLE.json"))
