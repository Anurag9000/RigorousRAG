"""Offline source identity and bootstrap checks for staged controller v42."""
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
INNER = ROOT / "tools/universal_training_controller_entry_v42_bundle.py"
OUTER = ROOT / "tools/universal_training_controller_entry_v42.py"
V41_INNER = ROOT / "tools/universal_training_controller_entry_v41_bundle.py"
ACTIVE = ROOT / "tools/universal_training_controller_entry.py"
OVERLAY_DIR = ROOT / "controller_bundle_v42"

CHANGED_CONTROLLER_FILES = {
    "tools/universal_training_controller_job_catalog_v2.py",
    "tools/universal_training_controller_workload_closure.py",
    "tools/universal_training_controller_scientific_surface_v25.py",
    "tools/universal_training_controller_selector_closure_v26.py",
    "tools/universal_training_controller_declaration_closure_v30.py",
}
OVERLAY_FILES = CHANGED_CONTROLLER_FILES | {
    "tools/universal_training_controller_opf_reference_v2.py",
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
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "frozenset" and len(value.args) == 1):
                found[target.id] = frozenset(ast.literal_eval(value.args[0]))
            else:
                found[target.id] = ast.literal_eval(value)
    assert set(found) == names, (path, names - set(found))
    return found


def _outer_module():
    spec = importlib.util.spec_from_file_location("_v42_outer_test", OUTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v42_changes_only_five_controller_sources_relative_to_v41():
    old = _constants(V41_INNER, {"CONTROLLER_FILES", "OPF_FILES", "OPF_COMMIT", "BUNDLE_VERSION"})
    new = _constants(INNER, {"CONTROLLER_FILES", "OPF_FILES", "OPF_COMMIT", "BUNDLE_VERSION"})
    assert old["BUNDLE_VERSION"] == "v41"
    assert new["BUNDLE_VERSION"] == "v42"
    assert len(old["CONTROLLER_FILES"]) == len(new["CONTROLLER_FILES"]) == 60
    changed = {
        key for key in old["CONTROLLER_FILES"]
        if old["CONTROLLER_FILES"][key] != new["CONTROLLER_FILES"][key]
    }
    assert changed == CHANGED_CONTROLLER_FILES
    assert new["OPF_COMMIT"] == old["OPF_COMMIT"]
    assert new["OPF_FILES"] == old["OPF_FILES"]


def test_v42_overlay_files_match_manifest_and_current_audited_sources():
    manifest = _constants(INNER, {"CONTROLLER_FILES"})["CONTROLLER_FILES"]
    for relative in sorted(OVERLAY_FILES):
        overlay = OVERLAY_DIR / Path(relative).name
        assert overlay.is_file(), relative
        assert _blob(overlay.read_bytes()) == manifest[relative]
        if relative in CHANGED_CONTROLLER_FILES:
            current = ROOT / relative
            assert current.is_file(), relative
            assert overlay.read_bytes() == current.read_bytes()


def test_v42_outer_pins_exact_inner_and_declares_all_overlays():
    values = _constants(OUTER, {
        "BUNDLE_VERSION", "V36_COMMIT", "V36_BLOB", "CONTROLLER_OVERLAYS",
        "HOST_ARCHIVE_COMMIT", "HOST_BUNDLE_DIR",
    })
    assert values["BUNDLE_VERSION"] == "v42"
    assert values["V36_COMMIT"] == "820ec1251c1caadfa385591e7f25396963b05274"
    assert values["V36_BLOB"] == _blob(INNER.read_bytes()) == (
        "cd18a0ee6aa2f23ad945d817ee82e2974741172a"
    )
    assert set(values["CONTROLLER_OVERLAYS"]) == OVERLAY_FILES
    assert values["HOST_BUNDLE_DIR"] == "controller_bundle_v39"
    assert _blob(ACTIVE.read_bytes()) == "2f34cf0a1319c00d04bcfa99972f6e209534a096"


def _archive_for(relative: str, source: Path) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as handle:
        handle.writestr("rigorousrag/controller_bundle_v39/" + Path(relative).name,
                        source.read_bytes())
    return payload.getvalue()


def test_v42_multiple_overlays_materialize_and_are_hash_verified(tmp_path):
    outer = _outer_module()
    overlay_keys = sorted(CHANGED_CONTROLLER_FILES)[:2]
    archived_key = "tools/universal_training_controller.py"
    archived = ROOT / "controller_bundle_v39" / Path(archived_key).name
    manifest = _constants(INNER, {"CONTROLLER_FILES"})["CONTROLLER_FILES"]
    source_map = {key: manifest[key] for key in overlay_keys}
    source_map[archived_key] = _blob(archived.read_bytes())
    inner = types.SimpleNamespace(
        HOST_REPO="Anurag9000/RigorousRAG",
        BUNDLE_VERSION="v42",
        CONTROLLER_FILES=source_map,
    )
    archive = _archive_for(archived_key, archived)

    def fetched(url: str) -> bytes:
        if "controller_bundle_v42/" in url:
            name = url.rsplit("/", 1)[-1]
            return (OVERLAY_DIR / name).read_bytes()
        assert url == outer.ARCHIVE_URL
        return archive

    with mock.patch.object(outer, "_fetch", side_effect=fetched) as download:
        cache = outer._materialize_bundle(tmp_path, inner)
        for relative, expected in source_map.items():
            assert _blob((cache / Path(relative).name).read_bytes()) == expected
        marker = json.loads((cache / "BUNDLE.json").read_text(encoding="utf-8"))
        assert marker["files"] == source_map
        assert download.call_count == len(overlay_keys) + 1
        outer._materialize_bundle(tmp_path, inner)
        assert download.call_count == len(overlay_keys) + 1


def test_corrupt_v42_overlay_never_writes_bundle_certificate(tmp_path):
    outer = _outer_module()
    key = "tools/universal_training_controller_workload_closure.py"
    expected = _constants(INNER, {"CONTROLLER_FILES"})["CONTROLLER_FILES"][key]
    inner = types.SimpleNamespace(
        HOST_REPO="Anurag9000/RigorousRAG",
        BUNDLE_VERSION="v42",
        CONTROLLER_FILES={key: expected},
    )
    with mock.patch.object(outer, "_fetch", return_value=b"corrupt-overlay"):
        with pytest.raises(RuntimeError, match="overlay blob mismatch"):
            outer._materialize_bundle(tmp_path, inner)
    assert not any(tmp_path.rglob("BUNDLE.json"))
