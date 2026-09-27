"""Immutable v38 shared controller release: source identity and backend pins."""
from pathlib import Path
import ast
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def blob(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def test_v38_bundle_filenames_and_blobs():
    source = ROOT / "tools" / "universal_training_controller_entry_v38_bundle.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    literals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in {"CONTROLLER_FILES", "OPF_FILES", "OPF_COMMIT"}:
                literals[key] = ast.literal_eval(node.value)
    assert len(literals["CONTROLLER_FILES"]) == 60
    assert literals["OPF_COMMIT"] == "1d3dea055736b1261a621436626ce89d69242a14"
    assert literals["OPF_FILES"]["utils/ml_backends.py"] == "33108a3e20e982188ebc089399c682b11f202c4c"
    assert literals["OPF_FILES"]["tests/test_ml_backends_gpu_first.py"] == "1dcc47b0195740acdca56c48ffddd648ef7c5fb3"
    for relative, expected in literals["CONTROLLER_FILES"].items():
        path = ROOT / "controller_bundle_v38" / Path(relative).name
        assert path.is_file(), relative
        assert blob(path.read_bytes()) == expected, relative

def test_v38_loader_pins_same_bundle_release():
    tree = ast.parse((ROOT / "tools" / "universal_training_controller_entry.py").read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"BUNDLE_VERSION", "HOST_ARCHIVE_COMMIT", "HOST_BUNDLE_DIR", "V36_COMMIT", "V36_BLOB"}:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    assert values["BUNDLE_VERSION"] == "v38"
    assert values["HOST_ARCHIVE_COMMIT"] == "470c5826d63f456195e61bc4aff73a97ebb88e52"
    assert values["HOST_BUNDLE_DIR"] == "controller_bundle_v38"
    assert values["V36_COMMIT"] == "470c5826d63f456195e61bc4aff73a97ebb88e52"
    assert values["V36_BLOB"] == "1589405dae02202973afc75c4a55710c2e16d9b2"
    manifest = ROOT / "tools" / "universal_training_controller_entry_v38_bundle.py"
    assert blob(manifest.read_bytes()) == values["V36_BLOB"]


def test_outer_archive_cache_reused_by_inner_v38(tmp_path):
    """One verified archive must satisfy the inner loader without API blob fetches."""
    import importlib.util
    import json
    from types import ModuleType

    outer_path = ROOT / "tools" / "universal_training_controller_entry.py"
    inner_path = ROOT / "tools" / "universal_training_controller_entry_v38_bundle.py"
    outer_spec = importlib.util.spec_from_file_location("outer_v38_cache_test", outer_path)
    inner_spec = importlib.util.spec_from_file_location("inner_v38_cache_test", inner_path)
    assert outer_spec is not None and outer_spec.loader is not None
    assert inner_spec is not None and inner_spec.loader is not None
    outer = importlib.util.module_from_spec(outer_spec)
    inner = importlib.util.module_from_spec(inner_spec)
    outer_spec.loader.exec_module(outer)
    inner_spec.loader.exec_module(inner)
    payload = b"verified-controller-source\\n"
    relative = "tools/example_controller.py"
    inner.CONTROLLER_FILES = {relative: blob(payload)}
    cache, files, marker = outer._bundle_cache(tmp_path, inner)
    assert cache.name.startswith("v38-")
    cache.mkdir(parents=True)
    (cache / "example_controller.py").write_bytes(payload)
    (cache / "BUNDLE.json").write_text(json.dumps(marker), encoding="utf-8")
    inner.fetch_git_blob = lambda *args: (_ for _ in ()).throw(
        AssertionError("inner bootstrap must reuse the verified archive cache")
    )
    assert inner.prepare_controller_cache(tmp_path) == cache
    assert outer._cache_valid(cache, files, marker)
