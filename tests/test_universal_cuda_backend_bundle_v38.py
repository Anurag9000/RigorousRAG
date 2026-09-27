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
