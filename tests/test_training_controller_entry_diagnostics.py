from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller_entry as entry


def _blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _delegate(monkeypatch, argv: list[str], result: int = 0):
    calls: list[list[str]] = []

    class FakeV36:
        def main(self) -> int:
            calls.append(list(sys.argv))
            return result

    monkeypatch.setattr(entry, "bootstrap", lambda: FakeV36())
    monkeypatch.setattr(sys, "argv", list(argv))
    return calls


def test_audit_flag_is_delegated_unchanged_to_pinned_v36(monkeypatch) -> None:
    calls = _delegate(
        monkeypatch,
        ["universal_training_controller_entry.py", "--training-control-audit"],
    )
    assert entry.main() == 0
    assert calls == [["universal_training_controller_entry.py", "--training-control-audit"]]


def test_list_flag_is_delegated_unchanged_to_pinned_v36(monkeypatch) -> None:
    calls = _delegate(
        monkeypatch,
        ["universal_training_controller_entry.py", "--training-control-list-jobs"],
    )
    assert entry.main() == 0
    assert calls == [["universal_training_controller_entry.py", "--training-control-list-jobs"]]


def test_native_diagnostic_flag_is_delegated_unchanged(monkeypatch) -> None:
    calls = _delegate(
        monkeypatch,
        ["universal_training_controller_entry.py", "--audit-training-coverage"],
    )
    assert entry.main() == 0
    assert calls == [["universal_training_controller_entry.py", "--audit-training-coverage"]]


def test_delegate_exit_code_is_preserved(monkeypatch) -> None:
    _delegate(monkeypatch, ["universal_training_controller_entry.py", "--skip-setup"], result=17)
    assert entry.main() == 17


def test_bootstrap_only_prepares_entry_loads_v36_and_materializes_bundle(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRAINING_CONTROL_REPO_ROOT", str(tmp_path))
    pinned = tmp_path / "pinned-v36.py"
    fake_v36 = SimpleNamespace()
    events: list[tuple[str, object]] = []

    def prepare(root: Path) -> Path:
        events.append(("prepare", root))
        return pinned

    def load(path: Path):
        events.append(("load", path))
        return fake_v36

    def materialize(root: Path, module):
        events.append(("materialize", (root, module)))
        return tmp_path / "bundle"

    monkeypatch.setattr(entry, "_prepare_v36_entry", prepare)
    monkeypatch.setattr(entry, "_load_v36", load)
    monkeypatch.setattr(entry, "_materialize_bundle", materialize)

    assert entry.bootstrap() is fake_v36
    assert events == [
        ("prepare", tmp_path.resolve()),
        ("load", pinned),
        ("materialize", (tmp_path.resolve(), fake_v36)),
    ]


def _zip_with_bundle(name: str, payload: bytes) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"RigorousRAG-test/{entry.HOST_BUNDLE_DIR}/{name}", payload)
    return stream.getvalue()


def test_materialize_bundle_accepts_only_exact_blob(monkeypatch, tmp_path: Path) -> None:
    payload = b"print('immutable controller')\n"
    expected = _blob(payload)
    module = SimpleNamespace(
        CONTROLLER_FILES={"tools/example_controller.py": expected},
        HOST_REPO=entry.HOST_REPO,
    )
    monkeypatch.setattr(entry, "_fetch", lambda _url: _zip_with_bundle("example_controller.py", payload))

    cache = entry._materialize_bundle(tmp_path, module)
    materialized = cache / "example_controller.py"
    assert materialized.read_bytes() == payload
    assert entry.git_blob_sha(materialized.read_bytes()) == expected
    assert (cache / "BUNDLE.json").is_file()


def test_materialize_bundle_rejects_archive_blob_drift(monkeypatch, tmp_path: Path) -> None:
    expected_payload = b"expected\n"
    wrong_payload = b"wrong\n"
    module = SimpleNamespace(
        CONTROLLER_FILES={"tools/example_controller.py": _blob(expected_payload)},
        HOST_REPO=entry.HOST_REPO,
    )
    monkeypatch.setattr(entry, "_fetch", lambda _url: _zip_with_bundle("example_controller.py", wrong_payload))

    with pytest.raises(RuntimeError, match="Controller bundle blob mismatch"):
        entry._materialize_bundle(tmp_path, module)


def test_bundle_lookup_flattens_historical_tool_paths() -> None:
    assert entry._bundle_member("tools/universal_training_controller_v34.py") == (
        f"{entry.HOST_BUNDLE_DIR}/universal_training_controller_v34.py"
    )


def test_v37_contains_no_scheduler_implementation() -> None:
    source = Path(entry.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "def resolve_concurrency(",
        "def resolve_launch_capacity(",
        "def select_launch_device(",
        "nvidia-smi",
    ):
        assert forbidden not in source
