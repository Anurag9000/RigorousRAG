from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller_entry as entry


def _isolate_entry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRAINING_CONTROL_REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("TRAINING_CONTROL_PREPARE_LEGACY_OPF", raising=False)
    monkeypatch.setattr(entry, "FILES", {})
    monkeypatch.setattr(entry.subprocess, "call", lambda *_args, **_kwargs: 0)


def test_audit_does_not_materialize_private_opf(monkeypatch, tmp_path: Path) -> None:
    _isolate_entry(monkeypatch, tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("diagnostic mode attempted OPF runtime materialization")

    monkeypatch.setattr(entry, "prepare_reference_cache", forbidden)
    monkeypatch.setattr(sys, "argv", ["universal_training_controller_entry.py", "--training-control-audit"])
    assert entry.main() == 0


def test_list_does_not_materialize_private_opf(monkeypatch, tmp_path: Path) -> None:
    _isolate_entry(monkeypatch, tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("list mode attempted OPF runtime materialization")

    monkeypatch.setattr(entry, "prepare_reference_cache", forbidden)
    monkeypatch.setattr(sys, "argv", ["universal_training_controller_entry.py", "--list-training-jobs"])
    assert entry.main() == 0


def test_real_run_materializes_only_current_opf_by_default(monkeypatch, tmp_path: Path) -> None:
    _isolate_entry(monkeypatch, tmp_path)
    calls: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(entry, "prepare_reference_cache", lambda _root, commit, files: calls.append((commit, files)))
    monkeypatch.setattr(sys, "argv", ["universal_training_controller_entry.py", "--skip-setup"])
    assert entry.main() == 0
    assert calls == [(entry.OPF_COMMIT, entry.OPF_FILES)]


def test_legacy_reference_is_explicit_opt_in(monkeypatch, tmp_path: Path) -> None:
    _isolate_entry(monkeypatch, tmp_path)
    monkeypatch.setenv("TRAINING_CONTROL_PREPARE_LEGACY_OPF", "1")
    calls: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(entry, "prepare_reference_cache", lambda _root, commit, files: calls.append((commit, files)))
    monkeypatch.setattr(sys, "argv", ["universal_training_controller_entry.py", "--skip-setup"])
    assert entry.main() == 0
    assert calls == [
        (entry.OPF_COMMIT, entry.OPF_FILES),
        (entry.LEGACY_OPF_COMMIT, entry.LEGACY_OPF_FILES),
    ]
