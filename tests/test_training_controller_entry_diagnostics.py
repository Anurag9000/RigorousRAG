from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller_entry as entry


def _isolate_entry(monkeypatch, tmp_path: Path, calls: list | None = None) -> None:
    monkeypatch.setenv("TRAINING_CONTROL_REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("TRAINING_CONTROL_PREPARE_LEGACY_OPF", raising=False)
    monkeypatch.setattr(entry, "FILES", {})
    if calls is None:
        monkeypatch.setattr(entry.subprocess, "call", lambda *_args, **_kwargs: 0)
    else:
        def capture(args, **kwargs):
            calls.append((list(args), kwargs))
            return 0
        monkeypatch.setattr(entry.subprocess, "call", capture)


def test_audit_does_not_materialize_private_opf_and_forwards_canonical_flag(monkeypatch, tmp_path: Path) -> None:
    calls: list = []
    _isolate_entry(monkeypatch, tmp_path, calls)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("diagnostic mode attempted OPF runtime materialization")

    monkeypatch.setattr(entry, "prepare_reference_cache", forbidden)
    monkeypatch.setattr(sys, "argv", ["universal_training_controller_entry.py", "--training-control-audit"])
    assert entry.main() == 0
    assert len(calls) == 1
    assert calls[0][0][-1] == "--audit-training-coverage"
    assert "--training-control-audit" not in calls[0][0]


def test_list_does_not_materialize_private_opf_and_aliases_consistently(monkeypatch, tmp_path: Path) -> None:
    calls: list = []
    _isolate_entry(monkeypatch, tmp_path, calls)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("list mode attempted OPF runtime materialization")

    monkeypatch.setattr(entry, "prepare_reference_cache", forbidden)
    monkeypatch.setattr(sys, "argv", ["universal_training_controller_entry.py", "--training-control-list-jobs"])
    assert entry.main() == 0
    assert calls[0][0][-1] == "--list-training-jobs"


def test_native_diagnostic_flags_are_idempotent(monkeypatch, tmp_path: Path) -> None:
    calls: list = []
    _isolate_entry(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(entry, "prepare_reference_cache", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected OPF materialization")))
    monkeypatch.setattr(sys, "argv", ["universal_training_controller_entry.py", "--audit-training-coverage"])
    assert entry.main() == 0
    assert calls[0][0][-1] == "--audit-training-coverage"


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
