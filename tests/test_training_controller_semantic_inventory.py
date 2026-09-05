from __future__ import annotations

import importlib
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

semantic = importlib.import_module("universal_training_controller_semantic_inventory")


def _scanner(*, learners=(), errors=(), evidence=None):
    payload = {
        "schema": "test.semantic.v1",
        "inventory_sha256": "a" * 64,
        "base_inventory_sha256": "b" * 64,
        "scanned_by_language": {"python": len(tuple(learners))},
        "learner_files": list(learners),
        "parse_errors": list(errors),
        "evidence": evidence or {path: [{"kind": "function"}] for path in learners},
    }
    return lambda root: payload


def test_unaccounted_production_learner_fails_inventory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        semantic,
        "_load_scanner",
        lambda: _scanner(learners=("training/quiet_fit.py",)),
    )
    monkeypatch.setattr(
        semantic.current,
        "_reachability",
        lambda root, jobs: {"reachable_sources": []},
    )
    report = semantic._production_semantic_inventory(tmp_path, [])
    assert report["unaccounted_learner_files"] == ["training/quiet_fit.py"]
    assert report["accounted_learner_files"] == []


def test_reachable_semantic_learner_is_accounted(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        semantic,
        "_load_scanner",
        lambda: _scanner(learners=("training/quiet_fit.py",)),
    )
    monkeypatch.setattr(
        semantic.current,
        "_reachability",
        lambda root, jobs: {"reachable_sources": ["training/quiet_fit.py"]},
    )
    report = semantic._production_semantic_inventory(tmp_path, [{"id": "fit"}])
    assert report["unaccounted_learner_files"] == []
    assert report["accounted_learner_files"] == ["training/quiet_fit.py"]


def test_structural_tests_and_controller_files_do_not_become_production_obligations(monkeypatch, tmp_path: Path) -> None:
    learners = (
        "tests/test_training.py",
        "tools/universal_training_controller_current.py",
        "training/real_fit.py",
    )
    monkeypatch.setattr(semantic, "_load_scanner", lambda: _scanner(learners=learners))
    monkeypatch.setattr(
        semantic.current,
        "_reachability",
        lambda root, jobs: {"reachable_sources": ["training/real_fit.py"]},
    )
    report = semantic._production_semantic_inventory(tmp_path, [])
    assert report["production_learner_files"] == ["training/real_fit.py"]
    assert report["unaccounted_learner_files"] == []


def test_production_parse_errors_fail_but_test_parse_errors_are_structural(monkeypatch, tmp_path: Path) -> None:
    errors = (
        "training/broken_fit.py:7: syntax error: invalid syntax",
        "tests/test_broken.py:3: syntax error: invalid syntax",
    )
    monkeypatch.setattr(semantic, "_load_scanner", lambda: _scanner(errors=errors))
    monkeypatch.setattr(
        semantic.current,
        "_reachability",
        lambda root, jobs: {"reachable_sources": []},
    )
    report = semantic._production_semantic_inventory(tmp_path, [])
    assert report["production_parse_errors"] == [errors[0]]


def test_install_ands_semantic_result_into_coverage_ok(monkeypatch, tmp_path: Path) -> None:
    base_report = semantic.current._enhanced_coverage_report
    monkeypatch.setattr(
        semantic.current,
        "_enhanced_coverage_report",
        lambda root, profile, jobs: {"coverage_ok": True, "strict_controls": {}},
    )
    monkeypatch.setattr(
        semantic,
        "_production_semantic_inventory",
        lambda root, jobs: {
            "schema": "test.semantic.v1",
            "inventory_sha256": "a" * 64,
            "base_inventory_sha256": "b" * 64,
            "scanned_by_language": {"python": 1},
            "production_learner_files": ["training/missing.py"],
            "production_learner_evidence": {"training/missing.py": [{"kind": "function"}]},
            "accounted_learner_files": [],
            "unaccounted_learner_files": ["training/missing.py"],
            "production_parse_errors": [],
            "structural_exclusion_policy": {},
        },
    )
    semantic.install()
    try:
        report = semantic.current._enhanced_coverage_report(tmp_path, {}, [])
        assert report["coverage_ok"] is False
        assert report["strict_semantic_training_surface_pass"] is False
        assert report["unaccounted_semantic_learner_surfaces"] == ["training/missing.py"]
        assert report["strict_controls"]["require_semantic_training_surface_accounting"] is True
    finally:
        monkeypatch.setattr(semantic.current, "_enhanced_coverage_report", base_report)
