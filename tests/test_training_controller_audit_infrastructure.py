from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller_audit_infrastructure as scope
import universal_training_controller_current as current


def _sample_inventory(_root):
    return {
        "training_files": [
            {"path": "tools/training_surface_census.py", "training_logic": True},
            {"path": "tools/training_surface_semantic_scan.py", "training_logic": True},
            {"path": "training/real_learner.py", "training_logic": True},
        ],
        "executable_training_candidates": [
            "tools/training_surface_census.py",
            "training/real_learner.py",
        ],
        "model_surfaces": [],
        "training_logic_surfaces": [
            "tools/training_surface_census.py",
            "tools/training_surface_semantic_scan.py",
            "training/real_learner.py",
        ],
        "quiet_training_logic_surfaces": [],
        "excluded_nontraining_sources": [
            {"path": "tests/example.py", "reason": "test_source"},
        ],
    }


def test_install_wraps_training_inventory_and_preserves_real_learner():
    original = current._training_inventory
    try:
        current._training_inventory = _sample_inventory
        scope.install()
        report = current._training_inventory(ROOT)
    finally:
        current._training_inventory = original

    assert report["training_logic_surfaces"] == ["training/real_learner.py"]
    assert report["executable_training_candidates"] == ["training/real_learner.py"]
    assert [row["path"] for row in report["training_files"]] == ["training/real_learner.py"]
    ledger = {row["path"]: row["reason"] for row in report["excluded_nontraining_sources"]}
    assert ledger["tools/training_surface_census.py"] == "training_audit_infrastructure"
    assert ledger["tools/training_surface_semantic_scan.py"] == "training_audit_infrastructure"
    assert ledger["tests/example.py"] == "test_source"
    assert report["audit_infrastructure_scope_schema"] == scope.AUDIT_INFRASTRUCTURE_SCHEMA
