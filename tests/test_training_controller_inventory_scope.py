from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller_inventory_scope as inventory_scope


def test_inventory_scope_excludes_only_structural_infrastructure() -> None:
    report = {
        "training_files": [
            {"path": "training/real_trainer.py", "training_logic": True},
            {"path": "tests/test_real_trainer.py", "training_logic": True},
            {"path": "tools/universal_training_controller_exact_resume.py", "model_surface": True},
            {"path": "tools/account_wide_training_control_audit.py", "training_logic": True},
        ],
        "executable_training_candidates": [
            "training/real_trainer.py",
            "tests/test_real_trainer.py",
        ],
        "model_surfaces": [
            "training/real_model.py",
            "tests/unit/test_public_payload.py",
            "tools/universal_training_controller_exact_resume.py",
        ],
        "training_logic_surfaces": [
            "training/real_trainer.py",
            "tests/test_real_trainer.py",
            "tools/account_wide_training_control_audit.py",
        ],
    }

    filtered = inventory_scope._filter_inventory(report)

    assert filtered["executable_training_candidates"] == ["training/real_trainer.py"]
    assert filtered["model_surfaces"] == ["training/real_model.py"]
    assert filtered["training_logic_surfaces"] == ["training/real_trainer.py"]
    assert [row["path"] for row in filtered["training_files"]] == ["training/real_trainer.py"]

    ledger = {row["path"]: row["reason"] for row in filtered["excluded_nontraining_sources"]}
    assert ledger["tests/test_real_trainer.py"] == "test_source"
    assert ledger["tests/unit/test_public_payload.py"] == "test_source"
    assert ledger["tools/universal_training_controller_exact_resume.py"] == "training_controller_infrastructure"
    assert ledger["tools/account_wide_training_control_audit.py"] == "account_audit_infrastructure"


def test_inventory_scope_does_not_hide_repository_training_or_model_paths() -> None:
    assert inventory_scope._exclusion_reason("training/model_architectures.py") is None
    assert inventory_scope._exclusion_reason("scripts/test_time_training.py") is None
    assert inventory_scope._exclusion_reason("models/local_hf_adapters.py") is None
