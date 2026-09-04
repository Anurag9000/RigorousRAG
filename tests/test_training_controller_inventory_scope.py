from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller_inventory_scope as inventory_scope


def _touch(root: Path, *paths: str) -> None:
    for rel in paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")


def test_inventory_scope_excludes_only_structural_infrastructure(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _touch(
        root,
        "training/real_trainer.py",
        "training/real_model.py",
        "tests/test_real_trainer.py",
        "tests/unit/test_public_payload.py",
        "tools/universal_training_controller_exact_resume.py",
        "tools/account_wide_training_control_audit.py",
    )
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

    filtered = inventory_scope._filter_inventory(root, report)

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


def test_exact_path_nontraining_accounting_removes_only_model_only_surfaces(tmp_path: Path) -> None:
    root = tmp_path / "repo-accounting"
    _touch(root, "models/frozen.py", "training/real_trainer.py")
    accounting = root / inventory_scope.ACCOUNTING_PATH
    accounting.parent.mkdir(parents=True, exist_ok=True)
    accounting.write_text(
        json.dumps(
            {
                "schema": inventory_scope.ACCOUNTING_SCHEMA,
                "surfaces": [
                    {
                        "path": "models/frozen.py",
                        "category": "inference_adapter",
                        "reason": "Loads a previously trained model for inference only.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    filtered = inventory_scope._filter_inventory(
        root,
        {
            "training_files": [
                {"path": "models/frozen.py", "model_surface": True, "training_logic": False, "executable": False},
                {"path": "training/real_trainer.py", "training_logic": True},
            ],
            "executable_training_candidates": [],
            "model_surfaces": ["models/frozen.py"],
            "training_logic_surfaces": ["training/real_trainer.py"],
        },
    )
    assert filtered["model_surfaces"] == []
    assert filtered["non_training_surface_accounting_pass"] is True
    assert filtered["non_training_surface_accounting"][0]["path"] == "models/frozen.py"


def test_nontraining_accounting_rejects_training_logic_and_globs(tmp_path: Path) -> None:
    root = tmp_path / "repo-invalid"
    _touch(root, "training/real_trainer.py")
    accounting = root / inventory_scope.ACCOUNTING_PATH
    accounting.parent.mkdir(parents=True, exist_ok=True)
    accounting.write_text(
        json.dumps(
            {
                "schema": inventory_scope.ACCOUNTING_SCHEMA,
                "surfaces": [
                    {
                        "path": "training/real_trainer.py",
                        "category": "inference_adapter",
                        "reason": "This deliberately invalid declaration must fail closed.",
                    },
                    {
                        "path": "models/*.py",
                        "category": "inference_adapter",
                        "reason": "Glob declarations are deliberately forbidden by the contract.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    filtered = inventory_scope._filter_inventory(
        root,
        {
            "training_files": [{"path": "training/real_trainer.py", "training_logic": True}],
            "executable_training_candidates": ["training/real_trainer.py"],
            "model_surfaces": ["training/real_trainer.py"],
            "training_logic_surfaces": ["training/real_trainer.py"],
        },
    )
    assert filtered["non_training_surface_accounting_pass"] is False
    assert len(filtered["non_training_surface_accounting_errors"]) == 2
    assert "cannot be declared non-training" in filtered["non_training_surface_accounting_errors"][0]
    assert "glob/meta characters are forbidden" in filtered["non_training_surface_accounting_errors"][1]
