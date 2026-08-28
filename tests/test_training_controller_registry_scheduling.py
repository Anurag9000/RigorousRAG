from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller_current as current
import universal_training_controller_registry_scheduling as scheduling


def _inner_report(_root, profile, _jobs):
    # The v11 wrapper must relax only the two legacy scheduling flags before
    # asking the v10 stack to evaluate every other strict invariant.
    assert profile["require_registered_training_scheduling"] is False
    assert profile["require_registered_training_subcommand_scheduling"] is False
    return {
        "coverage_ok": True,
        "strict_controls": {},
        "console_registry": {
            "unresolved_targets": [],
            "unconfigured_training_entrypoints": [],
            "training_entries": [
                {
                    "name": "fixture-train",
                    "source": "src/pkg/train.py",
                    "configured": True,
                    "ignored": False,
                }
            ],
        },
        "console_subcommand_registry": {
            "unresolved_targets": [],
            "unconfigured_training_subcommands": [],
            "training_subcommands": [
                {
                    "key": "fixture:train-vision",
                    "source": "src/pkg/cli.py",
                    "configured": True,
                    "ignored": False,
                }
            ],
        },
    }


def test_curated_jobs_satisfy_strict_registry_scheduling(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(current, "_enhanced_coverage_report", _inner_report)
    scheduling.install()
    jobs = [
        {"id": "curated-package", "console_script": "fixture-train"},
        {
            "id": "curated-fold-0",
            "console_script": "fixture",
            "console_subcommand": "train-vision",
        },
    ]
    report = current._enhanced_coverage_report(
        tmp_path,
        {
            "require_registered_training_entrypoints": True,
            "require_registered_training_subcommands": True,
            "require_registered_training_scheduling": True,
            "require_registered_training_subcommand_scheduling": True,
        },
        jobs,
    )
    assert report["coverage_ok"] is True
    assert report["missing_console_job_materialization"] == []
    assert report["missing_console_subcommand_job_materialization"] == []
    assert report["compiled_console_training_jobs"] == ["fixture-train"]
    assert report["compiled_console_training_subcommands"] == ["fixture:train-vision"]


def test_missing_curated_registered_surface_still_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(current, "_enhanced_coverage_report", _inner_report)
    scheduling.install()
    report = current._enhanced_coverage_report(
        tmp_path,
        {
            "require_registered_training_entrypoints": True,
            "require_registered_training_subcommands": True,
            "require_registered_training_scheduling": True,
            "require_registered_training_subcommand_scheduling": True,
        },
        [],
    )
    assert report["coverage_ok"] is False
    assert report["missing_console_job_materialization"] == ["fixture-train"]
    assert report["missing_console_subcommand_job_materialization"] == [
        "fixture:train-vision"
    ]
