from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller_console as console
import universal_training_controller_subcommands as subcommands


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    package = root / "src" / "fixturepkg"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        """[project]\nname='fixture'\n[project.scripts]\n"
        "fixture-train='fixturepkg.train:main'\n"
        "fixture-pretrain='fixturepkg.pretrain:main'\n"
        "fixture-predict='fixturepkg.predict:main'\n"
        """,
        encoding="utf-8",
    )
    (package / "train.py").write_text(
        "import argparse, torch\n"
        "def main():\n"
        " p=argparse.ArgumentParser(); p.add_argument('--data', required=True); p.parse_args()\n"
        " model=torch.nn.Linear(1,1); opt=torch.optim.SGD(model.parameters(), lr=.1)\n",
        encoding="utf-8",
    )
    (package / "pretrain.py").write_text(
        "import argparse\n"
        "def main():\n"
        " p=argparse.ArgumentParser(); p.add_argument('--epochs', type=int, default=2); p.parse_args()\n",
        encoding="utf-8",
    )
    (package / "predict.py").write_text("def main(): return 0\n", encoding="utf-8")
    return root


def _subcommand_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "repo-subcommands"
    package = root / "src" / "fixturepkg"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        """[project]\nname='fixture-subcommands'\n[project.scripts]\n"
        "fixture-training='fixturepkg.training_cli:main'\n"
        """,
        encoding="utf-8",
    )
    (package / "training_cli.py").write_text(
        "import argparse\n"
        "def main():\n"
        " p=argparse.ArgumentParser(); subs=p.add_subparsers(dest='command', required=True)\n"
        " train=subs.add_parser('train'); train.add_argument('--config', required=True)\n"
        " check=subs.add_parser('verify'); check.add_argument('--checkpoint', required=True)\n"
        " p.parse_args()\n",
        encoding="utf-8",
    )
    return root


def test_registry_finds_required_training_arguments(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    report = console._inventory(root, {"auto_console_training_jobs": True})
    assert report["registered_script_count"] == 3
    assert report["registered_training_script_count"] == 2
    assert report["unconfigured_training_entrypoints"] == ["fixture-train"]
    train = next(x for x in report["training_entries"] if x["name"] == "fixture-train")
    assert train["required_options"] == ["--data"]


def test_registry_materializes_only_proven_commands(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    profile = {
        "auto_console_training_jobs": True,
        "console_script_args": {"fixture-train": ["--data", "data/processed"]},
    }
    jobs = console._jobs(root, profile)
    assert {job["console_script"] for job in jobs} == {"fixture-train", "fixture-pretrain"}
    assert all(str(job["entrypoint_source"]).startswith("src/fixturepkg/") for job in jobs)
    train = next(job for job in jobs if job["console_script"] == "fixture-train")
    assert train["command"][-2:] == ["--data", "data/processed"]


def test_training_subcommand_materializes_once_and_delegates_parent(tmp_path: Path) -> None:
    root = _subcommand_fixture(tmp_path)
    profile = {
        "auto_console_subcommand_jobs": True,
        "console_subcommand_args": {
            "fixture-training:train": ["--config", "config/train.json"],
        },
    }
    jobs = subcommands._jobs(root, profile)
    assert len(jobs) == 1
    assert jobs[0]["console_script"] == "fixture-training"
    assert jobs[0]["console_subcommand"] == "train"
    assert jobs[0]["command"][-3:] == ["train", "--config", "config/train.json"]

    delegated = subcommands._delegated_console_scripts(jobs)
    assert delegated == {"fixture-training": ["train"]}

    # Mimic the lower console report after its temporary internal ignore.  The
    # normalization must restore the parent as an auditable, non-exempted entry
    # and mark it satisfied by the one concrete child job.
    report = {
        "coverage_ok": True,
        "console_registry": {
            "entries": [
                {
                    "name": "fixture-training",
                    "training_surface": True,
                    "ignored": True,
                    "configured": True,
                }
            ],
            "training_entries": [],
            "unresolved_targets": [],
            "unconfigured_training_entrypoints": [],
        },
        "compiled_console_training_jobs": [],
        "missing_console_job_materialization": [],
        "unscheduled_registered_training_entrypoints": [],
        "strict_registered_training_entrypoints_pass": True,
        "strict_controls": {},
    }
    subcommands._normalize_delegated_parent_report(report, delegated)
    parent = report["console_registry"]["training_entries"][0]
    assert parent["ignored"] is False
    assert parent["satisfied_by_subcommand"] is True
    assert parent["satisfied_by_subcommands"] == ["train"]
    assert report["console_entrypoints_satisfied_by_subcommands"] == {"fixture-training": ["train"]}
    assert report["unscheduled_registered_training_entrypoints"] == []
    assert report["strict_registered_training_entrypoints_pass"] is True
