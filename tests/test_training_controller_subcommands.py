from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller_subcommands as subcommands


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    package = root / "src" / "fixturepkg"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        """[project]\nname='fixture'\n[project.scripts]\nfixture='fixturepkg.cli:main'\n""",
        encoding="utf-8",
    )
    (package / "cli.py").write_text(
        "import argparse\n"
        "def main():\n"
        " p=argparse.ArgumentParser(); s=p.add_subparsers(dest='command', required=True)\n"
        " train=s.add_parser('train-vision'); train.add_argument('--manifest', required=True); train.add_argument('--epochs', type=int, default=2)\n"
        " fit=s.add_parser('fit-head'); fit.add_argument('--features', required=True); fit.add_argument('--output', required=True)\n"
        " pred=s.add_parser('predict'); pred.add_argument('--checkpoint', required=True)\n"
        " p.parse_args()\n",
        encoding="utf-8",
    )
    return root


def test_subcommand_registry_finds_training_contracts(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    report = subcommands._inventory(root, {})
    assert report["registered_training_subcommand_count"] == 2
    assert report["unconfigured_training_subcommands"] == ["fixture:fit-head", "fixture:train-vision"]
    by_key = {item["key"]: item for item in report["training_subcommands"]}
    assert by_key["fixture:train-vision"]["required_options"] == ["--manifest"]
    assert by_key["fixture:fit-head"]["required_options"] == ["--features", "--output"]
    assert "fixture:predict" not in by_key


def test_subcommand_jobs_use_explicit_and_safe_global_defaults(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    profile = {
        "auto_console_subcommand_jobs": True,
        "console_subcommand_args": {
            "fixture:fit-head": ["--features", "features.csv", "--output", "artifacts/head"],
        },
        "console_argument_defaults": {"--manifest": "manifest.csv"},
    }
    jobs = subcommands._jobs(root, profile)
    assert {job["console_subcommand"] for job in jobs} == {"fit-head", "train-vision"}
    vision = next(job for job in jobs if job["console_subcommand"] == "train-vision")
    assert vision["command"][-2:] == ["--manifest", "manifest.csv"]
    head = next(job for job in jobs if job["console_subcommand"] == "fit-head")
    assert head["command"][-4:] == ["--features", "features.csv", "--output", "artifacts/head"]
