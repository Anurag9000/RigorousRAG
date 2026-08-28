from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller_console as console


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
