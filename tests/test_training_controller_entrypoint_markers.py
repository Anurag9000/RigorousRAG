from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller_entrypoint_markers as markers


def test_console_entrypoint_marker_is_reachable_but_not_visible_to_cli(tmp_path: Path) -> None:
    source = tmp_path / "training" / "cli.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# authoritative training source\n", encoding="utf-8")
    code = "import json,sys;print(json.dumps(sys.argv))"
    job = {
        "id": "example:train",
        "console_subcommand": True,
        "entrypoint_source": "training/cli.py",
        "command": [sys.executable, "-c", code, "--config", "recipe.json"],
    }

    marked = markers._mark_job(tmp_path, job)

    assert marked["entrypoint_marker_schema"] == markers.MARKER_SCHEMA
    assert marked["command"][3] == "training/cli.py"
    assert marked["command"][2].startswith("import sys;sys.argv.pop(1);")
    assert "training/cli.py" in marked["command"]

    completed = subprocess.run(
        marked["command"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    argv = json.loads(completed.stdout)
    assert argv == ["-c", "--config", "recipe.json"]


def test_console_entrypoint_marker_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "training" / "cli.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# source\n", encoding="utf-8")
    job = {
        "console_subcommand": True,
        "entrypoint_source": "training/cli.py",
        "command": [sys.executable, "-c", "print('ok')"],
    }
    first = markers._mark_job(tmp_path, job)
    second = markers._mark_job(tmp_path, first)
    assert second["command"] == first["command"]


def test_marker_refuses_missing_or_non_console_sources(tmp_path: Path) -> None:
    plain = {
        "console_subcommand": False,
        "entrypoint_source": "training/missing.py",
        "command": [sys.executable, "-c", "print('ok')"],
    }
    assert markers._mark_job(tmp_path, plain) == plain
    console_missing = {**plain, "console_subcommand": True}
    assert markers._mark_job(tmp_path, console_missing) == console_missing
