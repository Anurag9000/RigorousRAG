from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_import_script(source: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment.pop("OPENAI_BASE_URL", None)
    environment["PYTHONPATH"] = str(_REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def _assert_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize(
    "source",
    [
        """
import search_agent_legacy as legacy
names = [item["function"]["name"] for item in legacy.TOOLS_SCHEMA]
assert names.count("search_uploaded_docs_multihop") == 1
assert legacy._multihop_agent_tool_installed is True
""",
        """
import search_agent
names = [item["function"]["name"] for item in search_agent.TOOLS_SCHEMA]
assert names.count("search_uploaded_docs_multihop") == 1
assert search_agent._multihop_agent_tool_installed is True
""",
        """
import importlib
import search_agent_legacy as legacy
legacy = importlib.reload(legacy)
names = [item["function"]["name"] for item in legacy.TOOLS_SCHEMA]
assert names.count("search_uploaded_docs_multihop") == 1
assert legacy._multihop_agent_tool_installed is True
""",
    ],
)
def test_multihop_agent_installation_survives_import_orders(source: str) -> None:
    _assert_success(_run_import_script(source))
