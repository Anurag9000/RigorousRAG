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


_ASSERT_PIPELINE = """
names = [item["function"]["name"] for item in MODULE.TOOLS_SCHEMA]
assert names.count("search_uploaded_docs_adaptive") == 1
assert names.count("search_uploaded_docs_multihop") == 1
assert MODULE._agent_tool_registry_bridge_installed is True
assert MODULE._agent_tool_registry_dispatcher_name == "_dispatch"
assert MODULE._adaptive_agent_tool_installed is True
assert MODULE._evidence_graph_agent_tool_installed is True
assert MODULE._multihop_agent_tool_installed is True
assert MODULE._source_status_agent_gate_installed is True
assert MODULE._claim_entailment_agent_gate_installed is True
assert MODULE._evidence_admissibility_agent_gate_installed is True
"""


@pytest.mark.parametrize(
    "source",
    [
        """
import search_agent_legacy as MODULE
""" + _ASSERT_PIPELINE,
        """
import search_agent as MODULE
""" + _ASSERT_PIPELINE,
        """
import importlib
import search_agent_legacy as MODULE
MODULE = importlib.reload(MODULE)
""" + _ASSERT_PIPELINE,
    ],
)
def test_governed_agent_installation_survives_import_orders(source: str) -> None:
    _assert_success(_run_import_script(source))
