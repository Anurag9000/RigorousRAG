import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run(script: str):
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_integrity_layers_preserve_original_call_chains_across_reimports():
    result = _run(
        r"""
import importlib
import sys
import tools

legacy = importlib.import_module("tools.integrity_legacy")
public = importlib.import_module("tools.integrity")
boundary_original = legacy._integrity_boundary_original_compare_papers
final_original = legacy._integrity_final_original_compare_papers
for _ in range(3):
    for name in ("tools.integrity", "tools.integrity_boundary"):
        sys.modules.pop(name, None)
        tools.__dict__.pop(name.rsplit(".", 1)[-1], None)
    public = importlib.import_module("tools.integrity")
    assert legacy._integrity_boundary_original_compare_papers is boundary_original
    assert legacy._integrity_final_original_compare_papers is final_original
    assert public.compare_papers.__globals__["_original_compare_papers"] is final_original
assert boundary_original.__module__ == "tools.integrity_legacy"
"""
    )
    assert result.returncode == 0, result.stderr


def test_rag_reimports_do_not_stack_wrappers_or_replace_singleton_state():
    result = _run(
        r"""
import importlib
import sys
import tools

legacy = importlib.import_module("tools.rag_legacy")
public = importlib.import_module("tools.rag")
base = legacy._boundary_original_RAGLayer
instances = legacy._boundary_rag_instances
lock = legacy._boundary_rag_lock
for _ in range(3):
    sys.modules.pop("tools.rag", None)
    tools.__dict__.pop("rag", None)
    public = importlib.import_module("tools.rag")
    assert legacy._boundary_original_RAGLayer is base
    assert public.RAGLayer.__mro__[1] is base
    assert public._RAG_INSTANCES is instances
    assert public._RAG_LOCK is lock
"""
    )
    assert result.returncode == 0, result.stderr
