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

def test_stateful_class_wrappers_preserve_public_identity_across_reimports():
    result = _run(
        r"""
import importlib
import sys
import tools

# Classic storage.
legacy_storage = importlib.import_module("storage_legacy")
public_storage = importlib.import_module("storage")
storage_base = legacy_storage._boundary_original_StorageManager
storage_public = legacy_storage._boundary_public_StorageManager
for _ in range(3):
    sys.modules.pop("storage", None)
    public_storage = importlib.import_module("storage")
    assert public_storage.StorageManager is storage_public
    assert storage_public.__mro__[1] is storage_base

# Document registry.
legacy_document = importlib.import_module("tools.document_store_legacy")
public_document = importlib.import_module("tools.document_store")
document_base = legacy_document._boundary_original_DocumentStore
document_public = legacy_document._boundary_public_DocumentStore
for _ in range(3):
    sys.modules.pop("tools.document_store", None)
    tools.__dict__.pop("document_store", None)
    public_document = importlib.import_module("tools.document_store")
    assert public_document.DocumentStore is document_public
    assert document_public.__mro__[1] is document_base

# Search agent.
legacy_agent = importlib.import_module("search_agent_legacy")
public_agent = importlib.import_module("search_agent")
agent_base = legacy_agent._boundary_original_SearchAgent
agent_public = legacy_agent._boundary_public_SearchAgent
execution_base = legacy_agent._boundary_original_ToolExecution
execution_public = legacy_agent._boundary_public_ToolExecution
validator = legacy_agent._boundary_original_validate_schema_value
for _ in range(3):
    sys.modules.pop("search_agent", None)
    public_agent = importlib.import_module("search_agent")
    assert public_agent.SearchAgent is agent_public
    assert agent_public.__mro__[1] is agent_base
    assert public_agent.ToolExecution is execution_public
    assert execution_public.__mro__[1] is execution_base
    assert legacy_agent._boundary_original_validate_schema_value is validator

# RAG public class, in addition to base/cache checks from pass fourteen.
legacy_rag = importlib.import_module("tools.rag_legacy")
public_rag = importlib.import_module("tools.rag")
rag_public = legacy_rag._boundary_public_RAGLayer
for _ in range(3):
    sys.modules.pop("tools.rag", None)
    tools.__dict__.pop("rag", None)
    public_rag = importlib.import_module("tools.rag")
    assert public_rag.RAGLayer is rag_public
"""
    )
    assert result.returncode == 0, result.stderr
