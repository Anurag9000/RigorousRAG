import importlib
import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


@dataclass
class Citation:
    label: str
    title: str
    url: str
    source_type: str
    snippet: str | None = None
    quote: str | None = None
    source_id: str | None = None
    doc_id: str | None = None
    chunk_id: str | None = None
    page_number: int | None = None
    metadata: dict = field(default_factory=dict)


def load_module(monkeypatch, rag):
    models = ModuleType("tools.models")
    models.Citation = Citation
    rag_module = ModuleType("tools.rag")
    rag_module.get_rag_layer = lambda: rag
    security = ModuleType("tools.security")
    security.normalize_owner_id = lambda value: value.strip() if isinstance(value, str) and value.strip() else (_ for _ in ()).throw(ValueError("owner"))
    monkeypatch.setitem(sys.modules, "tools.models", models)
    monkeypatch.setitem(sys.modules, "tools.rag", rag_module)
    monkeypatch.setitem(sys.modules, "tools.security", security)
    sys.modules.pop("tools.rag_tool", None)
    return importlib.import_module("tools.rag_tool")


def chunk(identifier, text, score, owner="alice", doc="doc-1", **metadata):
    return SimpleNamespace(id=identifier, text=text, score=score, metadata={"owner_id": owner, "doc_id": doc, **metadata})


def test_dense_default_preserves_backend_limit_and_order(monkeypatch):
    rag = MagicMock()
    rag.query.return_value = [chunk("a", "first", 0.2), chunk("b", "second", 0.9)]
    module = load_module(monkeypatch, rag)
    citations = module.search_uploaded_docs("question", owner_id="alice", n_results=2)
    assert [item.chunk_id for item in citations] == ["a", "b"]
    assert rag.query.call_args.kwargs["n_results"] == 2
    assert citations[0].metadata["relevance"] == 0.2
    assert citations[0].metadata["retrieval_mode"] == "dense"


def test_hybrid_uses_pool_and_lexical_signal(monkeypatch):
    rag = MagicMock()
    rag.query.return_value = [chunk("dense", "unrelated", 1.0), chunk("target", "target target evidence", 0.1)]
    module = load_module(monkeypatch, rag)
    citations = module.search_uploaded_docs(
        "target", owner_id="alice", n_results=1, retrieval_mode="hybrid",
        reranker="heuristic", candidate_pool=20,
    )
    assert rag.query.call_args.kwargs["n_results"] == 20
    assert citations[0].chunk_id == "target"
    assert citations[0].metadata["lexical_score"] > 0.0
    assert citations[0].metadata["relevance"] == 0.1


def test_cross_owner_and_hostile_metadata_never_influence_ranking(monkeypatch):
    class BrokenMetadata(dict):
        def get(self, *_args, **_kwargs):
            raise RuntimeError("secret")

    rag = MagicMock()
    rag.query.return_value = [
        chunk("wrong", "target", 1.0, owner="bob"),
        SimpleNamespace(id="broken", text="target", score=1.0, metadata=BrokenMetadata()),
        chunk("good", "target", 0.1),
    ]
    module = load_module(monkeypatch, rag)
    citations = module.search_uploaded_docs("target", owner_id="alice", retrieval_mode="hybrid")
    assert [item.chunk_id for item in citations] == ["good"]


def test_direct_controls_are_strict(monkeypatch):
    rag = MagicMock()
    module = load_module(monkeypatch, rag)
    for kwargs in (
        {"retrieval_mode": "bad"}, {"reranker": "bad"}, {"candidate_pool": True},
        {"diversity_lambda": float("nan")}, {"diversity_lambda": 2.0},
    ):
        with pytest.raises(ValueError):
            module.search_uploaded_docs("q", owner_id="alice", **kwargs)
    rag.query.assert_not_called()
