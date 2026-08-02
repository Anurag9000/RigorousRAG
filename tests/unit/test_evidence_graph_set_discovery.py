from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tools import evidence_graph_set_discovery as discovery


def value(key="review", *, owner="alice"):
    return SimpleNamespace(
        owner_id=owner,
        graph_set_key=key,
        graph_set_id=("1" if key == "review" else "2") * 64,
        graph_set_digest=("3" if key == "review" else "4") * 64,
        members=(object(), object()),
        edges=(object(),),
        created_at=5.0,
    )


class PublicStore:
    def __init__(self, values):
        self.values = values

    def list_current(self, *, owner_id, limit):
        return self.values[:limit]


class Connection:
    def __init__(self, rows):
        self.rows = rows
        self.args = None

    def execute(self, query, params):
        self.args = (query, params)
        return self

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class PrivateStore:
    def __init__(self, rows):
        self.connection = Connection(rows)
        self._lock = nullcontext()

    def _connect(self):
        return self.connection

    @staticmethod
    def _value(row):
        return row["value"]


def report(current=True):
    return SimpleNamespace(
        authoritative_current=current,
        authority_digest=("a" if current else "b") * 64,
        stale_member_doc_ids=() if current else ("doc-a",),
        missing_member_doc_ids=(),
    )


def test_schema_is_closed_and_owner_is_not_caller_controlled():
    function = discovery.LIST_EVIDENCE_GRAPH_SETS_TOOL_DEF["function"]
    assert function["name"] == "list_evidence_graph_sets"
    assert function["parameters"]["additionalProperties"] is False
    assert "owner_id" not in function["parameters"]["properties"]
    assert function["parameters"]["properties"]["limit"]["maximum"] == 50


def test_public_store_path_filters_stale_and_returns_no_text(monkeypatch):
    values = (value("zeta"), value("review"))
    monkeypatch.setattr(
        discovery,
        "assess_graph_set_authority",
        lambda current, **kwargs: report(
            current.graph_set_key == "review"
        ),
    )
    result = discovery.list_evidence_graph_sets(
        owner_id="alice",
        set_store=PublicStore(values),
        generations=object(),
        graphs=object(),
    )
    assert [item["graph_set_key"] for item in result] == ["review"]
    assert result[0]["authoritative_current"] is True
    assert result[0]["member_count"] == 2
    assert result[0]["edge_count"] == 1
    assert "text" not in result[0]
    assert "owner_id" not in result[0]


def test_include_unavailable_reports_counts_without_document_ids(monkeypatch):
    monkeypatch.setattr(
        discovery,
        "assess_graph_set_authority",
        lambda *args, **kwargs: report(False),
    )
    result = discovery.list_evidence_graph_sets(
        owner_id="alice",
        include_unavailable=True,
        set_store=PublicStore((value(),)),
        generations=object(),
        graphs=object(),
    )
    assert result[0]["authoritative_current"] is False
    assert result[0]["stale_member_count"] == 1
    assert "stale_member_doc_ids" not in result[0]


def test_private_store_path_revalidates_pointer_identity(monkeypatch):
    current = value()
    row = {
        "value": current,
        "pointer_graph_set_key": current.graph_set_key,
        "pointer_graph_set_id": current.graph_set_id,
        "pointer_graph_set_digest": current.graph_set_digest,
        "pointer_schema_version": 1,
    }
    store = PrivateStore([row])
    monkeypatch.setattr(
        discovery,
        "assess_graph_set_authority",
        lambda *args, **kwargs: report(True),
    )
    result = discovery.list_evidence_graph_sets(
        owner_id="alice",
        set_store=store,
        generations=object(),
        graphs=object(),
    )
    assert result[0]["graph_set_key"] == "review"
    assert store.connection.args[1] == ("alice", 20)

    for bad in (
        dict(row, pointer_graph_set_key="other"),
        dict(row, pointer_graph_set_digest="f" * 64),
    ):
        with pytest.raises(RuntimeError, match="pointer identity"):
            discovery.list_evidence_graph_sets(
                owner_id="alice",
                set_store=PrivateStore([bad]),
                generations=object(),
                graphs=object(),
            )


def test_owner_scope_and_invalid_bounds_fail_closed(monkeypatch):
    monkeypatch.setattr(
        discovery,
        "assess_graph_set_authority",
        lambda *args, **kwargs: report(True),
    )
    with pytest.raises(RuntimeError, match="owner scope"):
        discovery.list_evidence_graph_sets(
            owner_id="alice",
            set_store=PublicStore((value(owner="bob"),)),
            generations=object(),
            graphs=object(),
        )
    with pytest.raises(ValueError, match="limit"):
        discovery.list_evidence_graph_sets(
            owner_id="alice",
            limit=51,
            set_store=PublicStore(()),
            generations=object(),
            graphs=object(),
        )
    with pytest.raises(ValueError, match="include_unavailable"):
        discovery.list_evidence_graph_sets(
            owner_id="alice",
            include_unavailable=1,
            set_store=PublicStore(()),
            generations=object(),
            graphs=object(),
        )
