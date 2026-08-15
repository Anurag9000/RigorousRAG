import pytest

from tools.evidence_graph_external_backend import (
    ExternalGraphEdge,
    ExternalGraphNode,
    Neo4jEvidenceGraphBackend,
)


class FakeTx:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))


class FakeSession:
    def __init__(self, tx):
        self.tx = tx
        self.write_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute_write(self, callback):
        self.write_calls += 1
        callback(self.tx)


class FakeDriver:
    def __init__(self):
        self.tx = FakeTx()
        self.session_kwargs = None
        self.session_obj = FakeSession(self.tx)

    def session(self, **kwargs):
        self.session_kwargs = kwargs
        return self.session_obj


def _graph():
    nodes = (
        ExternalGraphNode("n1", "claim", "Alpha", {"source": "doc-a"}),
        ExternalGraphNode("n2", "claim", "Beta", {"source": "doc-b"}),
    )
    edges = (ExternalGraphEdge("e1", "n1", "n2", "supports", {"score": 0.8}),)
    return nodes, edges


def test_neo4j_backend_is_injected_parameterized_and_owner_scoped():
    driver = FakeDriver()
    backend = Neo4jEvidenceGraphBackend(driver, database="evidence")
    nodes, edges = _graph()

    result = backend.upsert_generation(
        owner_id="tenant-a",
        graph_id="graph-1",
        generation="g7",
        nodes=nodes,
        edges=edges,
        activate=True,
    )

    assert result.owner_id == "tenant-a"
    assert result.node_count == 2
    assert result.edge_count == 1
    assert len(result.digest) == 64
    assert result.activated is True
    assert driver.session_kwargs == {"database": "evidence"}
    assert driver.session_obj.write_calls == 1
    assert len(driver.tx.calls) == 4
    for query, params in driver.tx.calls:
        assert "$owner_id" in query
        assert params["owner_id"] == "tenant-a"
        assert params["graph_id"] == "graph-1"
        assert params["generation"] == "g7"
        assert "tenant-a" not in query


def test_external_graph_digest_is_deterministic_across_input_order():
    driver_a = FakeDriver()
    driver_b = FakeDriver()
    backend_a = Neo4jEvidenceGraphBackend(driver_a)
    backend_b = Neo4jEvidenceGraphBackend(driver_b)
    nodes, edges = _graph()

    a = backend_a.upsert_generation(
        owner_id="o", graph_id="g", generation="1", nodes=nodes, edges=edges
    )
    b = backend_b.upsert_generation(
        owner_id="o", graph_id="g", generation="1", nodes=tuple(reversed(nodes)), edges=edges
    )
    assert a.digest == b.digest


def test_external_graph_rejects_dangling_edges_before_driver_use():
    driver = FakeDriver()
    backend = Neo4jEvidenceGraphBackend(driver)
    nodes, _ = _graph()
    bad = (ExternalGraphEdge("e1", "n1", "missing", "supports", {}),)

    with pytest.raises(ValueError, match="missing node"):
        backend.upsert_generation(
            owner_id="o", graph_id="g", generation="1", nodes=nodes, edges=bad
        )
    assert driver.session_kwargs is None


def test_external_graph_rejects_duplicate_ids_and_invalid_driver():
    with pytest.raises(TypeError):
        Neo4jEvidenceGraphBackend(object())
    driver = FakeDriver()
    backend = Neo4jEvidenceGraphBackend(driver)
    node = ExternalGraphNode("same", "claim", "x", {})
    with pytest.raises(ValueError, match="duplicate node_id"):
        backend.upsert_generation(
            owner_id="o", graph_id="g", generation="1", nodes=(node, node), edges=()
        )
