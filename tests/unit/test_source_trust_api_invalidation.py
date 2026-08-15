from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tools.dependency_invalidation import DependencyInvalidationStore, DependencyRef
from tools.source_trust_api import build_source_trust_router
from tools.source_trust_store import SourceTrustStore


def _app(tmp_path):
    trust = SourceTrustStore(tmp_path / "trust.sqlite3")
    invalidations = DependencyInvalidationStore(tmp_path / "invalidations.sqlite3")

    def principal():
        return SimpleNamespace(owner_id="owner-a")

    app = FastAPI()
    app.include_router(
        build_source_trust_router(
            principal_dependency=principal,
            store=trust,
            invalidation_store=invalidations,
        )
    )
    return TestClient(app), trust, invalidations


def _review(methodology: float, basis: str):
    return {
        "source_id": "paper-1",
        "source_type": "primary_study",
        "status": "active",
        "provenance_integrity": 1.0,
        "methodological_quality": methodology,
        "topical_applicability": 0.8,
        "freshness": 0.7,
        "independent_replication": 0.2,
        "conflicts_of_interest_known": True,
        "notes": ["reviewed"],
        "review_basis": basis,
    }


def test_review_change_invalidates_source_dependent_results(tmp_path):
    client, trust, invalidations = _app(tmp_path)

    first_response = client.post("/research/source-trust", json=_review(0.4, "first review"))
    assert first_response.status_code == 201
    first = first_response.json()
    assert first["invalidation"]["changed"] is True
    assert trust.latest("owner-a", "paper-1").revision_id == first["revision_id"]

    invalidations.register_dependency(
        "owner-a",
        upstream=DependencyRef("source", "paper-1"),
        downstream=DependencyRef("result", "result-1"),
        relation="cites",
    )
    invalidations.register_dependency(
        "owner-a",
        upstream=DependencyRef("result", "result-1"),
        downstream=DependencyRef("report", "report-1"),
        relation="derived_from_result",
    )

    second_response = client.post("/research/source-trust", json=_review(0.9, "second review"))
    assert second_response.status_code == 201
    second = second_response.json()
    assert second["revision_id"] != first["revision_id"]
    assert second["invalidation"]["changed"] is True
    assert second["invalidation"]["affected_artifacts"] == 2

    stale = invalidations.list_stale("owner-a", limit=100)
    assert {item.artifact.key for item in stale} >= {"result:result-1", "report:report-1"}

    history_response = client.get("/research/source-trust/paper-1")
    assert history_response.status_code == 200
    decision = history_response.json()["decision"]
    assert 0.0 <= decision["trust_score"] <= 1.0
    assert len(decision["policy_sha256"]) == 64
    assert isinstance(decision["reasons"], list)


def test_identical_review_is_idempotent_for_invalidation(tmp_path):
    client, _trust, invalidations = _app(tmp_path)
    first = client.post("/research/source-trust", json=_review(0.7, "same review"))
    assert first.status_code == 201
    invalidations.register_dependency(
        "owner-a",
        upstream=DependencyRef("source", "paper-1"),
        downstream=DependencyRef("result", "result-1"),
        relation="cites",
    )

    again = client.post("/research/source-trust", json=_review(0.7, "same review"))
    assert again.status_code == 201
    assert again.json()["revision_id"] == first.json()["revision_id"]
    assert again.json()["invalidation"]["changed"] is False
    assert invalidations.list_stale("owner-a", limit=100) == ()
