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
    assert first["invalidation"]["completed"] == 1
    assert trust.latest("owner-a", "paper-1").revision_id == first["revision_id"]
    assert trust.pending_activations("owner-a", source_id="paper-1") == ()

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
    assert second["invalidation"]["failed"] == 0
    assert second["invalidation"]["affected_artifacts"] == 2

    stale = invalidations.list_stale("owner-a", limit=100)
    stale_refs = {(item.artifact.kind, item.artifact.resource_id) for item in stale}
    assert stale_refs >= {("result", "result-1"), ("report", "report-1")}

    history_response = client.get("/research/source-trust/paper-1")
    assert history_response.status_code == 200
    history = history_response.json()
    assert history["active_revision_id"] == second["revision_id"]
    assert all(not item["pending"] for item in history["activations"])
    decision = history["decision"]
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


def test_failed_invalidation_survives_as_pending_activation_and_retries(tmp_path, monkeypatch):
    client, trust, invalidations = _app(tmp_path)
    original_invalidate = invalidations.invalidate

    def fail_invalidation(*args, **kwargs):
        raise RuntimeError("simulated invalidation outage")

    monkeypatch.setattr(invalidations, "invalidate", fail_invalidation)
    failed = client.post("/research/source-trust", json=_review(0.6, "durable review"))
    assert failed.status_code == 503

    current = trust.latest("owner-a", "paper-1")
    assert current is not None
    pending = trust.pending_activations("owner-a", source_id="paper-1")
    assert len(pending) == 1
    assert pending[0].revision_id == current.revision_id
    assert pending[0].last_error == "RuntimeError"

    visible = client.get("/research/source-trust/pending?source_id=paper-1")
    assert visible.status_code == 200
    assert len(visible.json()["pending"]) == 1

    monkeypatch.setattr(invalidations, "invalidate", original_invalidate)
    retried = client.post("/research/source-trust", json=_review(0.6, "durable review"))
    assert retried.status_code == 201
    payload = retried.json()
    assert payload["revision_id"] == current.revision_id
    assert payload["invalidation"]["changed"] is True
    assert payload["invalidation"]["completed"] == 1
    assert payload["invalidation"]["failed"] == 0
    assert trust.pending_activations("owner-a", source_id="paper-1") == ()


def test_explicit_reconcile_drains_owner_scoped_pending_activation(tmp_path, monkeypatch):
    client, trust, invalidations = _app(tmp_path)
    original_invalidate = invalidations.invalidate
    monkeypatch.setattr(
        invalidations,
        "invalidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    assert client.post("/research/source-trust", json=_review(0.5, "review" )).status_code == 503
    assert len(trust.pending_activations("owner-a")) == 1

    monkeypatch.setattr(invalidations, "invalidate", original_invalidate)
    reconciled = client.post(
        "/research/source-trust/reconcile",
        json={"source_id": "paper-1", "limit": 10},
    )
    assert reconciled.status_code == 200
    assert reconciled.json()["completed"] == 1
    assert reconciled.json()["failed"] == 0
    assert trust.pending_activations("owner-a") == ()
