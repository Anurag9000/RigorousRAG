from __future__ import annotations

import hashlib

import pytest

from evaluation.active_learning import AcquisitionSignals, ActiveLearningCandidate, ActiveLearningPolicy
from evaluation.expert_adjudication import ExpertAdjudicationStore, LabelSchema
from orchestration.active_learning_adjudication import ActiveLearningRoute, SQLiteActiveLearningJournal
from orchestration.active_learning_cycle import ActiveLearningCandidatePage, ActiveLearningCycleSpec, run_active_learning_cycle


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def candidate(name: str, *, owner: str = "alice") -> ActiveLearningCandidate:
    return ActiveLearningCandidate(
        owner,
        "support",
        sha(f"item:{name}"),
        (sha(f"evidence:{name}"),),
        f"group:{name}",
        AcquisitionSignals(uncertainty=1.0),
    )


class Provider:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def fetch_candidates(self, *, owner_id, cursor, limit):
        self.calls.append((owner_id, cursor, limit))
        return self.pages[cursor]


def route():
    return ActiveLearningRoute("support", LabelSchema("support", "1", ("entailed", "neutral", "contradicted")))


def cycle_spec():
    return ActiveLearningCycleSpec(
        owner_id="alice",
        candidate_source_sha256=sha("candidate-source"),
        policy=ActiveLearningPolicy(max_items=10, max_total_cost=10.0),
        page_limit=50,
    )


def stores(tmp_path):
    return ExpertAdjudicationStore(tmp_path / "reviews.sqlite3"), SQLiteActiveLearningJournal(tmp_path / "active.sqlite3")


def test_cycle_materializes_page_and_returns_opaque_continuation(tmp_path) -> None:
    first_page = ActiveLearningCandidatePage(sha("snapshot"), (candidate("a"), candidate("b")), next_cursor="page-2")
    provider = Provider({None: first_page})
    adjudication, journal = stores(tmp_path)
    result = run_active_learning_cycle(
        cycle_spec(),
        provider=provider,
        routes={"support": route()},
        adjudication_store=adjudication,
        journal=journal,
        now=1.0,
    )
    assert result.next_cursor == "page-2"
    assert result.batch is not None
    assert result.materialization is not None
    assert result.receipt.candidate_count == 2
    assert result.receipt.selected_count == 2
    assert provider.calls == [("alice", None, 50)]


def test_empty_page_is_explicit_noop_without_synthetic_batch(tmp_path) -> None:
    provider = Provider({None: ActiveLearningCandidatePage(sha("snapshot"), (), next_cursor=None)})
    adjudication, journal = stores(tmp_path)
    result = run_active_learning_cycle(
        cycle_spec(),
        provider=provider,
        routes={"support": route()},
        adjudication_store=adjudication,
        journal=journal,
        now=1.0,
    )
    assert result.batch is None
    assert result.materialization is None
    assert result.receipt.candidate_count == 0
    assert result.receipt.batch_sha256 is None


def test_cycle_automatically_blocks_items_materialized_on_prior_page(tmp_path) -> None:
    repeated = candidate("repeat")
    pages = {
        None: ActiveLearningCandidatePage(sha("snapshot-1"), (repeated,), next_cursor="next"),
        "next": ActiveLearningCandidatePage(sha("snapshot-2"), (repeated, candidate("new")), next_cursor=None),
    }
    provider = Provider(pages)
    adjudication, journal = stores(tmp_path)
    first = run_active_learning_cycle(
        cycle_spec(), provider=provider, routes={"support": route()}, adjudication_store=adjudication, journal=journal, now=1.0
    )
    second = run_active_learning_cycle(
        cycle_spec(), provider=provider, routes={"support": route()}, adjudication_store=adjudication, journal=journal, cursor=first.next_cursor, now=2.0
    )
    assert second.batch is not None
    assert [row.item_sha256 for row in second.batch.selected] == [sha("item:new")]


def test_cycle_rejects_provider_cross_owner_candidate(tmp_path) -> None:
    provider = Provider({None: ActiveLearningCandidatePage(sha("snapshot"), (candidate("x", owner="bob"),))})
    adjudication, journal = stores(tmp_path)
    with pytest.raises(ValueError, match="another owner's"):
        run_active_learning_cycle(
            cycle_spec(), provider=provider, routes={"support": route()}, adjudication_store=adjudication, journal=journal, now=1.0
        )


def test_cycle_receipt_binds_input_cursor_without_storing_it_as_plaintext() -> None:
    page = ActiveLearningCandidatePage(sha("snapshot"), (), next_cursor="next-secret-looking-token")
    provider = Provider({"opaque-cursor": page})
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as root:
        adjudication = ExpertAdjudicationStore(f"{root}/reviews.sqlite3")
        journal = SQLiteActiveLearningJournal(f"{root}/active.sqlite3")
        result = run_active_learning_cycle(
            cycle_spec(), provider=provider, routes={"support": route()}, adjudication_store=adjudication, journal=journal, cursor="opaque-cursor", now=1.0
        )
    assert len(result.receipt.input_cursor_sha256) == 64
    assert result.receipt.input_cursor_sha256 != "opaque-cursor"
    assert result.receipt.next_cursor_sha256 != "next-secret-looking-token"
