from __future__ import annotations

import hashlib

import pytest

from evaluation.active_learning import AcquisitionSignals, ActiveLearningCandidate, ActiveLearningPolicy, select_active_learning_batch
from evaluation.expert_adjudication import AdjudicationPolicy, ExpertAdjudicationStore, LabelSchema
from orchestration.active_learning_adjudication import ActiveLearningRoute, SQLiteActiveLearningJournal, materialize_active_learning_batch


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def candidate(
    name: str,
    *,
    owner: str = "alice",
    task: str = "support",
    group: str = "g1",
    uncertainty: float = 0.5,
    disagreement: float = 0.0,
    abstained: bool = False,
    cost: float = 1.0,
    impact: float = 0.0,
):
    return ActiveLearningCandidate(
        owner_id=owner,
        task_id=task,
        item_sha256=sha(f"item:{name}"),
        evidence_sha256s=(sha(f"evidence:{name}"),),
        group_id=group,
        signals=AcquisitionSignals(
            uncertainty=uncertainty,
            disagreement=disagreement,
            expected_impact=impact,
            abstained=abstained,
        ),
        estimated_label_cost=cost,
        source_model_sha256=sha("model"),
    )


def policy(**overrides):
    values = dict(
        max_items=10,
        max_total_cost=10.0,
        max_per_group=2,
        max_per_task=10,
        uncertainty_weight=1.0,
        disagreement_weight=1.0,
        impact_weight=1.0,
        abstention_bonus=1.0,
        drift_weight=0.0,
        novelty_weight=0.0,
        cost_exponent=1.0,
    )
    values.update(overrides)
    return ActiveLearningPolicy(**values)


def test_selection_prioritizes_high_information_per_cost_and_is_deterministic() -> None:
    rows = (
        candidate("high", uncertainty=1.0, disagreement=1.0, impact=1.0, cost=1.0),
        candidate("medium", uncertainty=0.7, disagreement=0.2, impact=0.2, cost=1.0),
        candidate("expensive", uncertainty=1.0, disagreement=1.0, impact=1.0, cost=10.0),
    )
    first = select_active_learning_batch(rows, policy=policy(max_items=2, max_total_cost=2.0))
    second = select_active_learning_batch(rows, policy=policy(max_items=2, max_total_cost=2.0))
    assert first.batch_sha256 == second.batch_sha256
    assert first.selected[0].item_sha256 == sha("item:high")
    assert sha("item:expensive") not in {row.item_sha256 for row in first.selected}


def test_selection_enforces_group_caps_and_blocked_item_deduplication() -> None:
    rows = (
        candidate("a", group="same", uncertainty=1.0),
        candidate("b", group="same", uncertainty=0.9),
        candidate("c", group="other", uncertainty=0.8),
    )
    batch = select_active_learning_batch(
        rows,
        policy=policy(max_per_group=1),
        blocked_item_keys=(("support", sha("item:c")),),
    )
    assert len(batch.selected) == 1
    assert batch.selected[0].group_id == "same"


def test_selection_rejects_cross_owner_or_duplicate_task_item_pool() -> None:
    with pytest.raises(ValueError, match="exactly one owner"):
        select_active_learning_batch((candidate("a", owner="alice"), candidate("b", owner="bob")))
    duplicate = candidate("same")
    with pytest.raises(ValueError, match="duplicate task/item"):
        select_active_learning_batch((duplicate, duplicate))


def route(task: str = "support") -> ActiveLearningRoute:
    return ActiveLearningRoute(
        task,
        LabelSchema(task, "1", ("entailed", "neutral", "contradicted")),
        AdjudicationPolicy(minimum_independent_reviews=2),
    )


def test_materialization_reuses_existing_adjudication_store_and_is_idempotent(tmp_path) -> None:
    rows = (candidate("a", uncertainty=1.0), candidate("b", uncertainty=0.9))
    batch = select_active_learning_batch(rows, policy=policy(max_items=2))
    adjudication = ExpertAdjudicationStore(tmp_path / "reviews.sqlite3")
    journal = SQLiteActiveLearningJournal(tmp_path / "active.sqlite3")

    first = materialize_active_learning_batch(
        batch,
        rows,
        routes={"support": route()},
        adjudication_store=adjudication,
        journal=journal,
        now=1.0,
    )
    second = materialize_active_learning_batch(
        batch,
        rows,
        routes={"support": route()},
        adjudication_store=adjudication,
        journal=journal,
        now=2.0,
    )
    assert first.receipt_sha256 == second.receipt_sha256
    assert tuple(row.case_id for row in first.cases) == tuple(row.case_id for row in second.cases)
    assert set(journal.blocked_item_keys(owner_id="alice")) == {
        ("support", sha("item:a")),
        ("support", sha("item:b")),
    }


def test_materialization_requires_route_and_preserves_owner_scope(tmp_path) -> None:
    rows = (candidate("a"),)
    batch = select_active_learning_batch(rows)
    adjudication = ExpertAdjudicationStore(tmp_path / "reviews.sqlite3")
    journal = SQLiteActiveLearningJournal(tmp_path / "active.sqlite3")
    with pytest.raises(ValueError, match="no adjudication route"):
        materialize_active_learning_batch(
            batch,
            rows,
            routes={},
            adjudication_store=adjudication,
            journal=journal,
            now=1.0,
        )

    wrong_owner_rows = (candidate("a", owner="bob"),)
    with pytest.raises(ValueError, match="candidate owner differs"):
        materialize_active_learning_batch(
            batch,
            wrong_owner_rows,
            routes={"support": route()},
            adjudication_store=adjudication,
            journal=journal,
            now=1.0,
        )


def test_materialization_receipt_binds_route_schema_and_case_policy(tmp_path) -> None:
    rows = (candidate("a"),)
    batch = select_active_learning_batch(rows)
    adjudication = ExpertAdjudicationStore(tmp_path / "reviews.sqlite3")
    journal = SQLiteActiveLearningJournal(tmp_path / "active.sqlite3")
    selected_route = route()
    receipt = materialize_active_learning_batch(
        batch,
        rows,
        routes={"support": selected_route},
        adjudication_store=adjudication,
        journal=journal,
        now=1.0,
    )
    assert receipt.cases[0].route_sha256 == selected_route.route_sha256
    assert len(receipt.receipt_sha256) == 64
