from __future__ import annotations

import hashlib

import pytest

from evaluation.active_learning import AcquisitionSignals, ActiveLearningCandidate, select_active_learning_batch
from evaluation.active_learning_gold import BinaryLabelMapping, ScoredGoldItem, build_active_learning_gold_manifest, calibration_examples_from_active_learning_gold
from evaluation.expert_adjudication import ExpertAdjudicationStore, GoldLabel, LabelSchema
from orchestration.active_learning_adjudication import ActiveLearningRoute, SQLiteActiveLearningJournal, materialize_active_learning_batch


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def candidates():
    return (
        ActiveLearningCandidate("alice", "support", sha("item-a"), (sha("evidence-a"),), "g1", AcquisitionSignals(uncertainty=1.0)),
        ActiveLearningCandidate("alice", "support", sha("item-b"), (sha("evidence-b"),), "g2", AcquisitionSignals(uncertainty=0.9)),
    )


def materialized(tmp_path):
    rows = candidates()
    batch = select_active_learning_batch(rows)
    store = ExpertAdjudicationStore(tmp_path / "reviews.sqlite3")
    journal = SQLiteActiveLearningJournal(tmp_path / "active.sqlite3")
    receipt = materialize_active_learning_batch(
        batch,
        rows,
        routes={"support": ActiveLearningRoute("support", LabelSchema("support", "1", ("entailed", "neutral", "contradicted")))},
        adjudication_store=store,
        journal=journal,
        now=1.0,
    )
    return receipt


def gold_for(receipt, labels=("entailed", "contradicted")):
    return tuple(
        GoldLabel(
            case_id=row.case_id,
            item_sha256=row.item_sha256,
            round_index=index,
            label=labels[index],
            resolution_revision=index + 5,
            resolution_digest=sha(f"resolution-{index}"),
        )
        for index, row in enumerate(receipt.cases)
    )


def test_gold_manifest_binds_materialization_and_resolution_lineage(tmp_path) -> None:
    receipt = materialized(tmp_path)
    manifest = build_active_learning_gold_manifest(
        owner_id="alice",
        gold_labels=gold_for(receipt),
        materializations=(receipt,),
        label_contract_sha256=sha("label-contract"),
        require_all_materialized_resolved=True,
    )
    assert manifest.materialization_receipt_sha256s == (receipt.receipt_sha256,)
    assert len(manifest.examples) == 2
    assert {row.resolution_digest for row in manifest.examples} == {sha("resolution-0"), sha("resolution-1")}
    assert len(manifest.manifest_sha256) == 64


def test_gold_manifest_rejects_unknown_case_and_incomplete_required_resolution(tmp_path) -> None:
    receipt = materialized(tmp_path)
    unknown = GoldLabel("unknown-case", sha("unknown-item"), 0, "entailed", 1, sha("unknown-resolution"))
    with pytest.raises(ValueError, match="outside"):
        build_active_learning_gold_manifest(
            owner_id="alice",
            gold_labels=(unknown,),
            materializations=(receipt,),
            label_contract_sha256=sha("label-contract"),
        )

    with pytest.raises(ValueError, match="not every"):
        build_active_learning_gold_manifest(
            owner_id="alice",
            gold_labels=gold_for(receipt)[:1],
            materializations=(receipt,),
            label_contract_sha256=sha("label-contract"),
            require_all_materialized_resolved=True,
        )


def test_binary_label_mapping_is_explicit_and_unmapped_labels_are_not_guessed() -> None:
    mapping = BinaryLabelMapping.build(task_id="support", positive_labels=("entailed",), negative_labels=("contradicted",))
    assert mapping.target("entailed") is True
    assert mapping.target("contradicted") is False
    assert mapping.target("neutral") is None
    with pytest.raises(ValueError, match="disjoint"):
        BinaryLabelMapping.build(task_id="support", positive_labels=("same",), negative_labels=("same",))


def test_active_learning_gold_converts_to_calibration_examples_only_with_bound_scores(tmp_path) -> None:
    receipt = materialized(tmp_path)
    manifest = build_active_learning_gold_manifest(
        owner_id="alice",
        gold_labels=gold_for(receipt),
        materializations=(receipt,),
        label_contract_sha256=sha("label-contract"),
    )
    mapping = BinaryLabelMapping.build(task_id="support", positive_labels=("entailed",), negative_labels=("contradicted",))
    scores = (
        ScoredGoldItem(receipt.cases[0].item_sha256, 2.0, 1.0),
        ScoredGoldItem(receipt.cases[1].item_sha256, -1.0, 2.0),
    )
    examples = calibration_examples_from_active_learning_gold(manifest, mapping=mapping, scored_items=scores)
    assert len(examples) == 2
    assert {row.relevant for row in examples} == {True, False}
    assert {row.weight for row in examples} == {1.0, 2.0}


def test_missing_bound_score_fails_closed_by_default(tmp_path) -> None:
    receipt = materialized(tmp_path)
    manifest = build_active_learning_gold_manifest(
        owner_id="alice",
        gold_labels=gold_for(receipt),
        materializations=(receipt,),
        label_contract_sha256=sha("label-contract"),
    )
    mapping = BinaryLabelMapping.build(task_id="support", positive_labels=("entailed",), negative_labels=("contradicted",))
    with pytest.raises(ValueError, match="no bound raw score"):
        calibration_examples_from_active_learning_gold(
            manifest,
            mapping=mapping,
            scored_items=(ScoredGoldItem(receipt.cases[0].item_sha256, 2.0),),
        )
