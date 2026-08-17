from __future__ import annotations

import hashlib

import pytest

from evaluation.expert_adjudication import ExpertAdjudicationStore, LabelSchema
from models.local_hf_multimodal_entailment import MultimodalLabelMapping
from orchestration.continual_adaptation import SQLiteContinualWorkflowStore
from tests.unit.test_continual_adaptation import spec as continual_spec


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_multimodal_label_mapping_rejects_non_integer_boolean_and_negative_indices() -> None:
    with pytest.raises(ValueError, match="entailment_index"):
        MultimodalLabelMapping("0", 1, 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="entailment_index"):
        MultimodalLabelMapping(True, 1, 2)
    with pytest.raises(ValueError, match="entailment_index"):
        MultimodalLabelMapping(-1, 1, 2)


def test_multimodal_label_mapping_rejects_duplicate_semantics() -> None:
    with pytest.raises(ValueError, match="unique"):
        MultimodalLabelMapping(0, 0, 2)


def test_stale_continual_claim_cannot_commit_transition_after_takeover(tmp_path) -> None:
    workflow = continual_spec()
    store = SQLiteContinualWorkflowStore(tmp_path / "continual.sqlite3")
    record = store.ensure(workflow, now=1.0)
    stale = store.claim(workflow.workflow_id, worker_id="worker-a", now=1.0, lease_seconds=1.0)
    current = store.claim(workflow.workflow_id, worker_id="worker-b", now=3.0, lease_seconds=10.0)

    assert current.fencing_token > stale.fencing_token
    with pytest.raises(RuntimeError, match="stale or fenced"):
        store.transition(
            stale,
            expected_state=record.state,
            expected_revision=record.revision,
            new_state="build_requested",
            now=3.0,
        )
    assert store.get(workflow.workflow_id).state == "detected"


def _case(store: ExpertAdjudicationStore):
    return store.create_case(
        owner_id="alice",
        item_sha256=sha("item"),
        evidence_sha256=(sha("evidence-a"), sha("evidence-b")),
        schema=LabelSchema("support", "1", ("entailed", "neutral", "contradicted")),
        now=1.0,
    )


def test_live_reviewer_claim_blocks_same_identity_adjudicator_claim(tmp_path) -> None:
    store = ExpertAdjudicationStore(tmp_path / "reviews.sqlite3")
    case = _case(store)
    store.claim_review(
        case.case.case_id,
        reviewer_id="expert-a",
        role="reviewer",
        now=2.0,
        lease_seconds=100.0,
    )

    with pytest.raises(ValueError, match="reviewer and adjudicator claims"):
        store.claim_review(
            case.case.case_id,
            reviewer_id="expert-a",
            role="adjudicator",
            now=3.0,
            lease_seconds=100.0,
        )


def test_stale_expert_claim_cannot_append_judgment_after_takeover(tmp_path) -> None:
    store = ExpertAdjudicationStore(tmp_path / "reviews.sqlite3")
    case = _case(store)
    stale = store.claim_review(
        case.case.case_id,
        reviewer_id="reviewer-a",
        role="reviewer",
        now=2.0,
        lease_seconds=1.0,
    )
    current = store.claim_review(
        case.case.case_id,
        reviewer_id="reviewer-a",
        role="reviewer",
        now=4.0,
        lease_seconds=10.0,
    )
    assert current.fencing_token > stale.fencing_token

    with pytest.raises(RuntimeError, match="expired or fenced"):
        store.submit_judgment(
            stale,
            label="entailed",
            confidence=0.9,
            rationale_sha256=sha("rationale"),
            expected_case_revision=case.revision,
            now=4.1,
        )
    assert store.judgments(case.case.case_id) == ()


def test_second_adjudicator_identity_cannot_replace_committed_adjudicator(tmp_path) -> None:
    store = ExpertAdjudicationStore(tmp_path / "reviews.sqlite3")
    case = _case(store)
    claim = store.claim_review(
        case.case.case_id,
        reviewer_id="adjudicator-a",
        role="adjudicator",
        now=2.0,
        lease_seconds=2.0,
    )
    store.submit_judgment(
        claim,
        label="neutral",
        confidence=0.95,
        rationale_sha256=sha("expert-rationale"),
        expected_case_revision=case.revision,
        now=2.1,
    )

    with pytest.raises(ValueError, match="different adjudicator identity"):
        store.claim_review(
            case.case.case_id,
            reviewer_id="adjudicator-b",
            role="adjudicator",
            now=5.0,
            lease_seconds=10.0,
        )
