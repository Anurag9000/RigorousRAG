from __future__ import annotations

import hashlib

import pytest

from evaluation.expert_adjudication import (
    AdjudicationPolicy,
    ExpertAdjudicationStore,
    LabelSchema,
    write_gold_manifest,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def schema() -> LabelSchema:
    return LabelSchema("citation-support", "1", ("entailed", "neutral", "contradicted"))


def create_case(store: ExpertAdjudicationStore, *, item: str = "item-1"):
    return store.create_case(
        owner_id="alice",
        item_sha256=sha(item),
        evidence_sha256=(sha(f"{item}:evidence-a"), sha(f"{item}:evidence-b")),
        schema=schema(),
        now=1.0,
    )


def submit(
    store: ExpertAdjudicationStore,
    case_id: str,
    reviewer: str,
    label: str,
    *,
    role: str = "reviewer",
    confidence: float = 0.9,
    now: float,
    supersedes: str | None = None,
):
    claim = store.claim_review(case_id, reviewer_id=reviewer, role=role, now=now, lease_seconds=100.0)
    revision = store.get_case(case_id).revision
    return store.submit_judgment(
        claim,
        label=label,
        confidence=confidence,
        rationale_sha256=sha(f"rationale:{reviewer}:{label}:{now}"),
        expected_case_revision=revision,
        now=now + 0.1,
        supersedes_judgment_id=supersedes,
    )


def test_two_independent_unanimous_reviews_resolve_and_export_gold(tmp_path) -> None:
    store = ExpertAdjudicationStore(tmp_path / "reviews.sqlite3")
    case = create_case(store)
    submit(store, case.case.case_id, "reviewer-a", "entailed", now=2.0)
    submit(store, case.case.case_id, "reviewer-b", "entailed", now=3.0)

    current = store.get_case(case.case.case_id)
    resolved = store.reconcile_case(
        case.case.case_id,
        policy=AdjudicationPolicy(),
        expected_case_revision=current.revision,
        now=4.0,
    )

    assert resolved.state == "resolved"
    receipt = store.resolution(case.case.case_id)
    assert receipt.label == "entailed"
    assert receipt.method == "reviewer_consensus"
    assert receipt.reviewer_count == 2

    manifest = store.build_gold_manifest(owner_id="alice", task_id="citation-support")
    assert len(manifest.records) == 1
    assert manifest.records[0].label == "entailed"
    exported = write_gold_manifest(tmp_path / "gold.json", manifest)
    payload = exported.read_text(encoding="utf-8")
    assert "entailed" in payload
    assert "rationale:" not in payload


def test_disagreement_requires_independent_adjudicator(tmp_path) -> None:
    store = ExpertAdjudicationStore(tmp_path / "reviews.sqlite3")
    case = create_case(store)
    submit(store, case.case.case_id, "reviewer-a", "entailed", now=2.0)
    submit(store, case.case.case_id, "reviewer-b", "contradicted", now=3.0)

    current = store.get_case(case.case.case_id)
    waiting = store.reconcile_case(
        case.case.case_id,
        policy=AdjudicationPolicy(),
        expected_case_revision=current.revision,
        now=4.0,
    )
    assert waiting.state == "needs_adjudication"
    assert store.resolution(case.case.case_id) is None

    submit(store, case.case.case_id, "expert-c", "neutral", role="adjudicator", confidence=0.95, now=5.0)
    current = store.get_case(case.case.case_id)
    resolved = store.reconcile_case(
        case.case.case_id,
        policy=AdjudicationPolicy(),
        expected_case_revision=current.revision,
        now=6.0,
    )
    assert resolved.state == "resolved"
    receipt = store.resolution(case.case.case_id)
    assert receipt.label == "neutral"
    assert receipt.method == "adjudicator"


def test_reviewer_cannot_become_adjudicator_on_same_case(tmp_path) -> None:
    store = ExpertAdjudicationStore(tmp_path / "reviews.sqlite3")
    case = create_case(store)
    submit(store, case.case.case_id, "reviewer-a", "entailed", now=2.0)

    with pytest.raises(ValueError, match="switch"):
        store.claim_review(
            case.case.case_id,
            reviewer_id="reviewer-a",
            role="adjudicator",
            now=3.0,
            lease_seconds=100.0,
        )


def test_stale_review_claim_is_fenced(tmp_path) -> None:
    store = ExpertAdjudicationStore(tmp_path / "reviews.sqlite3")
    case = create_case(store)
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


def test_correction_is_append_only_and_must_supersede_current_judgment(tmp_path) -> None:
    store = ExpertAdjudicationStore(tmp_path / "reviews.sqlite3")
    case = create_case(store)
    first_claim = store.claim_review(case.case.case_id, reviewer_id="reviewer-a", role="reviewer", now=2.0, lease_seconds=100.0)
    first = store.submit_judgment(
        first_claim,
        label="neutral",
        confidence=0.6,
        rationale_sha256=sha("first-rationale"),
        expected_case_revision=case.revision,
        now=2.1,
    )
    current = store.get_case(case.case.case_id)

    with pytest.raises(ValueError, match="supersede"):
        store.submit_judgment(
            first_claim,
            label="entailed",
            confidence=0.9,
            rationale_sha256=sha("corrected-rationale"),
            expected_case_revision=current.revision,
            now=2.2,
        )

    corrected = store.submit_judgment(
        first_claim,
        label="entailed",
        confidence=0.9,
        rationale_sha256=sha("corrected-rationale"),
        expected_case_revision=current.revision,
        now=2.3,
        supersedes_judgment_id=first.judgment_id,
    )
    history = store.judgments(case.case.case_id)
    assert len(history) == 2
    assert corrected.reviewer_revision == 2
    assert corrected.supersedes_judgment_id == first.judgment_id
    assert store.active_judgments(case.case.case_id)[0].label == "entailed"


def test_reopening_resolved_case_suspends_old_gold_until_new_round_resolves(tmp_path) -> None:
    store = ExpertAdjudicationStore(tmp_path / "reviews.sqlite3")
    case = create_case(store)
    submit(store, case.case.case_id, "reviewer-a", "entailed", now=2.0)
    submit(store, case.case.case_id, "reviewer-b", "entailed", now=3.0)
    current = store.get_case(case.case.case_id)
    store.reconcile_case(
        case.case.case_id,
        policy=AdjudicationPolicy(),
        expected_case_revision=current.revision,
        now=4.0,
    )
    assert store.build_gold_manifest(owner_id="alice", task_id="citation-support").records[0].round_number == 1

    reopened = store.reopen_resolved_case(
        case.case.case_id,
        reason_sha256=sha("source correction"),
        actor_id="review-admin",
        now=5.0,
    )
    assert reopened.case.round_number == 2
    assert reopened.case.parent_case_id == case.case.case_id

    with pytest.raises(ValueError, match="no current resolved"):
        store.build_gold_manifest(owner_id="alice", task_id="citation-support")

    submit(store, reopened.case.case_id, "reviewer-c", "contradicted", now=6.0)
    submit(store, reopened.case.case_id, "reviewer-d", "contradicted", now=7.0)
    current = store.get_case(reopened.case.case_id)
    store.reconcile_case(
        reopened.case.case_id,
        policy=AdjudicationPolicy(),
        expected_case_revision=current.revision,
        now=8.0,
    )
    manifest = store.build_gold_manifest(owner_id="alice", task_id="citation-support")
    assert manifest.records[0].round_number == 2
    assert manifest.records[0].label == "contradicted"
    assert manifest.records[0].case_id == reopened.case.case_id


def test_label_outside_governed_schema_is_rejected(tmp_path) -> None:
    store = ExpertAdjudicationStore(tmp_path / "reviews.sqlite3")
    case = create_case(store)
    claim = store.claim_review(case.case.case_id, reviewer_id="reviewer-a", role="reviewer", now=2.0)
    with pytest.raises(ValueError, match="governed schema"):
        store.submit_judgment(
            claim,
            label="made-up-label",
            confidence=0.5,
            rationale_sha256=None,
            expected_case_revision=case.revision,
            now=2.1,
        )
