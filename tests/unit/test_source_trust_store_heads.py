from tools.source_trust import SourceTrustFeatures
from tools.source_trust_store import SourceTrustStore


def _features(source_id: str, *, methodology: float) -> SourceTrustFeatures:
    return SourceTrustFeatures(
        source_id=source_id,
        source_type="primary_study",
        status="active",
        provenance_integrity=1.0,
        methodological_quality=methodology,
        topical_applicability=0.8,
        freshness=0.7,
        independent_replication=0.2,
        reviewed=True,
        conflicts_of_interest_known=True,
    )


def test_reactivating_immutable_revision_moves_current_head_and_emits_activation(tmp_path):
    store = SourceTrustStore(tmp_path / "trust.sqlite3")
    first = store.put(
        "owner-a",
        _features("paper-1", methodology=0.4),
        reviewer_id="reviewer-a",
        review_basis="first review",
    )
    second = store.put(
        "owner-a",
        _features("paper-1", methodology=0.9),
        reviewer_id="reviewer-a",
        review_basis="second review",
    )
    assert store.latest("owner-a", "paper-1").revision_id == second.revision_id

    reactivated = store.put(
        "owner-a",
        _features("paper-1", methodology=0.4),
        reviewer_id="reviewer-a",
        review_basis="first review",
    )
    assert reactivated.revision_id == first.revision_id
    assert store.latest("owner-a", "paper-1").revision_id == first.revision_id
    assert [item.revision_id for item in store.list_latest("owner-a")] == [first.revision_id]

    history = store.history("owner-a", "paper-1")
    assert {item.revision_id for item in history} == {first.revision_id, second.revision_id}
    assert len(history) == 2

    activations = store.activation_history("owner-a", "paper-1")
    assert len(activations) == 3
    assert activations[0].revision_id == first.revision_id
    assert activations[0].previous_revision_id == second.revision_id
    assert activations[1].revision_id == second.revision_id
    assert activations[1].previous_revision_id == first.revision_id
    assert activations[2].revision_id == first.revision_id
    assert activations[2].previous_revision_id == ""
    assert len({item.activation_id for item in activations}) == 3
    assert all(item.pending for item in activations)


def test_identical_active_revision_does_not_emit_duplicate_activation(tmp_path):
    store = SourceTrustStore(tmp_path / "trust.sqlite3")
    first = store.put(
        "owner-a",
        _features("paper-1", methodology=0.4),
        reviewer_id="reviewer-a",
        review_basis="same review",
    )
    pending = store.pending_activations("owner-a", source_id="paper-1")
    assert len(pending) == 1

    repeated = store.put(
        "owner-a",
        _features("paper-1", methodology=0.4),
        reviewer_id="reviewer-a",
        review_basis="same review",
    )
    assert repeated.revision_id == first.revision_id
    assert [item.activation_id for item in store.pending_activations("owner-a", source_id="paper-1")] == [pending[0].activation_id]

    store.mark_activation_failed("owner-a", pending[0].activation_id, "TemporaryFailure")
    failed = store.pending_activations("owner-a", source_id="paper-1")[0]
    assert failed.last_error == "TemporaryFailure"
    store.mark_activation_completed("owner-a", pending[0].activation_id)
    assert store.pending_activations("owner-a", source_id="paper-1") == ()
    assert store.activation_history("owner-a", "paper-1")[0].pending is False


def test_source_trust_heads_and_activations_are_owner_scoped(tmp_path):
    store = SourceTrustStore(tmp_path / "trust.sqlite3")
    left = store.put(
        "owner-a",
        _features("paper-1", methodology=0.4),
        reviewer_id="reviewer-a",
        review_basis="owner a review",
    )
    right = store.put(
        "owner-b",
        _features("paper-1", methodology=0.9),
        reviewer_id="reviewer-b",
        review_basis="owner b review",
    )

    assert store.latest("owner-a", "paper-1").revision_id == left.revision_id
    assert store.latest("owner-b", "paper-1").revision_id == right.revision_id
    assert len(store.pending_activations("owner-a")) == 1
    assert len(store.pending_activations("owner-b")) == 1
    assert store.pending_activations("owner-a")[0].revision_id == left.revision_id
    assert store.pending_activations("owner-b")[0].revision_id == right.revision_id
