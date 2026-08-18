from __future__ import annotations

import hashlib

import pytest

from evaluation.retrieval_interleaving import InterleavingOutcome, InterleavingSpec, RankedIdentity, build_team_draft_interleaving
from orchestration.retrieval_interleaving_journal import OwnerScopedInterleavingExperiment, SQLiteInterleavingJournal


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def experiment(owner: str = "alice") -> OwnerScopedInterleavingExperiment:
    return OwnerScopedInterleavingExperiment(owner, InterleavingSpec(sha("experiment"), sha("baseline"), sha("candidate"), max_positions=4))


def ranking(prefix: str):
    return tuple(RankedIdentity(f"{prefix}-{index}", f"source-{prefix}-{index}") for index in range(4))


def impression(exp: OwnerScopedInterleavingExperiment, *, query: str = "q", index: int = 0):
    return build_team_draft_interleaving(
        exp.spec,
        query_sha256=sha(query),
        impression_index=index,
        ranking_a=ranking(f"a-{query}-{index}"),
        ranking_b=ranking(f"b-{query}-{index}"),
    )


def test_experiment_impression_and_outcome_replay_are_idempotent(tmp_path) -> None:
    journal = SQLiteInterleavingJournal(tmp_path / "journal.sqlite3")
    exp = experiment()
    experiment_id = journal.ensure_experiment(exp, now=1.0)
    assert journal.ensure_experiment(exp, now=2.0) == experiment_id
    shown = impression(exp)
    assert journal.record_impression(owner_id="alice", experiment_id=experiment_id, impression=shown, now=3.0) == shown.impression_sha256
    assert journal.record_impression(owner_id="alice", experiment_id=experiment_id, impression=shown, now=4.0) == shown.impression_sha256
    outcome = InterleavingOutcome.build(shown, ())
    assert journal.record_outcome(owner_id="alice", experiment_id=experiment_id, outcome=outcome, now=5.0) == outcome.outcome_sha256
    assert journal.record_outcome(owner_id="alice", experiment_id=experiment_id, outcome=outcome, now=6.0) == outcome.outcome_sha256


def test_cross_owner_reads_and_mutations_are_rejected(tmp_path) -> None:
    journal = SQLiteInterleavingJournal(tmp_path / "journal.sqlite3")
    exp = experiment("alice")
    experiment_id = journal.ensure_experiment(exp, now=1.0)
    shown = impression(exp)
    with pytest.raises(PermissionError, match="owner mismatch"):
        journal.record_impression(owner_id="bob", experiment_id=experiment_id, impression=shown, now=2.0)
    with pytest.raises(PermissionError, match="owner mismatch"):
        journal.export_complete_evidence(owner_id="bob", experiment_id=experiment_id)


def test_outcome_cannot_be_recorded_under_another_experiment(tmp_path) -> None:
    journal = SQLiteInterleavingJournal(tmp_path / "journal.sqlite3")
    first = experiment("alice")
    second = OwnerScopedInterleavingExperiment("alice", InterleavingSpec(sha("experiment-two"), sha("baseline"), sha("candidate-two"), max_positions=4))
    first_id = journal.ensure_experiment(first, now=1.0)
    second_id = journal.ensure_experiment(second, now=1.0)
    shown = impression(first)
    journal.record_impression(owner_id="alice", experiment_id=first_id, impression=shown, now=2.0)
    outcome = InterleavingOutcome.build(shown, ())
    with pytest.raises(ValueError, match="does not belong"):
        journal.record_outcome(owner_id="alice", experiment_id=second_id, outcome=outcome, now=3.0)


def test_query_and_impression_index_cannot_be_re_randomized_after_commit(tmp_path) -> None:
    journal = SQLiteInterleavingJournal(tmp_path / "journal.sqlite3")
    exp = experiment()
    experiment_id = journal.ensure_experiment(exp, now=1.0)
    first = impression(exp, query="same", index=0)
    journal.record_impression(owner_id="alice", experiment_id=experiment_id, impression=first, now=2.0)
    second = build_team_draft_interleaving(
        exp.spec,
        query_sha256=sha("same"),
        impression_index=0,
        ranking_a=tuple(reversed(ranking("different-a"))),
        ranking_b=tuple(reversed(ranking("different-b"))),
    )
    assert second.impression_sha256 != first.impression_sha256
    with pytest.raises(RuntimeError, match="already belongs"):
        journal.record_impression(owner_id="alice", experiment_id=experiment_id, impression=second, now=3.0)


def test_outcome_is_immutable_after_first_record(tmp_path) -> None:
    journal = SQLiteInterleavingJournal(tmp_path / "journal.sqlite3")
    exp = experiment()
    experiment_id = journal.ensure_experiment(exp, now=1.0)
    shown = impression(exp)
    journal.record_impression(owner_id="alice", experiment_id=experiment_id, impression=shown, now=2.0)
    first = InterleavingOutcome.build(shown, ())
    journal.record_outcome(owner_id="alice", experiment_id=experiment_id, outcome=first, now=3.0)
    changed = InterleavingOutcome.build(shown, (1,))
    with pytest.raises(RuntimeError, match="immutable"):
        journal.record_outcome(owner_id="alice", experiment_id=experiment_id, outcome=changed, now=4.0)


def test_complete_export_contains_only_impressions_with_recorded_outcomes(tmp_path) -> None:
    journal = SQLiteInterleavingJournal(tmp_path / "journal.sqlite3")
    exp = experiment()
    experiment_id = journal.ensure_experiment(exp, now=1.0)
    complete = impression(exp, query="complete", index=0)
    incomplete = impression(exp, query="incomplete", index=1)
    journal.record_impression(owner_id="alice", experiment_id=experiment_id, impression=complete, now=2.0)
    journal.record_impression(owner_id="alice", experiment_id=experiment_id, impression=incomplete, now=2.0)
    outcome = InterleavingOutcome.build(complete, ())
    journal.record_outcome(owner_id="alice", experiment_id=experiment_id, outcome=outcome, now=3.0)
    exported = journal.export_complete_evidence(owner_id="alice", experiment_id=experiment_id)
    assert exported.owner_id == "alice"
    assert exported.experiment_id == experiment_id
    assert exported.impressions == (complete,)
    assert exported.outcomes == (outcome,)
    assert len(exported.export_sha256) == 64
