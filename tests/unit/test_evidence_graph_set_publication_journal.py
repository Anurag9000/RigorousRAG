from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tools import evidence_graph_set_publish_journal_cli as cli
from tools import evidence_graph_set_publish_reconcile as reconcile
from tools.evidence_graph_set_publish_attempts import (
    EvidenceGraphSetPublicationAttempt,
    EvidenceGraphSetPublicationJournal,
)


@dataclass(frozen=True)
class Endpoint:
    doc_id: str


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    owner_id: str
    graph_set_key: str
    source: Endpoint
    target: Endpoint


@dataclass(frozen=True)
class FakeSet:
    owner_id: str
    graph_set_key: str
    graph_set_id: str
    graph_set_digest: str
    members: tuple[int, ...] = (1, 2)
    edges: tuple[int, ...] = (1,)


class Ledger:
    def __init__(self, proposals: tuple[Proposal, ...]):
        self.proposals = {item.proposal_id: item for item in proposals}

    def get_proposal(self, proposal_id: str) -> Proposal:
        return self.proposals[proposal_id]


class Store:
    _UNSET = object()

    def __init__(self) -> None:
        self.values: dict[str, FakeSet] = {}
        self.pointers: dict[tuple[str, str], FakeSet] = {}

    def commit(
        self,
        value: FakeSet,
        *,
        make_current: bool = True,
        expected_current_set_id: str | None | object = _UNSET,
        now: float | None = None,
    ) -> FakeSet:
        self.values[value.graph_set_id] = value
        if make_current:
            key = (value.owner_id, value.graph_set_key)
            actual = self.pointers.get(key)
            actual_id = None if actual is None else actual.graph_set_id
            if (
                expected_current_set_id is not self._UNSET
                and actual_id != expected_current_set_id
            ):
                raise RuntimeError("graph set current pointer changed concurrently.")
            self.pointers[key] = value
        return value

    def get(self, *, owner_id: str, graph_set_id: str) -> FakeSet:
        value = self.values[graph_set_id]
        assert value.owner_id == owner_id
        return value

    def current(self, *, owner_id: str, graph_set_key: str) -> FakeSet | None:
        return self.pointers.get((owner_id, graph_set_key))


def _clear_pointer(
    store: Store,
    *,
    owner_id: str,
    graph_set_key: str,
    expected_current_set_id: str,
) -> bool:
    key = (owner_id, graph_set_key)
    value = store.pointers.get(key)
    if value is None:
        return False
    if value.graph_set_id != expected_current_set_id:
        raise RuntimeError("graph set current pointer changed concurrently.")
    del store.pointers[key]
    return True


def _candidate(owner_id: str, graph_set_key: str, proposal_id: str) -> FakeSet:
    graph_set_id = hashlib.sha256(
        f"{owner_id}\0{graph_set_key}\0{proposal_id}".encode()
    ).hexdigest()
    return FakeSet(
        owner_id=owner_id,
        graph_set_key=graph_set_key,
        graph_set_id=graph_set_id,
        graph_set_digest=hashlib.sha256(graph_set_id.encode()).hexdigest(),
    )


def _install(monkeypatch: pytest.MonkeyPatch, *, authority: bool = True) -> None:
    monkeypatch.setattr(reconcile, "_document_lock", lambda *args: nullcontext())
    monkeypatch.setattr(
        reconcile,
        "resolve_evidence_graph",
        lambda **kwargs: SimpleNamespace(doc_id=kwargs["doc_id"]),
    )
    monkeypatch.setattr(
        reconcile,
        "approved_relations",
        lambda **kwargs: tuple(kwargs["proposal_ids"]),
    )
    monkeypatch.setattr(
        reconcile,
        "build_evidence_graph_set",
        lambda *, owner_id, graph_set_key, relations, **kwargs: _candidate(
            owner_id, graph_set_key, relations[0]
        ),
    )
    monkeypatch.setattr(
        reconcile,
        "assess_graph_set_authority",
        lambda value, **kwargs: SimpleNamespace(
            authoritative_current=authority,
            authority_digest=("a" if authority else "b") * 64,
        ),
    )
    monkeypatch.setattr(reconcile, "clear_current_graph_set_pointer", _clear_pointer)


def _setup(
    tmp_path,
    monkeypatch,
    *,
    expected=None,
    proposal_digit="1",
    maximum=4,
):
    _install(monkeypatch)
    proposal = Proposal(
        proposal_id=proposal_digit * 64,
        owner_id="alice",
        graph_set_key="review",
        source=Endpoint("doc-a"),
        target=Endpoint("doc-b"),
    )
    ledger = Ledger((proposal,))
    store = Store()
    journal = EvidenceGraphSetPublicationJournal(tmp_path / "publications.sqlite3")
    attempt = journal.seed(
        EvidenceGraphSetPublicationAttempt.create(
            owner_id="alice",
            graph_set_key="review",
            proposal_ids=(proposal.proposal_id,),
            expected_current_set_id=expected,
            max_attempts=maximum,
            now=1.0,
        )
    )
    return proposal, ledger, store, journal, attempt


def _execute(values, *, now=2.0, hook=None):
    _, ledger, store, journal, attempt = values
    return reconcile.execute_publication_attempt(
        attempt.operation_id,
        worker_id="worker",
        lease_seconds=10,
        journal=journal,
        ledger=ledger,
        set_store=store,
        generations=object(),
        graphs=object(),
        now=now,
        _phase_hook=hook,
    )


def test_journal_identity_seed_and_lease_reclaim(tmp_path, monkeypatch):
    *_, journal, attempt = _setup(tmp_path, monkeypatch, maximum=2)
    same = journal.seed(
        EvidenceGraphSetPublicationAttempt.create(
            owner_id="alice",
            graph_set_key="review",
            proposal_ids=("1" * 64,),
            expected_current_set_id=None,
            max_attempts=2,
            now=9.0,
        )
    )
    assert same.created_at == attempt.created_at == 1.0
    assert journal.claim(
        attempt.operation_id, worker_id="one", lease_seconds=1, now=2.0
    ).attempt_count == 1
    with pytest.raises(RuntimeError, match="claimable"):
        journal.claim(
            attempt.operation_id, worker_id="two", lease_seconds=1, now=2.5
        )
    assert journal.claim(
        attempt.operation_id, worker_id="two", lease_seconds=1, now=4.0
    ).attempt_count == 2
    with pytest.raises(RuntimeError, match="ceiling"):
        journal.claim(
            attempt.operation_id, worker_id="three", lease_seconds=1, now=6.0
        )


def test_success_and_terminal_replay(tmp_path, monkeypatch):
    values = _setup(tmp_path, monkeypatch)
    first = _execute(values)
    assert first.state == "completed"
    assert first.pointer_current_set_id == first.candidate_graph_set_id
    second = _execute(values, now=3.0)
    assert second.state == "completed"
    assert second.graph_set_mutation_performed is False


def test_crash_after_pointer_commit_recovers(tmp_path, monkeypatch):
    values = _setup(tmp_path, monkeypatch)

    def crash(name, attempt):
        if name == "pointer_committed":
            raise SystemExit()

    with pytest.raises(SystemExit):
        _execute(values, hook=crash)
    _, _, store, journal, attempt = values
    assert journal.get(attempt.operation_id).phase == "candidate_stored"
    assert store.current(owner_id="alice", graph_set_key="review") is not None
    recovered = _execute(values, now=20.0)
    assert recovered.state == "completed"


def test_exception_after_unjournaled_activation_compensates(tmp_path, monkeypatch):
    values = _setup(tmp_path, monkeypatch)

    def fail(name, attempt):
        if name == "pointer_committed":
            raise RuntimeError("between pointer and phase journal")

    with pytest.raises(reconcile.EvidenceGraphSetPublicationRecoveryError):
        _execute(values, hook=fail)
    _, _, store, journal, attempt = values
    assert store.current(owner_id="alice", graph_set_key="review") is None
    assert journal.get(attempt.operation_id).state == "compensated"


def test_crash_after_compensation_is_reconciled(tmp_path, monkeypatch):
    values = _setup(tmp_path, monkeypatch)

    def crash(name, attempt):
        if name == "before_final_verification":
            monkeypatch.setattr(
                reconcile,
                "assess_graph_set_authority",
                lambda value, **kwargs: SimpleNamespace(
                    authoritative_current=False,
                    authority_digest="b" * 64,
                ),
            )
        if name == "pointer_compensated":
            raise SystemExit()

    with pytest.raises(SystemExit):
        _execute(values, hook=crash)
    _, _, store, journal, attempt = values
    assert store.current(owner_id="alice", graph_set_key="review") is None
    assert journal.get(attempt.operation_id).phase == "pointer_activated"
    assert _execute(values, now=20.0).state == "compensated"


def test_replacement_failure_restores_previous_pointer(tmp_path, monkeypatch):
    proposal, ledger, store, journal, first_attempt = _setup(
        tmp_path, monkeypatch
    )
    first = _execute((proposal, ledger, store, journal, first_attempt))
    second = Proposal(
        proposal_id="2" * 64,
        owner_id="alice",
        graph_set_key="review",
        source=Endpoint("doc-a"),
        target=Endpoint("doc-b"),
    )
    ledger.proposals[second.proposal_id] = second
    second_attempt = journal.seed(
        EvidenceGraphSetPublicationAttempt.create(
            owner_id="alice",
            graph_set_key="review",
            proposal_ids=(second.proposal_id,),
            expected_current_set_id=first.candidate_graph_set_id,
            now=5.0,
        )
    )

    def fail(name, attempt):
        if name == "pointer_recorded":
            raise RuntimeError("after activation")

    with pytest.raises(reconcile.EvidenceGraphSetPublicationRecoveryError):
        _execute(
            (second, ledger, store, journal, second_attempt),
            now=6.0,
            hook=fail,
        )
    assert (
        store.current(owner_id="alice", graph_set_key="review").graph_set_id
        == first.candidate_graph_set_id
    )
    assert journal.get(second_attempt.operation_id).state == "compensated"


def test_database_and_row_tampering_fail_closed(tmp_path, monkeypatch):
    *_, journal, attempt = _setup(tmp_path, monkeypatch)
    with sqlite3.connect(journal.path) as connection:
        connection.execute(
            "UPDATE evidence_graph_set_publications SET proposal_ids_json='[\"bad\"]'"
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        journal.get(attempt.operation_id)

    path = tmp_path / "identity.sqlite3"
    guarded = EvidenceGraphSetPublicationJournal(path)
    path.rename(tmp_path / "old.sqlite3")
    path.write_bytes(b"")
    with pytest.raises(RuntimeError, match="identity changed"):
        guarded.list(owner_id="alice")


def test_retry_cancel_and_confirmation_boundaries(tmp_path, monkeypatch):
    values = _setup(tmp_path, monkeypatch, maximum=3)

    def fail(name, attempt):
        if name == "pointer_recorded":
            raise RuntimeError("boom")

    with pytest.raises(reconcile.EvidenceGraphSetPublicationRecoveryError):
        _execute(values, hook=fail)
    _, _, _, journal, attempt = values
    with pytest.raises(ValueError, match="confirmation"):
        journal.retry(
            attempt.operation_id,
            owner_id="alice",
            confirm_operation_id="f" * 64,
        )
    retried = journal.retry(
        attempt.operation_id,
        owner_id="alice",
        confirm_operation_id=attempt.operation_id,
        now=10.0,
    )
    assert retried.state == "planned" and retried.phase == "candidate_stored"

    other = journal.seed(
        EvidenceGraphSetPublicationAttempt.create(
            owner_id="alice",
            graph_set_key="other",
            proposal_ids=("3" * 64,),
            expected_current_set_id=None,
            now=11.0,
        )
    )
    with pytest.raises(RuntimeError, match="owner"):
        journal.cancel(
            other.operation_id,
            owner_id="bob",
            confirm_operation_id=other.operation_id,
        )
    assert journal.cancel(
        other.operation_id,
        owner_id="alice",
        confirm_operation_id=other.operation_id,
    ).state == "cancelled"


def test_cli_seed_status_list_cancel_and_idle(tmp_path, monkeypatch, capsys):
    journal = EvidenceGraphSetPublicationJournal(tmp_path / "cli.sqlite3")
    monkeypatch.setattr(
        cli, "get_evidence_graph_set_publication_journal", lambda: journal
    )
    assert cli.main(
        [
            "seed",
            "--owner-id",
            "alice",
            "--graph-set-key",
            "review",
            "--proposal-id",
            "1" * 64,
            "--expect-no-current",
        ]
    ) == 0
    seeded = json.loads(capsys.readouterr().out)
    operation_id = seeded["operation_id"]
    assert seeded["source_text_returned"] is False
    assert cli.main(["status", operation_id]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "planned"
    assert cli.main(["list", "--owner-id", "alice"]) == 0
    assert json.loads(capsys.readouterr().out)["count"] == 1
    assert cli.main(
        [
            "cancel",
            operation_id,
            "--owner-id",
            "alice",
            "--confirm-operation-id",
            operation_id,
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "cancelled"

    monkeypatch.setattr(
        cli,
        "_dependencies",
        lambda: {
            "journal": journal,
            "ledger": object(),
            "set_store": object(),
            "generations": object(),
            "graphs": object(),
        },
    )
    assert cli.main(
        ["reconcile-one", "--owner-id", "alice", "--worker-id", "worker"]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "idle"
