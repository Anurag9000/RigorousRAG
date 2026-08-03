from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_timestamp_operations as operations,
)
from tools import (
    evidence_graph_set_signed_retirement_restore_custody_timestamp_operations_cli as cli,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_operations import (
    CustodyTimestampRotationPolicy,
    assess_custody_timestamp_authority_rotation,
)


class Registry:
    def __init__(self, values=()):
        self.values = tuple(values)
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return self.values[: kwargs["limit"]]


def record(
    key: str,
    registered: float,
    *,
    state: str = "active",
    authority: str = "tsa",
    retired: float | None = None,
):
    return SimpleNamespace(
        authority_id=authority,
        key_id=key,
        public_key_sha256=(key[0] if key else "a") * 64,
        state=state,
        registered_at=float(registered),
        retired_at=retired,
    )


def policy():
    return CustodyTimestampRotationPolicy(
        maximum_active_key_age_seconds=100,
        minimum_overlap_seconds=10,
        maximum_active_keys=2,
    )


def test_timestamp_rotation_classifications():
    assert assess_custody_timestamp_authority_rotation(
        owner_id="alice", registry=Registry(), policy=policy(), now=200
    ).classification == "initial_key_required"
    assert assess_custody_timestamp_authority_rotation(
        owner_id="alice",
        registry=Registry((record("a1", 150),)),
        policy=policy(),
        now=200,
    ).classification == "healthy_single_active"
    assert assess_custody_timestamp_authority_rotation(
        owner_id="alice",
        registry=Registry((record("a1", 50),)),
        policy=policy(),
        now=200,
    ).classification == "rotation_required_no_successor"
    assert assess_custody_timestamp_authority_rotation(
        owner_id="alice",
        registry=Registry((record("a1", 50), record("b1", 195))),
        policy=policy(),
        now=200,
    ).classification == "overlap_window_active"
    assert assess_custody_timestamp_authority_rotation(
        owner_id="alice",
        registry=Registry((record("a1", 50), record("b1", 180))),
        policy=policy(),
        now=200,
    ).classification == "retire_oldest_after_overlap"
    assert assess_custody_timestamp_authority_rotation(
        owner_id="alice",
        registry=Registry(
            (record("a1", 50), record("b1", 180), record("c1", 190))
        ),
        policy=policy(),
        now=200,
    ).classification == "too_many_active_keys"


def test_timestamp_rotation_report_is_deterministic_and_tamper_checked():
    first = assess_custody_timestamp_authority_rotation(
        owner_id="alice",
        registry=Registry((record("b1", 180), record("a1", 50))),
        policy=policy(),
        now=200,
    )
    second = assess_custody_timestamp_authority_rotation(
        owner_id="alice",
        registry=Registry((record("a1", 50), record("b1", 180))),
        policy=policy(),
        now=200,
    )

    assert first == second
    assert first.oldest_active_authority_id == "tsa"
    assert first.oldest_active_key_id == "a1"
    assert first.newest_active_key_id == "b1"
    assert len(first.report_digest) == 64
    with pytest.raises(ValueError, match="report_digest"):
        first.__class__(**{**first.__dict__, "report_digest": "f" * 64})


def test_timestamp_rotation_refuses_duplicates_bounds_and_invalid_rows():
    duplicate = record("a1", 50)
    with pytest.raises(RuntimeError, match="duplicate"):
        assess_custody_timestamp_authority_rotation(
            owner_id="alice",
            registry=Registry((duplicate, duplicate)),
            policy=policy(),
            now=200,
        )
    with pytest.raises(RuntimeError, match="bounded"):
        assess_custody_timestamp_authority_rotation(
            owner_id="alice",
            registry=Registry((duplicate,)),
            policy=policy(),
            now=200,
            limit=1,
        )
    with pytest.raises(ValueError, match="retired_at"):
        assess_custody_timestamp_authority_rotation(
            owner_id="alice",
            registry=Registry((record("a1", 50, state="retired"),)),
            policy=policy(),
            now=200,
        )


def test_timestamp_rotation_report_is_non_mutating_and_private():
    report = assess_custody_timestamp_authority_rotation(
        owner_id="alice",
        registry=Registry((record("a1", 50),)),
        policy=policy(),
        now=200,
    )

    assert report.registry_mutation_performed is False
    assert report.key_material_mutation_performed is False
    assert report.contains_actor_ids is False
    assert report.contains_raw_paths is False


def test_timestamp_rotation_cli_is_query_only_and_path_free(monkeypatch, capsys):
    registry = Registry((record("a1", 50), record("b1", 195)))
    observed = {}

    def read_only(path):
        observed["path"] = path
        return registry

    monkeypatch.setattr(cli, "ReadOnlyCustodyTimestampAuthorityRegistry", read_only)
    monkeypatch.setattr(operations.time, "time", lambda: 200.0)
    assert cli.main(
        [
            "--owner-id",
            "alice",
            "--registry-db-path",
            "/private/authority.sqlite3",
            "--maximum-active-key-age-seconds",
            "100",
            "--minimum-overlap-seconds",
            "10",
            "--maximum-active-keys",
            "2",
        ]
    ) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert captured.err == ""
    assert observed["path"] == "/private/authority.sqlite3"
    assert payload["classification"] == "overlap_window_active"
    assert payload["registry_mutation_performed"] is False
    assert payload["key_material_mutation_performed"] is False
    assert payload["key_deletion_performed"] is False
    assert payload["attestation_created"] is False
    assert payload["raw_path_returned"] is False
    assert "/private/authority.sqlite3" not in captured.out
