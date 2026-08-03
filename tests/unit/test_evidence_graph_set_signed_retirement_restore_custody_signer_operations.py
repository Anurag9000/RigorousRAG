from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_signer_operations_cli as cli,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_operations import (
    CustodySignerRotationPolicy,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_operations_boundary import (
    assess_custody_signer_rotation,
)


def key(
    key_id: str,
    *,
    issuer: str = "lab-security",
    state: str = "active",
    registered_at: float = 1.0,
    retired_at: float | None = None,
):
    return SimpleNamespace(
        owner_id="alice",
        key_id=key_id,
        issuer=issuer,
        algorithm="ed25519",
        public_key_sha256=(key_id[-1] if key_id[-1] in "0123456789abcdef" else "a") * 64,
        state=state,
        registered_at=registered_at,
        retired_at=retired_at,
    )


class Registry:
    def __init__(self, values):
        self.values = tuple(values)
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return self.values[: kwargs["limit"]]


def policy(**overrides):
    values = {
        "maximum_active_keys": 2,
        "maximum_key_age_seconds": 100.0,
        "rotation_warning_seconds": 20.0,
        "minimum_overlap_seconds": 10.0,
        "allowed_issuers": ("lab-security",),
    }
    values.update(overrides)
    return CustodySignerRotationPolicy.create(**values)


def test_rotation_policy_is_deterministic_and_tamper_evident():
    first = policy()
    second = policy(allowed_issuers=("lab-security", "lab-security"))
    assert first == second
    assert len(first.policy_digest) == 64
    with pytest.raises(ValueError, match="policy_digest"):
        replace(first, policy_digest="0" * 64)
    with pytest.raises(ValueError, match="warning"):
        policy(maximum_key_age_seconds=10.0, rotation_warning_seconds=11.0)


def test_empty_or_fully_retired_registry_uses_global_initial_key_action():
    empty = assess_custody_signer_rotation(
        owner_id="alice",
        registry=Registry(()),
        policy=policy(),
        now=100.0,
        limit=100,
    )
    assert empty.active_count == 0
    assert empty.retired_count == 0
    assert empty.items == ()
    assert empty.global_actions == ("register_initial_key",)

    retired = assess_custody_signer_rotation(
        owner_id="alice",
        registry=Registry(
            (key("key-a", state="retired", retired_at=50.0),)
        ),
        policy=policy(),
        now=100.0,
        limit=100,
    )
    assert retired.active_count == 0
    assert retired.retired_count == 1
    assert retired.global_actions == ("register_initial_key",)
    assert retired.items[0].classification == "retired"
    with pytest.raises(ValueError, match="assessment_digest"):
        replace(retired, assessment_digest="0" * 64)


def test_rotation_assessment_detects_age_overlap_count_and_issuer_issues():
    values = (
        key("key-1", registered_at=0.0),
        key("key-2", registered_at=95.0),
        key("key-3", registered_at=96.0),
        key("key-4", issuer="unapproved", registered_at=97.0),
        key("key-a", state="retired", registered_at=1.0, retired_at=50.0),
    )
    report = assess_custody_signer_rotation(
        owner_id="alice",
        registry=Registry(values),
        policy=policy(maximum_active_keys=2),
        now=110.0,
        limit=100,
    )
    actions = {item.key_id: item.action for item in report.items}
    classifications = {item.key_id: item.classification for item in report.items}
    assert report.active_count == 4
    assert report.retired_count == 1
    assert actions["key-1"] == "reduce_active_key_count"
    assert actions["key-4"] == "investigate_unapproved_issuer"
    assert classifications["key-4"] == "active_unapproved_issuer"
    assert report.global_actions == ()
    assert report.registry_mutation_performed is False


def test_rotation_assessment_refuses_truncation_and_cli_is_read_only(
    monkeypatch,
    capsys,
):
    registry = Registry((key("key-1", registered_at=1.0),))
    with pytest.raises(RuntimeError, match="bounded"):
        assess_custody_signer_rotation(
            owner_id="alice",
            registry=registry,
            policy=policy(),
            now=100.0,
            limit=1,
        )

    monkeypatch.setattr(
        cli,
        "ReadOnlyCustodySignerKeyRegistry",
        lambda _path: Registry((key("key-1", registered_at=90.0),)),
    )
    assert cli.main(
        [
            "--owner-id",
            "alice",
            "--registry-db-path",
            "signers.sqlite3",
            "--allowed-issuer",
            "lab-security",
            "--maximum-key-age-seconds",
            "100",
            "--rotation-warning-seconds",
            "20",
            "--minimum-overlap-seconds",
            "10",
            "--limit",
            "100",
        ]
    ) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["registry_mutation_performed"] is False
    assert payload["key_material_mutation_performed"] is False
    assert payload["key_deletion_performed"] is False
    assert payload["raw_path_returned"] is False
    assert payload["policy"]["allowed_issuers"] == ["lab-security"]
