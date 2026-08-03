from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools.evidence_graph_set_signed_retirement_restore_custody_signer_compliance import (
    audit_custody_signer_compliance,
)


def signer(
    key_id: str,
    *,
    state="active",
    registration_method="process_environment",
    registration_binding="1",
    retirement_method=None,
    retirement_binding=None,
):
    return SimpleNamespace(
        owner_id="alice",
        key_id=key_id,
        issuer="lab-security",
        algorithm="ed25519",
        public_key_sha256=(key_id[-1] if key_id[-1] in "0123456789abcdef" else "a") * 64,
        state=state,
        registered_binding_method=registration_method,
        registered_binding_digest=registration_binding * 64,
        retired_binding_method=retirement_method,
        retired_binding_digest=(
            None if retirement_binding is None else retirement_binding * 64
        ),
    )


def use(
    *,
    binding="2",
    action="register",
    key_id="key-2",
    state="committed",
    action_digest="3" * 64,
):
    return SimpleNamespace(
        use_id="f" * 64,
        binding_digest=binding * 64,
        owner_id="alice",
        action=action,
        key_id=key_id,
        action_digest=action_digest,
        state=state,
    )


class Values:
    def __init__(self, values):
        self.values = tuple(values)

    def list(self, **kwargs):
        return self.values[: kwargs["limit"]]


def test_direct_and_committed_signed_registrations_are_compliant(monkeypatch):
    direct = signer("key-1")
    signed = signer(
        "key-2",
        registration_method="signed_assertion",
        registration_binding="2",
    )
    monkeypatch.setattr(
        "tools.evidence_graph_set_signed_retirement_restore_custody_signer_compliance._register_action_digest",
        lambda value: "3" * 64 if value.key_id == "key-2" else "4" * 64,
    )
    report = audit_custody_signer_compliance(
        owner_id="alice",
        registry=Values((direct, signed)),
        admin_use_store=Values((use(),)),
        now=10.0,
        limit=100,
    )
    items = {item.key_id: item for item in report.items}
    assert items["key-1"].registration_classification == "direct_compliant"
    assert items["key-2"].registration_classification == "signed_committed_compliant"
    assert report.compliant_active_count == 2
    assert report.noncompliant_active_count == 0
    with pytest.raises(ValueError, match="report_digest"):
        replace(report, report_digest="0" * 64)


def test_missing_reserved_and_scope_mismatched_signed_uses_are_noncompliant(monkeypatch):
    missing = signer(
        "key-a",
        registration_method="signed_assertion",
        registration_binding="a",
    )
    reserved_signer = signer(
        "key-b",
        registration_method="signed_assertion",
        registration_binding="b",
    )
    mismatch = signer(
        "key-c",
        registration_method="signed_assertion",
        registration_binding="c",
    )
    monkeypatch.setattr(
        "tools.evidence_graph_set_signed_retirement_restore_custody_signer_compliance._register_action_digest",
        lambda value: value.key_id[-1] * 64,
    )
    reserved_use = use(
        binding="b",
        key_id="key-b",
        state="reserved",
        action_digest="b" * 64,
    )
    mismatch_use = use(
        binding="c",
        key_id="other",
        state="committed",
        action_digest="c" * 64,
    )
    mismatch_use.use_id = "e" * 64
    report = audit_custody_signer_compliance(
        owner_id="alice",
        registry=Values((missing, reserved_signer, mismatch)),
        admin_use_store=Values((reserved_use, mismatch_use)),
        now=10.0,
        limit=100,
    )
    items = {item.key_id: item for item in report.items}
    assert items["key-a"].registration_classification == "signed_missing_reservation"
    assert items["key-b"].registration_classification == "signed_reserved_incomplete"
    assert items["key-c"].registration_classification == "signed_scope_mismatch"
    assert report.compliant_active_count == 0
    assert report.noncompliant_active_count == 3


def test_retired_key_requires_governed_registration_and_retirement(monkeypatch):
    value = signer(
        "key-d",
        state="retired",
        registration_method="signed_assertion",
        registration_binding="d",
        retirement_method="signed_assertion",
        retirement_binding="e",
    )
    monkeypatch.setattr(
        "tools.evidence_graph_set_signed_retirement_restore_custody_signer_compliance._register_action_digest",
        lambda value: "4" * 64,
    )
    monkeypatch.setattr(
        "tools.evidence_graph_set_signed_retirement_restore_custody_signer_compliance._retire_action_digest",
        lambda value: "5" * 64,
    )
    register_use = use(
        binding="d",
        action="register",
        key_id="key-d",
        action_digest="4" * 64,
    )
    retire_use = use(
        binding="e",
        action="retire",
        key_id="key-d",
        action_digest="5" * 64,
    )
    retire_use.use_id = "e" * 64
    report = audit_custody_signer_compliance(
        owner_id="alice",
        registry=Values((value,)),
        admin_use_store=Values((register_use, retire_use)),
        now=10.0,
        limit=100,
    )
    item = report.items[0]
    assert item.registration_classification == "signed_committed_compliant"
    assert item.retirement_classification == "signed_committed_compliant"
    assert item.eligible_for_new_signatures is False
    assert item.governance_compliant_for_historical_verification is True


def test_compliance_audit_refuses_duplicate_and_bounded_results():
    value = signer("key-1")
    with pytest.raises(RuntimeError, match="bounded signer"):
        audit_custody_signer_compliance(
            owner_id="alice",
            registry=Values((value,)),
            now=10.0,
            limit=1,
        )
    with pytest.raises(RuntimeError, match="duplicate key"):
        audit_custody_signer_compliance(
            owner_id="alice",
            registry=Values((value, value)),
            now=10.0,
            limit=100,
        )
