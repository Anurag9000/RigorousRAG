from __future__ import annotations

import json
from types import SimpleNamespace

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_signer_enforced_cli as cli,
)


def item(*, eligible=True, historical=True):
    return SimpleNamespace(
        key_id="key-1",
        registration_classification=(
            "direct_compliant" if eligible else "signed_missing_reservation"
        ),
        registration_use_id=None,
        retirement_classification="not_applicable",
        retirement_use_id=None,
        eligible_for_new_signatures=eligible,
        governance_compliant_for_historical_verification=historical,
    )


def report(value):
    return SimpleNamespace(
        items=(value,),
        report_digest="1" * 64,
    )


def record():
    return SimpleNamespace(
        owner_id="alice",
        key_id="key-1",
        issuer="lab-security",
        public_key_sha256="2" * 64,
    )


class Registry:
    def get(self, **kwargs):
        return record()


def install(monkeypatch, compliance_item):
    monkeypatch.setattr(cli, "_stores", lambda args: (Registry(), None))
    monkeypatch.setattr(
        cli,
        "audit_custody_signer_compliance",
        lambda **kwargs: report(compliance_item),
    )


def test_sign_compliant_refuses_ungoverned_active_key(monkeypatch, capsys):
    install(monkeypatch, item(eligible=False, historical=False))
    assert cli.main(
        [
            "sign-compliant",
            "chain.json",
            "--owner-id",
            "alice",
            "--key-id",
            "key-1",
            "--private-key-path",
            "private.pem",
            "--output",
            "signed.json",
        ]
    ) == 1
    assert json.loads(capsys.readouterr().err) == {
        "error": "not_authorized_or_noncompliant"
    }


def test_sign_compliant_requires_matching_manifest_and_private_fingerprint(
    monkeypatch,
    capsys,
):
    install(monkeypatch, item())
    monkeypatch.setattr(
        cli,
        "verify_restore_chain_of_custody",
        lambda path: SimpleNamespace(
            owner_id="alice",
            restore_id="3" * 64,
            chain_digest="4" * 64,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_private",
        lambda path: SimpleNamespace(public_key=lambda: object()),
    )
    monkeypatch.setattr(cli, "_public_fingerprint", lambda key: "2" * 64)
    monkeypatch.setattr(
        cli,
        "sign_restore_chain_of_custody",
        lambda **kwargs: SimpleNamespace(
            manifest=SimpleNamespace(
                restore_id="3" * 64,
                chain_digest="4" * 64,
            )
        ),
    )

    assert cli.main(
        [
            "sign-compliant",
            "chain.json",
            "--owner-id",
            "alice",
            "--key-id",
            "key-1",
            "--private-key-path",
            "/private/private.pem",
            "--output",
            "/private/signed.json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["signature_created"] is True
    assert payload["eligible_for_new_signatures"] is True
    assert payload["registry_mutation_performed"] is False
    assert "/private" not in json.dumps(payload)


def test_historical_verification_reports_or_enforces_governance(monkeypatch, capsys):
    noncompliant = item(eligible=False, historical=False)
    install(monkeypatch, noncompliant)
    monkeypatch.setattr(
        cli,
        "verify_signed_restore_chain_of_custody",
        lambda **kwargs: SimpleNamespace(
            manifest=SimpleNamespace(
                owner_id="alice",
                restore_id="3" * 64,
                chain_digest="4" * 64,
            )
        ),
    )

    arguments = [
        "verify-compliance",
        "signed.json",
        "--owner-id",
        "alice",
        "--key-id",
        "key-1",
        "--public-key-path",
        "public.pem",
    ]
    assert cli.main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["signature_valid"] is True
    assert payload["governance_compliant_for_historical_verification"] is False
    assert payload["governance_requirement_enforced"] is False

    assert cli.main(arguments + ["--require-governance-compliance"]) == 1
    assert json.loads(capsys.readouterr().err) == {
        "error": "not_authorized_or_noncompliant"
    }
