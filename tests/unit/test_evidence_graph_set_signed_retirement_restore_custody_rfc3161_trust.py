from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161 import (
    create_rfc3161_timestamp_request_bundle,
    emit_rfc3161_timestamp_request_der,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_contracts import (
    Rfc3161TimestampVerificationReceipt,
    canonical_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust import (
    Rfc3161TrustProfile,
    Rfc3161TrustRegistry,
    register_rfc3161_trust_profile,
    verify_rfc3161_timestamp_response_with_profile,
)

OPENSSL = shutil.which("openssl")


def actor(name: str = "operator") -> ReviewActorBinding:
    return ReviewActorBinding.create(
        actor_id=name,
        binding_method="process_environment",
        loaded_at=1.0,
    )


def run(*args: str, cwd: Path) -> None:
    result = subprocess.run(
        [OPENSSL, *args],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise AssertionError(result.stdout.decode("utf-8", errors="replace"))


def make_tsa(tmp_path: Path, name: str) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / name
    root.mkdir()
    run(
        "genpkey",
        "-algorithm",
        "RSA",
        "-pkeyopt",
        "rsa_keygen_bits:2048",
        "-out",
        "ca.key",
        cwd=root,
    )
    run(
        "req",
        "-x509",
        "-new",
        "-key",
        "ca.key",
        "-subj",
        f"/CN={name} Root",
        "-days",
        "3650",
        "-out",
        "ca.crt",
        cwd=root,
    )
    run(
        "genpkey",
        "-algorithm",
        "RSA",
        "-pkeyopt",
        "rsa_keygen_bits:2048",
        "-out",
        "tsa.key",
        cwd=root,
    )
    run(
        "req",
        "-new",
        "-key",
        "tsa.key",
        "-subj",
        f"/CN={name} TSA",
        "-out",
        "tsa.csr",
        cwd=root,
    )
    (root / "tsa_ext.cnf").write_text(
        "\n".join(
            [
                "basicConstraints=critical,CA:FALSE",
                "keyUsage=critical,digitalSignature",
                "extendedKeyUsage=critical,timeStamping",
                "subjectKeyIdentifier=hash",
                "authorityKeyIdentifier=keyid,issuer",
                "",
            ]
        )
    )
    run(
        "x509",
        "-req",
        "-in",
        "tsa.csr",
        "-CA",
        "ca.crt",
        "-CAkey",
        "ca.key",
        "-CAcreateserial",
        "-days",
        "365",
        "-extfile",
        "tsa_ext.cnf",
        "-out",
        "tsa.crt",
        cwd=root,
    )
    (root / "tsa.serial").write_text("01\n")
    (root / "tsa.cnf").write_text(
        "\n".join(
            [
                "[ tsa ]",
                "default_tsa = tsa_config1",
                "[ tsa_config1 ]",
                f"serial = {root / 'tsa.serial'}",
                "crypto_device = builtin",
                f"signer_cert = {root / 'tsa.crt'}",
                f"certs = {root / 'ca.crt'}",
                f"signer_key = {root / 'tsa.key'}",
                "signer_digest = sha256",
                "default_policy = 1.2.3.4.1",
                "digests = sha256",
                "ordering = yes",
                "tsa_name = yes",
                "ess_cert_id_chain = yes",
                "ess_cert_id_alg = sha256",
                "",
            ]
        )
    )
    return root, root / "ca.crt", root / "tsa.crt", root / "tsa.cnf"


@pytest.mark.skipif(OPENSSL is None, reason="OpenSSL unavailable")
def test_governed_profile_enforces_exact_trust_policy_and_signer(tmp_path: Path):
    subject = tmp_path / "custody.json"
    subject.write_text("custody")
    bundle = tmp_path / "request.json"
    request = tmp_path / "request.tsq"
    response = tmp_path / "response.tsr"
    create_rfc3161_timestamp_request_bundle(
        owner_id="alice",
        custody_envelope_path=subject,
        output_bundle_path=bundle,
        requested_policy_oid="1.2.3.4.1",
        nonce=(1 << 127) + 99,
        now=100.0,
    )
    emit_rfc3161_timestamp_request_der(
        request_bundle_path=bundle,
        output_der_path=request,
    )
    root, ca, tsa_cert, config = make_tsa(tmp_path, "primary")
    run(
        "ts",
        "-reply",
        "-queryfile",
        str(request),
        "-config",
        str(config),
        "-out",
        str(response),
        cwd=root,
    )
    signer_der = subprocess.check_output(
        [OPENSSL, "x509", "-in", str(tsa_cert), "-outform", "DER"]
    )

    registry = Rfc3161TrustRegistry(tmp_path / "trust.sqlite3")
    profile = register_rfc3161_trust_profile(
        registry=registry,
        owner_id="alice",
        profile_id="institutional-tsa",
        policy_oid="1.2.3.4.1",
        trust_anchor_bundle_path=ca,
        allowed_signer_certificate_sha256=(
            hashlib.sha256(signer_der).hexdigest(),
        ),
        valid_from=0.0,
        actor=actor(),
        now=10.0,
    )
    receipt, resolved = verify_rfc3161_timestamp_response_with_profile(
        registry=registry,
        owner_id="alice",
        profile_id=profile.profile_id,
        request_bundle_path=bundle,
        response_path=response,
        trust_anchor_bundle_path=ca,
    )
    assert resolved == profile
    assert profile.permits(receipt)

    _other_root, other_ca, _other_tsa, _other_config = make_tsa(
        tmp_path,
        "other",
    )
    with pytest.raises(PermissionError, match="trust anchor"):
        verify_rfc3161_timestamp_response_with_profile(
            registry=registry,
            owner_id="alice",
            profile_id=profile.profile_id,
            request_bundle_path=bundle,
            response_path=response,
            trust_anchor_bundle_path=other_ca,
        )


def test_registry_lifecycle_collision_and_database_identity(tmp_path: Path):
    registry = Rfc3161TrustRegistry(tmp_path / "trust.sqlite3")
    value = Rfc3161TrustProfile.active(
        owner_id="alice",
        profile_id="tsa",
        policy_oid="1.2.3.4.1",
        trust_anchor_bundle_sha256="1" * 64,
        untrusted_bundle_sha256=None,
        crl_bundle_sha256=None,
        allowed_signer_certificate_sha256=None,
        valid_from=1.0,
        valid_until=100.0,
        actor=actor(),
        now=2.0,
    )
    assert registry.register(value) == value
    assert registry.register(value) == value
    conflicting = Rfc3161TrustProfile.active(
        owner_id="alice",
        profile_id="tsa",
        policy_oid="1.2.3.4.2",
        trust_anchor_bundle_sha256="1" * 64,
        untrusted_bundle_sha256=None,
        crl_bundle_sha256=None,
        allowed_signer_certificate_sha256=None,
        valid_from=1.0,
        valid_until=100.0,
        actor=actor(),
        now=2.0,
    )
    with pytest.raises(RuntimeError, match="collision"):
        registry.register(conflicting)
    retired = registry.retire(
        owner_id="alice",
        profile_id="tsa",
        actor=actor("retirer"),
        now=50.0,
    )
    assert retired.state == "retired"
    assert registry.get(owner_id="alice", profile_id="tsa") == retired

    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(registry.path.read_bytes())
    os.replace(replacement, registry.path)
    with pytest.raises(RuntimeError, match="identity changed"):
        registry.get(owner_id="alice", profile_id="tsa")


def test_profile_validity_and_signer_allowlist_are_enforced():
    profile = Rfc3161TrustProfile.active(
        owner_id="alice",
        profile_id="tsa",
        policy_oid="1.2.3.4.1",
        trust_anchor_bundle_sha256="1" * 64,
        untrusted_bundle_sha256=None,
        crl_bundle_sha256=None,
        allowed_signer_certificate_sha256=("2" * 64,),
        valid_from=10.0,
        valid_until=20.0,
        actor=actor(),
        now=1.0,
    )
    stable = {
        "scope": "rigorousrag-restore-custody-rfc3161-receipt-v1",
        "owner_id": "alice",
        "request_bundle_digest": "3" * 64,
        "request_sha256": "4" * 64,
        "subject_sha256": "5" * 64,
        "response_sha256": "6" * 64,
        "token_sha256": "7" * 64,
        "status": "granted",
        "policy_oid": "1.2.3.4.1",
        "message_imprint_sha256": "5" * 64,
        "nonce_sha256": "8" * 64,
        "serial_decimal": "1",
        "generated_at_rfc3339": "1970-01-01T00:00:15Z",
        "generated_at_unix": 15.0,
        "accuracy_seconds": None,
        "accuracy_millis": None,
        "accuracy_micros": None,
        "ordering": False,
        "signer_certificate_sha256": "2" * 64,
        "signer_certificate_serial_hex": "1",
        "signer_public_key_algorithm": "rsa",
        "signature_algorithm": "rsassa_pkcs1v15",
        "digest_algorithm": "sha256",
        "trust_anchor_bundle_sha256": "1" * 64,
        "untrusted_bundle_sha256": None,
        "crl_bundle_sha256": None,
        "verifier_version_sha256": "9" * 64,
        "schema_version": 1,
    }
    receipt = Rfc3161TimestampVerificationReceipt(
        **{key: value for key, value in stable.items() if key != "scope"},
        receipt_digest=canonical_digest(stable),
    )
    assert profile.permits(receipt)
    wrong = dict(stable)
    wrong["signer_certificate_sha256"] = "a" * 64
    wrong_receipt = Rfc3161TimestampVerificationReceipt(
        **{key: value for key, value in wrong.items() if key != "scope"},
        receipt_digest=canonical_digest(wrong),
    )
    assert not profile.permits(wrong_receipt)
