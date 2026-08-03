from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from asn1crypto import tsp

from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161 import (
    create_rfc3161_timestamp_request_bundle,
    emit_rfc3161_timestamp_request_der,
    verify_rfc3161_timestamp_receipt,
    verify_rfc3161_timestamp_request_bundle,
    verify_rfc3161_timestamp_response,
)

OPENSSL = shutil.which("openssl")


def run(*args: str, cwd: Path) -> None:
    result = subprocess.run(
        [OPENSSL, *args],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise AssertionError(result.stdout.decode("utf-8", errors="replace"))


def make_tsa(tmp_path: Path, *, name: str = "one") -> tuple[Path, Path, Path]:
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
                "other_policies = 1.2.3.4.2",
                "digests = sha256",
                "accuracy = secs:1, millisecs:500",
                "ordering = yes",
                "tsa_name = yes",
                "ess_cert_id_chain = yes",
                "ess_cert_id_alg = sha256",
                "",
            ]
        )
    )
    return root, root / "ca.crt", root / "tsa.cnf"


def reply(root: Path, config: Path, request: Path, output: Path) -> None:
    run(
        "ts",
        "-reply",
        "-queryfile",
        str(request),
        "-config",
        str(config),
        "-out",
        str(output),
        cwd=root,
    )


@pytest.mark.skipif(OPENSSL is None, reason="OpenSSL unavailable")
def test_request_bundle_and_real_rfc3161_response_round_trip(tmp_path: Path):
    subject = tmp_path / "custody.json"
    subject.write_text('{"chain":"evidence"}\n')
    bundle_path = tmp_path / "request.json"
    request_path = tmp_path / "request.tsq"
    response_path = tmp_path / "response.tsr"
    receipt_path = tmp_path / "receipt.json"

    bundle = create_rfc3161_timestamp_request_bundle(
        owner_id="alice",
        custody_envelope_path=subject,
        output_bundle_path=bundle_path,
        requested_policy_oid="1.2.3.4.1",
        nonce=(1 << 127) + 12345,
    )
    assert bundle.rfc3161_request is True
    assert bundle.trusted_time_obtained is False
    assert verify_rfc3161_timestamp_request_bundle(bundle_path) == bundle
    emit_rfc3161_timestamp_request_der(
        request_bundle_path=bundle_path,
        output_der_path=request_path,
    )

    root, ca, config = make_tsa(tmp_path)
    reply(root, config, request_path, response_path)
    receipt = verify_rfc3161_timestamp_response(
        request_bundle_path=bundle_path,
        response_path=response_path,
        trust_anchor_bundle_path=ca,
        output_receipt_path=receipt_path,
        expected_policy_oid="1.2.3.4.1",
    )

    assert receipt.rfc3161_token is True
    assert receipt.certificate_chain_verified is True
    assert receipt.ess_signer_binding_verified is True
    assert receipt.independently_trusted_clock_proven is False
    assert receipt.hardware_clock_proven is False
    assert receipt.subject_sha256 == bundle.subject_sha256
    assert verify_rfc3161_timestamp_receipt(receipt_path) == receipt
    rendered = json.dumps(receipt.public_payload()).lower()
    assert str(subject).lower() not in rendered
    assert "private key" not in rendered


@pytest.mark.skipif(OPENSSL is None, reason="OpenSSL unavailable")
def test_wrong_trust_anchor_and_nonce_mismatch_fail_closed(tmp_path: Path):
    subject = tmp_path / "custody.json"
    subject.write_text("custody")
    first_bundle = tmp_path / "first.json"
    second_bundle = tmp_path / "second.json"
    first_request = tmp_path / "first.tsq"
    second_request = tmp_path / "second.tsq"
    response = tmp_path / "response.tsr"
    create_rfc3161_timestamp_request_bundle(
        owner_id="alice",
        custody_envelope_path=subject,
        output_bundle_path=first_bundle,
        requested_policy_oid="1.2.3.4.1",
        nonce=(1 << 127) + 1,
    )
    create_rfc3161_timestamp_request_bundle(
        owner_id="alice",
        custody_envelope_path=subject,
        output_bundle_path=second_bundle,
        requested_policy_oid="1.2.3.4.1",
        nonce=(1 << 127) + 2,
    )
    emit_rfc3161_timestamp_request_der(
        request_bundle_path=first_bundle,
        output_der_path=first_request,
    )
    emit_rfc3161_timestamp_request_der(
        request_bundle_path=second_bundle,
        output_der_path=second_request,
    )
    root, ca, config = make_tsa(tmp_path, name="primary")
    _other_root, other_ca, _other_config = make_tsa(tmp_path, name="other")
    reply(root, config, second_request, response)

    with pytest.raises(PermissionError, match="nonce"):
        verify_rfc3161_timestamp_response(
            request_bundle_path=first_bundle,
            response_path=response,
            trust_anchor_bundle_path=ca,
        )
    with pytest.raises(PermissionError, match="OpenSSL"):
        verify_rfc3161_timestamp_response(
            request_bundle_path=second_bundle,
            response_path=response,
            trust_anchor_bundle_path=other_ca,
        )


def test_rejected_response_and_bundle_tamper_fail_closed(tmp_path: Path):
    subject = tmp_path / "custody.json"
    subject.write_text("custody")
    bundle_path = tmp_path / "request.json"
    create_rfc3161_timestamp_request_bundle(
        owner_id="alice",
        custody_envelope_path=subject,
        output_bundle_path=bundle_path,
        nonce=(1 << 127) + 3,
    )
    raw = json.loads(bundle_path.read_text())
    raw["owner_id"] = "mallory"
    bundle_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="bundle_digest"):
        verify_rfc3161_timestamp_request_bundle(bundle_path)

    valid_bundle = tmp_path / "valid.json"
    create_rfc3161_timestamp_request_bundle(
        owner_id="alice",
        custody_envelope_path=subject,
        output_bundle_path=valid_bundle,
        nonce=(1 << 127) + 4,
    )
    status = tsp.PKIStatusInfo(
        {
            "status": "rejection",
            "status_string": ["not available"],
        }
    ).dump()

    def length(value: int) -> bytes:
        if value < 128:
            return bytes([value])
        encoded = value.to_bytes((value.bit_length() + 7) // 8, "big")
        return bytes([0x80 | len(encoded)]) + encoded

    response = tmp_path / "rejected.tsr"
    response.write_bytes(b"\x30" + length(len(status)) + status)
    trust = tmp_path / "trust.pem"
    trust.write_text("not needed")
    with pytest.raises(PermissionError, match="not granted"):
        verify_rfc3161_timestamp_response(
            request_bundle_path=valid_bundle,
            response_path=response,
            trust_anchor_bundle_path=trust,
        )


def test_no_overwrite_for_bundle_der_and_receipt(tmp_path: Path):
    subject = tmp_path / "custody.json"
    subject.write_text("custody")
    bundle_path = tmp_path / "request.json"
    bundle = create_rfc3161_timestamp_request_bundle(
        owner_id="alice",
        custody_envelope_path=subject,
        output_bundle_path=bundle_path,
        nonce=(1 << 127) + 5,
    )
    with pytest.raises(FileExistsError):
        create_rfc3161_timestamp_request_bundle(
            owner_id="alice",
            custody_envelope_path=subject,
            output_bundle_path=bundle_path,
            nonce=(1 << 127) + 5,
        )
    der = tmp_path / "request.tsq"
    emit_rfc3161_timestamp_request_der(
        request_bundle_path=bundle_path,
        output_der_path=der,
    )
    with pytest.raises(FileExistsError):
        emit_rfc3161_timestamp_request_der(
            request_bundle_path=bundle_path,
            output_der_path=der,
        )
    assert der.read_bytes() == bundle.request_der()
