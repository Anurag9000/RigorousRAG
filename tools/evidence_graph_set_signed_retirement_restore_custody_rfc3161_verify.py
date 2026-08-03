"""Strict RFC 3161 response verification for custody timestamp requests."""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from asn1crypto import cms, tsp, x509 as asn1_x509
from cryptography import x509
from cryptography.x509.oid import ExtendedKeyUsageOID

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _identifier,
    _integer,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_contracts import (
    GRANTED_STATUS,
    MAX_INPUT_BYTES,
    MAX_RESPONSE_BYTES,
    MAX_VERIFIER_OUTPUT_BYTES,
    NONTERMINAL_STATUS,
    Rfc3161TimeStampResp,
    Rfc3161TimestampVerificationReceipt,
    SCHEMA_VERSION,
    canonical_digest,
    optional_oid,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_io import (
    read_regular,
    verify_rfc3161_timestamp_request_bundle,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    _atomic_create,
    _canonical_bytes,
    _path,
)


def _certificate_choices(value: cms.SignedData) -> tuple[asn1_x509.Certificate, ...]:
    certificates = value["certificates"]
    if certificates.native is None:
        return ()
    rendered: list[asn1_x509.Certificate] = []
    for choice in certificates:
        if choice.name == "certificate":
            rendered.append(choice.chosen)
    return tuple(rendered)


def _select_signer_certificate(
    signed_data: cms.SignedData,
    signer_info: cms.SignerInfo,
) -> asn1_x509.Certificate:
    certificates = _certificate_choices(signed_data)
    if not certificates:
        raise ValueError("RFC 3161 token does not contain signer certificates.")
    sid = signer_info["sid"]
    matches: list[asn1_x509.Certificate] = []
    if sid.name == "issuer_and_serial_number":
        selected = sid.chosen
        for certificate in certificates:
            if (
                certificate.serial_number == selected["serial_number"].native
                and certificate.issuer == selected["issuer"]
            ):
                matches.append(certificate)
    elif sid.name == "subject_key_identifier":
        identifier = sid.chosen.native
        for certificate in certificates:
            if certificate.key_identifier == identifier:
                matches.append(certificate)
    else:
        raise ValueError("RFC 3161 signer identifier is unsupported.")
    if len(matches) != 1:
        raise ValueError("RFC 3161 token signer certificate is ambiguous or missing.")
    return matches[0]


def _attributes(signer_info: cms.SignerInfo) -> dict[str, list[Any]]:
    attrs = signer_info["signed_attrs"]
    if attrs.native is None:
        raise ValueError("RFC 3161 signer has no signed attributes.")
    rendered: dict[str, list[Any]] = {}
    for attr in attrs:
        name = attr["type"].native
        rendered.setdefault(name, []).append(attr["values"])
    return rendered


def _single_attribute(attrs: dict[str, list[Any]], name: str) -> Any:
    values = attrs.get(name, [])
    if len(values) != 1 or len(values[0]) != 1:
        raise ValueError(
            f"RFC 3161 signed attribute {name} is missing or repeated."
        )
    return values[0][0]


def _verify_ess_binding(
    *,
    signer_info: cms.SignerInfo,
    certificate_der: bytes,
) -> None:
    attrs = _attributes(signer_info)
    matched = False
    if "signing_certificate_v2" in attrs:
        value = _single_attribute(attrs, "signing_certificate_v2")
        certs = value["certs"]
        if len(certs) < 1:
            raise ValueError("SigningCertificateV2 contains no certificate ID.")
        cert_id = certs[0]
        algorithm = cert_id["hash_algorithm"]["algorithm"].native or "sha256"
        if algorithm not in hashlib.algorithms_available:
            raise ValueError("SigningCertificateV2 hash algorithm is unsupported.")
        matched = matched or (
            hashlib.new(algorithm, certificate_der).digest()
            == cert_id["cert_hash"].native
        )
    if "signing_certificate" in attrs:
        value = _single_attribute(attrs, "signing_certificate")
        certs = value["certs"]
        if len(certs) < 1:
            raise ValueError("SigningCertificate contains no certificate ID.")
        matched = matched or (
            hashlib.sha1(certificate_der).digest() == certs[0]["cert_hash"].native
        )
    if not matched:
        raise ValueError("RFC 3161 ESS signer-certificate binding failed.")


def _verify_tsa_certificate(
    *,
    certificate_der: bytes,
    generated_at: dt.datetime,
) -> tuple[str, str, str]:
    certificate = x509.load_der_x509_certificate(certificate_der)
    try:
        eku = certificate.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
    except x509.ExtensionNotFound as exc:
        raise ValueError("TSA certificate lacks extended key usage.") from exc
    if not eku.critical or set(eku.value) != {ExtendedKeyUsageOID.TIME_STAMPING}:
        raise ValueError("TSA certificate EKU must be critical and timeStamping-only.")
    try:
        constraints = certificate.extensions.get_extension_for_class(
            x509.BasicConstraints
        )
    except x509.ExtensionNotFound as exc:
        raise ValueError("TSA certificate lacks basic constraints.") from exc
    if constraints.value.ca:
        raise ValueError("TSA signer certificate may not be a CA certificate.")
    current = generated_at.astimezone(dt.timezone.utc)
    if (
        current < certificate.not_valid_before_utc
        or current > certificate.not_valid_after_utc
    ):
        raise ValueError("TSA certificate was not valid at token generation time.")
    public_key_name = type(certificate.public_key()).__name__
    return (
        hashlib.sha256(certificate_der).hexdigest(),
        format(certificate.serial_number, "x"),
        _identifier(public_key_name.lower(), "signer_public_key_algorithm", 100),
    )


def _resolve_openssl(value: str) -> str:
    selected = _identifier(value, "openssl_binary", 4096)
    candidate = shutil.which(selected) if os.path.sep not in selected else selected
    if candidate is None:
        raise RuntimeError("OpenSSL executable is unavailable.")
    path = Path(candidate).resolve()
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or not os.access(path, os.X_OK):
        raise RuntimeError("OpenSSL executable is invalid.")
    return str(path)


def _copy_temp(directory: str, name: str, payload: bytes) -> str:
    path = os.path.join(directory, name)
    with open(path, "xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _bounded_verifier_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "version"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
        check=False,
        env={"PATH": os.environ.get("PATH", "")},
    )
    output = result.stdout[:MAX_VERIFIER_OUTPUT_BYTES]
    if result.returncode != 0 or not output:
        raise RuntimeError("OpenSSL version discovery failed.")
    return hashlib.sha256(output).hexdigest()


def _run_openssl_verify(
    *,
    executable: str,
    request_der: bytes,
    response_der: bytes,
    trust_anchor: bytes,
    untrusted: bytes | None,
    crl: bytes | None,
    timeout_seconds: int,
) -> str:
    timeout = _integer(timeout_seconds, "timeout_seconds", 1, 300)
    with tempfile.TemporaryDirectory(prefix="rigorousrag-rfc3161-") as directory:
        request_path = _copy_temp(directory, "request.tsq", request_der)
        response_path = _copy_temp(directory, "response.tsr", response_der)
        trust_payload = trust_anchor
        if crl is not None:
            trust_payload = trust_anchor.rstrip() + b"\n" + crl.lstrip()
        trust_path = _copy_temp(directory, "trust.pem", trust_payload)
        empty_ca_path = os.path.join(directory, "empty-ca-path")
        os.mkdir(empty_ca_path)
        command = [
            executable,
            "ts",
            "-verify",
            "-queryfile",
            request_path,
            "-in",
            response_path,
            "-CAfile",
            trust_path,
            "-CApath",
            empty_ca_path,
            "-CAstore",
            f"file:{trust_path}",
            "-purpose",
            "timestampsign",
            "-check_ss_sig",
        ]
        if untrusted is not None:
            command.extend(
                ["-untrusted", _copy_temp(directory, "untrusted.pem", untrusted)]
            )
        if crl is not None:
            command.append("-crl_check_all")
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "SSL_CERT_FILE": trust_path,
                    "SSL_CERT_DIR": empty_ca_path,
                },
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("RFC 3161 verification timed out.") from exc
        output = result.stdout[:MAX_VERIFIER_OUTPUT_BYTES]
        if result.returncode != 0 or b"Verification: OK" not in output:
            raise PermissionError("RFC 3161 OpenSSL verification failed.")
    return _bounded_verifier_version(executable)


def verify_rfc3161_timestamp_response(
    *,
    request_bundle_path: str | os.PathLike[str],
    response_path: str | os.PathLike[str],
    trust_anchor_bundle_path: str | os.PathLike[str],
    output_receipt_path: str | os.PathLike[str] | None = None,
    untrusted_bundle_path: str | os.PathLike[str] | None = None,
    crl_bundle_path: str | os.PathLike[str] | None = None,
    expected_policy_oid: str | None = None,
    openssl_binary: str = "openssl",
    timeout_seconds: int = 30,
    now: float | None = None,
    maximum_future_seconds: float = 300.0,
) -> Rfc3161TimestampVerificationReceipt:
    bundle = verify_rfc3161_timestamp_request_bundle(request_bundle_path)
    response_der = read_regular(
        response_path, label="response_path", maximum=MAX_RESPONSE_BYTES
    )
    trust_anchor = read_regular(
        trust_anchor_bundle_path,
        label="trust_anchor_bundle_path",
        maximum=MAX_INPUT_BYTES,
    )
    untrusted = None if untrusted_bundle_path is None else read_regular(
        untrusted_bundle_path,
        label="untrusted_bundle_path",
        maximum=MAX_INPUT_BYTES,
    )
    crl = None if crl_bundle_path is None else read_regular(
        crl_bundle_path,
        label="crl_bundle_path",
        maximum=MAX_INPUT_BYTES,
    )
    try:
        response = Rfc3161TimeStampResp.load(response_der, strict=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("RFC 3161 response DER is invalid.") from exc
    status = response["status"]["status"].native
    if status in NONTERMINAL_STATUS:
        raise RuntimeError("RFC 3161 response is not terminal.")
    if status not in GRANTED_STATUS:
        raise PermissionError("RFC 3161 response was not granted.")
    token = response["time_stamp_token"]
    if token.native is None or token["content_type"].native != "signed_data":
        raise ValueError("RFC 3161 response lacks a SignedData token.")
    signed_data = token["content"]
    if signed_data["encap_content_info"]["content_type"].native != "tst_info":
        raise ValueError("RFC 3161 token content type is not TSTInfo.")
    content = signed_data["encap_content_info"]["content"]
    if content.native is None:
        raise ValueError("RFC 3161 token lacks TSTInfo content.")
    tst_info = content.parsed
    if not isinstance(tst_info, tsp.TSTInfo):
        raise ValueError("RFC 3161 token TSTInfo is invalid.")
    native = tst_info.native
    imprint = native["message_imprint"]
    if imprint["hash_algorithm"]["algorithm"] != "sha256":
        raise ValueError("RFC 3161 token imprint algorithm is not SHA-256.")
    if imprint["hashed_message"].hex() != bundle.subject_sha256:
        raise PermissionError("RFC 3161 message imprint differs from custody evidence.")
    nonce = native["nonce"]
    if nonce is None or str(nonce) != bundle.nonce_decimal:
        raise PermissionError("RFC 3161 response nonce differs from request.")
    policy = optional_oid(native["policy"], "policy_oid")
    if policy is None:
        raise ValueError("RFC 3161 token policy is missing.")
    if bundle.requested_policy_oid is not None and policy != bundle.requested_policy_oid:
        raise PermissionError("RFC 3161 token policy differs from request.")
    expected_policy = optional_oid(expected_policy_oid, "expected_policy_oid")
    if expected_policy is not None and policy != expected_policy:
        raise PermissionError("RFC 3161 token policy differs from expected policy.")
    generated_at = native["gen_time"]
    if not isinstance(generated_at, dt.datetime) or generated_at.tzinfo is None:
        raise ValueError("RFC 3161 token generation time is invalid.")
    generated_at = generated_at.astimezone(dt.timezone.utc)
    current = _timestamp(time.time() if now is None else now, "now")
    future = float(maximum_future_seconds)
    if not (0.0 <= future <= 86_400.0):
        raise ValueError("maximum_future_seconds is invalid.")
    generated_unix = generated_at.timestamp()
    if generated_unix > current + future:
        raise PermissionError("RFC 3161 generation time is too far in the future.")
    serial_number = native["serial_number"]
    if not isinstance(serial_number, int) or serial_number <= 0:
        raise ValueError("RFC 3161 token serial number is invalid.")
    if len(signed_data["signer_infos"]) != 1:
        raise ValueError("RFC 3161 token must contain exactly one signer.")
    signer_info = signed_data["signer_infos"][0]
    attrs = _attributes(signer_info)
    if _single_attribute(attrs, "content_type").native != "tst_info":
        raise ValueError("RFC 3161 signed content-type attribute differs.")
    digest_algorithm = signer_info["digest_algorithm"]["algorithm"].native
    if digest_algorithm not in hashlib.algorithms_available:
        raise ValueError("RFC 3161 signer digest algorithm is unsupported.")
    message_digest = _single_attribute(attrs, "message_digest").native
    if message_digest != hashlib.new(digest_algorithm, content.contents).digest():
        raise PermissionError("RFC 3161 signed message digest differs.")
    signer_certificate = _select_signer_certificate(signed_data, signer_info)
    certificate_der = signer_certificate.dump()
    _verify_ess_binding(
        signer_info=signer_info,
        certificate_der=certificate_der,
    )
    certificate_sha256, certificate_serial_hex, public_key_algorithm = (
        _verify_tsa_certificate(
            certificate_der=certificate_der,
            generated_at=generated_at,
        )
    )
    executable = _resolve_openssl(openssl_binary)
    verifier_version_digest = _run_openssl_verify(
        executable=executable,
        request_der=bundle.request_der(),
        response_der=response_der,
        trust_anchor=trust_anchor,
        untrusted=untrusted,
        crl=crl,
        timeout_seconds=timeout_seconds,
    )
    accuracy = native.get("accuracy") or {}
    stable = {
        "scope": "rigorousrag-restore-custody-rfc3161-receipt-v1",
        "owner_id": bundle.owner_id,
        "request_bundle_digest": bundle.bundle_digest,
        "request_sha256": bundle.request_sha256,
        "subject_sha256": bundle.subject_sha256,
        "response_sha256": hashlib.sha256(response_der).hexdigest(),
        "token_sha256": hashlib.sha256(token.dump()).hexdigest(),
        "status": status,
        "policy_oid": policy,
        "message_imprint_sha256": imprint["hashed_message"].hex(),
        "nonce_sha256": bundle.nonce_sha256,
        "serial_decimal": str(serial_number),
        "generated_at_rfc3339": generated_at.isoformat().replace("+00:00", "Z"),
        "generated_at_unix": generated_unix,
        "accuracy_seconds": accuracy.get("seconds"),
        "accuracy_millis": accuracy.get("millis"),
        "accuracy_micros": accuracy.get("micros"),
        "ordering": bool(native.get("ordering", False)),
        "signer_certificate_sha256": certificate_sha256,
        "signer_certificate_serial_hex": certificate_serial_hex,
        "signer_public_key_algorithm": public_key_algorithm,
        "signature_algorithm": signer_info["signature_algorithm"]["algorithm"].native,
        "digest_algorithm": digest_algorithm,
        "trust_anchor_bundle_sha256": hashlib.sha256(trust_anchor).hexdigest(),
        "untrusted_bundle_sha256": (
            None if untrusted is None else hashlib.sha256(untrusted).hexdigest()
        ),
        "crl_bundle_sha256": (
            None if crl is None else hashlib.sha256(crl).hexdigest()
        ),
        "verifier_version_sha256": verifier_version_digest,
        "schema_version": SCHEMA_VERSION,
    }
    receipt = Rfc3161TimestampVerificationReceipt(
        **{key: value for key, value in stable.items() if key != "scope"},
        receipt_digest=canonical_digest(stable),
        revocation_checked=crl is not None,
    )
    if output_receipt_path is not None:
        _atomic_create(
            _path(output_receipt_path, label="output_receipt_path"),
            _canonical_bytes(receipt.public_payload()) + b"\n",
        )
    return receipt


__all__ = ["verify_rfc3161_timestamp_response"]
