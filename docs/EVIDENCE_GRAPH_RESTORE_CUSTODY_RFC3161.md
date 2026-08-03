# RFC 3161 interoperability for restore chain-of-custody evidence

Last updated: 2026-08-03

This boundary creates offline RFC 3161 timestamp requests for an external restore chain-of-custody envelope and verifies returned `TimeStampResp` DER against explicitly supplied certificate evidence.

It is separate from the repository's existing Ed25519 asserted-time attestation. An asserted-time receipt is never relabeled as an RFC 3161 token.

## Security model

The RFC 3161 request binds:

- the complete custody-envelope file through SHA-256;
- one 64–256-bit positive nonce;
- an optional requested timestamp policy OID;
- `certReq = true`, requiring signer-certificate inclusion;
- the owner scope and request construction time;
- the exact DER request digest.

The verifier requires:

- a terminal, granted RFC 3161 response;
- a CMS `SignedData` token containing `TSTInfo`;
- SHA-256 message imprint equality;
- exact nonce equality;
- requested and optionally operator-pinned policy equality;
- one signer and one unambiguous included signer certificate;
- signed CMS `contentType` and `messageDigest` attributes;
- ESSCertID or ESSCertIDv2 signer-certificate binding;
- a critical, timeStamping-only TSA extended-key-usage certificate;
- a non-CA TSA signer certificate valid at token generation time;
- OpenSSL RFC 3161 verification against an explicitly supplied trust-anchor bundle;
- optional intermediate certificates and optional CRL evidence;
- a bounded future-time ceiling;
- atomic no-overwrite receipt publication.

The generated verification receipt contains only digests, bounded certificate and token metadata, policy, serial and generation time. It contains no request subject bytes, private key, raw path or certificate body.

## Prerequisites

Python runtime dependencies:

```text
cryptography==49.0.0
asn1crypto>=1.5,<2
```

The operator host must also provide an OpenSSL executable with `ts -verify` support. Missing or invalid OpenSSL fails closed.

## 1. Create an offline request bundle

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_rfc3161.py \
  create-request \
  --owner-id alice \
  --custody-envelope data/restore-custody.signed.json \
  --output-bundle data/restore-custody.rfc3161-request.json \
  --policy-oid 1.2.3.4.1
```

The JSON bundle stores the exact DER request in canonical base64 together with its digest and request scope. The command does not contact a timestamp authority.

The public summary explicitly reports:

```text
rfc3161_request=true
trusted_time_obtained=false
network_request_performed=false
contains_private_key_material=false
contains_raw_subject_content=false
```

## 2. Emit the DER request

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_rfc3161.py \
  emit-request \
  --request-bundle data/restore-custody.rfc3161-request.json \
  --output-der data/restore-custody.tsq
```

The output is created atomically and never overwrites an existing path.

Transport `restore-custody.tsq` to the selected TSA through a separately governed channel. This repository does not implement an implicit HTTP client, endpoint allowlist, credential store or retry loop for TSA transport.

## 3. Verify a returned response

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_rfc3161.py \
  verify-response \
  --request-bundle data/restore-custody.rfc3161-request.json \
  --response data/restore-custody.tsr \
  --trust-anchor-bundle config/tsa-roots.pem \
  --untrusted-bundle config/tsa-intermediates.pem \
  --crl-bundle config/tsa-crls.pem \
  --expected-policy-oid 1.2.3.4.1 \
  --output-receipt data/restore-custody.rfc3161-receipt.json
```

`--untrusted-bundle` and `--crl-bundle` are optional. When a CRL bundle is supplied, the receipt records its digest and sets `revocation_checked=true`. Without CRL evidence, `revocation_checked=false` is retained explicitly.

OpenSSL verification is isolated from ambient certificate configuration by using the supplied trust bundle, an empty CA directory and an explicit file-backed CA store.

## 4. Reconstruct and validate a receipt offline

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_rfc3161.py \
  verify-receipt \
  --receipt data/restore-custody.rfc3161-receipt.json
```

This command validates the strict receipt schema and deterministic receipt digest. It does not re-read the DER token or certificate chain; use `verify-response` whenever the source response and trust evidence must be revalidated.

## Failure behavior

Verification refuses:

- invalid or non-canonical request bundles;
- changed request DER;
- malformed or nonterminal responses;
- rejection responses;
- missing or non-`SignedData` tokens;
- wrong imprint, nonce or policy;
- absent or ambiguous signer certificates;
- invalid CMS signed attributes;
- failed ESS signer-certificate binding;
- invalid TSA EKU, CA status or certificate validity;
- untrusted or invalid certificate chains;
- future generation times beyond the configured ceiling;
- missing OpenSSL;
- verifier timeout;
- redirected, replaced, growing or oversized input files;
- existing output paths.

CLI failures are generic and do not expose private paths or verifier diagnostics.

## Exact trust claims

A successful receipt establishes that:

- the supplied RFC 3161 response is cryptographically bound to the exact request imprint and nonce;
- the token policy matches the required policy;
- the included TSA signer certificate matches the CMS signer binding;
- the CMS signature and certificate path validate against the supplied trust evidence;
- the token contains the recorded generation time and serial.

A successful receipt does **not** independently establish:

- that the supplied trust anchor represents the intended institution;
- that the TSA clock was accurate or independently audited;
- that the timestamp came from a hardware-backed clock;
- current revocation status when no governed CRL evidence is supplied;
- RFC 3161 transport authenticity outside the verified token;
- scientific correctness of the underlying restore evidence.

The receipt therefore keeps:

```text
independently_trusted_clock_proven=false
hardware_clock_proven=false
```

Institutional TSA onboarding, trust-anchor governance, external certificate-chain rotation, live revocation services and hardware-backed time remain separate controls.

## Verification boundary

Focused reconstructed execution used a real locally generated CA/TSA chain and OpenSSL-produced `TimeStampResp`:

```text
6 passed
```

The focused tests cover the valid round trip, request inspection and DER emission, wrong nonce, wrong trust anchor, request-bundle tampering, rejection response handling, no-overwrite behavior and generic CLI failures.

This result is not a complete exact-current repository test or platform matrix. Ruff, full pytest, Windows, containers, production TSA interoperability, OCSP, network transport and hardware-backed time remain open.
