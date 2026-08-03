# Wave 5 RFC 3161 interoperability verification boundary

Last updated: 2026-08-03

This ledger records the exact implementation and execution boundary for offline RFC 3161 request construction and response verification over external restore chain-of-custody evidence.

## Implemented code boundary

Code and test head before this ledger commit: `f8e14b18b48eccb3af0cb619312b6978666be9e4`.

Implemented modules:

- `tools/evidence_graph_set_signed_retirement_restore_custody_rfc3161_contracts.py`
- `tools/evidence_graph_set_signed_retirement_restore_custody_rfc3161_io.py`
- `tools/evidence_graph_set_signed_retirement_restore_custody_rfc3161_verify.py`
- `tools/evidence_graph_set_signed_retirement_restore_custody_rfc3161.py`
- `tools/evidence_graph_set_signed_retirement_restore_custody_rfc3161_cli.py`
- `scripts/evidence_graph_set_signed_retirement_restore_custody_rfc3161.py`

Repository-native focused contracts:

- `tests/unit/test_evidence_graph_set_signed_retirement_restore_custody_rfc3161.py`
- `tests/unit/test_evidence_graph_set_signed_retirement_restore_custody_rfc3161_cli.py`

Runtime dependency added:

```text
asn1crypto>=1.5,<2
```

The system OpenSSL executable remains an explicit host prerequisite rather than a Python package dependency.

## Executed focused evidence

Executed in the reconstructed focused workspace containing the exact split RFC 3161 implementation:

```text
python -m compileall -q \
  tools/evidence_graph_set_signed_retirement_restore_custody_rfc3161*.py \
  scripts/evidence_graph_set_signed_retirement_restore_custody_rfc3161.py \
  tests/unit/test_rfc3161*.py

python -m pytest -q \
  tests/unit/test_rfc3161.py \
  tests/unit/test_rfc3161_cli.py
```

Result:

```text
6 passed
```

Focused compilation passed.

Ruff was unavailable in the execution environment and is not claimed.

## Real cryptographic integration exercised

The focused integration test generated:

1. a local RSA root certificate and private key;
2. a non-CA TSA signer certificate;
3. critical timeStamping-only extended key usage;
4. an RFC 3161 SHA-256 request with nonce, policy and `certReq=true`;
5. an OpenSSL-produced `TimeStampResp` containing the TSA certificate;
6. a pinned trust-anchor verification receipt.

The successful path exercised real OpenSSL TSA response production and `openssl ts -verify` rather than mocking the standards boundary.

## Verified request properties

- bounded descriptor-safe custody-envelope read;
- SHA-256 message imprint;
- positive 64–256-bit nonce;
- optional requested policy OID;
- `certReq=true`;
- exact canonical DER regeneration;
- owner and subject-size scope;
- deterministic request and bundle digests;
- atomic no-overwrite bundle publication;
- atomic no-overwrite DER publication;
- no implicit network request.

## Verified response properties

- strict DER `TimeStampResp` parsing;
- terminal granted status;
- refusal of rejection and non-granted status;
- CMS `SignedData` and `TSTInfo` content types;
- SHA-256 message-imprint equality;
- exact nonce equality;
- requested and explicitly expected policy equality;
- bounded future generation time;
- positive serial number;
- exactly one CMS signer;
- signed `contentType` and `messageDigest` attributes;
- unambiguous included signer certificate;
- ESSCertID or ESSCertIDv2 certificate binding;
- critical timeStamping-only TSA EKU;
- non-CA TSA basic constraints;
- TSA certificate validity at token generation time;
- OpenSSL signature and pinned certificate-chain verification;
- optional intermediate-certificate bundle;
- optional CRL evidence and explicit revocation flag;
- atomic no-overwrite verification receipt;
- strict digest-bound offline receipt reconstruction.

## Fault and refusal coverage

The focused tests cover:

- wrong response nonce;
- wrong trust anchor;
- request-bundle scope tampering;
- RFC 3161 rejection response;
- repeated request-bundle output;
- repeated DER output;
- missing CLI input;
- generic path-free CLI failure output.

The implementation additionally contains fail-closed checks for malformed DER, absent TSTInfo, wrong imprint/policy, missing signer certificates, repeated signed attributes, failed ESS binding, invalid TSA EKU/basic constraints/validity, OpenSSL timeout or unavailability, oversized/redirected/replaced input files and existing receipt outputs.

## Distinct timestamp evidence types

This implementation does not modify the existing Ed25519 asserted-time attestation type.

- The Ed25519 receipt is a repository-governed authority assertion of a recorded time.
- The RFC 3161 receipt verifies an actual RFC 3161 token against the exact request and supplied certificate evidence.

Neither type is silently promoted or converted into the other.

## Exact non-claims

A valid RFC 3161 verification receipt does not independently prove:

- institutional identity of the supplied trust anchor;
- accuracy or audit status of the TSA clock;
- hardware-backed signing or clock evidence;
- current revocation when no governed CRL evidence is supplied;
- authenticity of any external request transport beyond the signed token;
- scientific correctness of the custody evidence.

The receipt therefore permanently records:

```text
independently_trusted_clock_proven=false
hardware_clock_proven=false
```

## Exact-current verification still required

- complete repository pytest and coverage;
- Ruff;
- full-tree compilation from an unchanged current checkout;
- production external TSA interoperability;
- governed online request transport;
- OCSP and long-term revocation evidence;
- Windows and container matrices;
- OpenSSL version matrix;
- malformed/fuzzed ASN.1 corpus;
- network, storage and directory-fsync fault injection;
- HSM/KMS and hardware-clock integration.

No exact-current CI success or release readiness is claimed.
