# Wave 5 status addendum — Ed25519 custody signatures

Last updated: 2026-08-03

## Implemented

- [x] `cryptography==49.0.0` runtime dependency.
- [x] Ed25519 PKCS8 PEM private-key loading.
- [x] Ed25519 public-key PEM loading.
- [x] Protected non-redirecting private-key files.
- [x] POSIX group/world private-key permission refusal.
- [x] Explicit signer key ID.
- [x] SHA-256 fingerprint over raw Ed25519 public-key bytes.
- [x] Canonical JSON signing payload.
- [x] Canonical base64 encoding of the 64-byte signature.
- [x] Atomic no-overwrite signed-envelope publication.
- [x] Offline public-key signature verification.
- [x] Optional exact key-ID and public-key-fingerprint pinning.
- [x] Wrong-key, changed-payload, and changed-signature refusal.
- [x] No key generation, private-key export, restore, import, overwrite, or deletion command.

## Operator commands

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_export.py sign ...
python scripts/evidence_graph_set_signed_retirement_restore_custody_export.py verify-signature ...
```

## Focused verification

A reconstructed Ed25519 harness passed:

```text
3 focused checks passed
```

Covered:

1. sign/verify round trip;
2. explicit key-ID and fingerprint pinning;
3. atomic no-overwrite output;
4. wrong-public-key refusal;
5. signature-tamper refusal;
6. broad private-key permission refusal.

The local environment carried `cryptography` 46.0.4; `main` pins official stable 49.0.0. The exercised Ed25519 and serialization interfaces are the same documented APIs. Repository-native tests additionally cover invalid key material and privacy-safe CLI summaries.

This is not a complete exact-current repository run.

## Still open

- [ ] Complete exact-current repository pytest and coverage.
- [ ] Ruff and full-tree compilation.
- [ ] Trusted timestamp authority integration.
- [ ] Durable signer-key registry and rotation policy.
- [ ] Issuer/key overlap windows.
- [ ] Hardware-backed private-key custody.
- [ ] External public-key transparency publication.
- [ ] Independent-process signing/output races.
- [ ] Windows and container signing matrices.

## Permanent non-claims

- Ed25519 proves possession of the signing private key, not scientific correctness.
- A signature does not prove creation time without a trusted timestamp.
- A fingerprint is not an externally governed identity by itself.
- Software-enforced file permissions are not hardware-backed key custody.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
