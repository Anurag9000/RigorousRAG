# Governed Ed25519 signatures for restore chain-of-custody evidence

Last updated: 2026-08-03

RigorousRAG can now place a public-key digital signature over a complete external restore chain-of-custody manifest. The signed envelope preserves the existing custody manifest, binds it to one owner-scoped governed key ID, and can optionally bind the resulting signature envelope to a verified RFC 3161 timestamp receipt.

The implementation uses Ed25519 only. It does not negotiate algorithms, encrypt data, import custody history, restore a database, delete evidence, or store private signing keys.

## 1. Security model

The signed envelope proves that the holder of the corresponding Ed25519 private key signed the exact canonical custody manifest represented by the envelope.

A governed verification additionally proves that:

- the owner and key ID exist in the repository-owned signer-key registry;
- the public-key bytes and SHA-256 fingerprint match the signed envelope;
- the key was active and within its configured validity window for current verification; or
- for a retired key, a newly reverified governed RFC 3161 timestamp places the signature inside the key's historical validity interval.

The signature does **not**, by itself, prove the real-world identity of the key holder. Deployment-grade identity requires an independently trusted key-distribution, certificate, directory, transparency, KMS, HSM, or organizational governance process.

## 2. Durable public-key registry

Configure only the public-key registry:

```bash
EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_KEY_DB_PATH=data/evidence_graph_restore_custody_signer_keys.sqlite3
```

Private-key paths are deliberately absent from `.env.example`. The registry stores only:

- normalized owner ID;
- key ID;
- algorithm (`ed25519`);
- raw public key encoded as canonical base64;
- public-key SHA-256 fingerprint;
- validity window;
- active or retired state;
- process-owned actor-binding digests and timestamps;
- deterministic record digest.

The registry includes parent-directory and database-inode identity checks and refuses symlink/reparse-point paths.

## 3. Generate a key outside the registry

One OpenSSL example is:

```bash
openssl genpkey -algorithm Ed25519 -out custody-signing-private.pem
openssl pkey -in custody-signing-private.pem -pubout -out custody-signing-public.pem
chmod 600 custody-signing-private.pem
```

The signing command accepts an unencrypted PEM PKCS#8 Ed25519 private key. On POSIX, any group or other permission bit causes a fail-closed refusal.

The application never copies, serializes, logs, registers, or returns the private key. Operators should use a short-lived protected file or a future HSM/KMS adapter and should securely remove transient key material according to their deployment policy.

## 4. Register a public key

A process-owned reviewer actor must already be configured through the established actor-binding boundary.

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signature_keys.py register \
  --owner-id alice \
  --key-id custody-ed25519-2026-01 \
  --public-key-path custody-signing-public.pem \
  --valid-from 1785715200 \
  --confirm-key-id custody-ed25519-2026-01
```

An optional `--valid-until` creates a bounded validity window.

Registration is idempotent only when owner, key ID, public-key bytes, fingerprint, and validity window are identical. Reusing the same owner/key ID for different material or scope is an identity collision and fails closed. A retired key cannot be reactivated.

Read-only inspection:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signature_keys.py status \
  --owner-id alice \
  --key-id custody-ed25519-2026-01

python scripts/evidence_graph_set_signed_retirement_restore_custody_signature_keys.py list \
  --owner-id alice
```

Outputs contain public-key fingerprints and governance digests, but no raw paths or private-key material.

## 5. Sign a complete custody manifest

The input must already pass complete chain-of-custody verification.

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signature.py sign \
  --owner-id alice \
  --key-id custody-ed25519-2026-01 \
  --manifest restore-custody-manifest.json \
  --private-key-path custody-signing-private.pem \
  --output restore-custody-signed.json
```

Before signing, the command:

1. resolves the owner/key record from the governed registry;
2. requires the record to be active and currently valid;
3. verifies the complete custody manifest;
4. derives the public key from the private key;
5. requires its fingerprint to equal the governed record;
6. signs a canonical JSON payload containing the full manifest, owner, key ID, algorithm, public-key fingerprint, creation time, and schema version;
7. writes the envelope atomically without overwriting an existing destination.

The signature itself is 64 Ed25519 bytes represented as canonical base64. The envelope digest commits both the signed payload and signature.

## 6. Offline verification with a pinned public key

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signature.py verify \
  restore-custody-signed.json \
  --public-key-path custody-signing-public.pem \
  --expected-owner-id alice \
  --expected-key-id custody-ed25519-2026-01
```

This verification requires no live registry. It reconstructs and validates the manifest, validates the envelope digest, checks the pinned key fingerprint, and verifies the Ed25519 signature.

Offline verification proves cryptographic integrity relative to the supplied public key. It does not prove that the supplied key was organizationally authorized.

## 7. Governed current verification

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signature.py verify-governed \
  restore-custody-signed.json \
  --owner-id alice
```

This path resolves the key from the governed registry and requires it to be active and currently valid. A retired key is not accepted merely because its signature is mathematically valid.

## 8. Bind an RFC 3161 timestamp

First obtain and verify an RFC 3161 receipt for the SHA-256 digest of the canonical signed-envelope file through the existing timestamp workflow. Then bind the receipt:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signature.py bind-timestamp \
  --signed-envelope restore-custody-signed.json \
  --receipt restore-custody-rfc3161-receipt.json \
  --public-key-path custody-signing-public.pem \
  --expected-key-id custody-ed25519-2026-01 \
  --output restore-custody-signed-timestamped.json
```

The timestamped envelope commits:

- the complete signed custody envelope;
- the RFC 3161 verification receipt;
- the SHA-256 digest timestamped by that receipt;
- a deterministic binding digest.

The command verifies the signature and receipt binding before publishing the output and never overwrites an existing destination.

Basic offline structural verification is available with:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signature.py verify-timestamped \
  restore-custody-signed-timestamped.json \
  --public-key-path custody-signing-public.pem \
  --expected-owner-id alice \
  --expected-key-id custody-ed25519-2026-01
```

This checks the embedded receipt's internal integrity and subject binding. It does not replace governed TSA trust verification.

## 9. Governed historical verification after key retirement

A key can be retired with exact confirmation:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signature_keys.py retire \
  --owner-id alice \
  --key-id custody-ed25519-2026-01 \
  --confirm-key-id custody-ed25519-2026-01
```

Retirement is monotonic. The public key and historical governance record remain available for verification; signing and current-time verification are disabled.

To verify a historical signature made before retirement, use the governed timestamp path. It **re-verifies** the RFC 3161 response in the same process using the governed TSA profile and then checks that the trusted timestamp lies inside the signer's validity interval:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signature.py verify-governed-timestamped \
  restore-custody-signed-timestamped.json \
  --owner-id alice \
  --tsa-profile-id external-tsa-2026 \
  --request-bundle timestamp-request-bundle.json \
  --response timestamp-response.tsr \
  --trust-anchor-bundle tsa-root.pem \
  --untrusted-bundle tsa-intermediates.pem \
  --crl-bundle tsa-crls.pem
```

The embedded receipt digest and timestamped subject must exactly match the newly governed verification result. A stale embedded receipt alone cannot authorize a retired key.

## 10. Rotation procedure

A conservative rotation is:

1. generate a new Ed25519 key pair outside the service;
2. register the new public key under a new key ID and validity window;
3. start new signatures with the new key;
4. preserve the old public-key record and timestamped signed envelopes;
5. ensure important old signatures have governed RFC 3161 evidence;
6. retire the old key with exact confirmation;
7. verify old signatures only through the governed historical timestamp path.

Do not reuse key IDs for replacement material. Do not delete retired public-key history when custody evidence still depends on it.

## 11. Permanent boundaries

- Only Ed25519 is supported.
- The registry stores public keys, never private keys.
- No command exports private material.
- Signing does not restore, import, merge, overwrite, or delete repository state.
- Signature output uses atomic no-overwrite creation.
- A local public-key registry is governance evidence, not a public certificate authority.
- Basic RFC 3161 receipt binding is not equivalent to governed TSA trust verification.
- Retired-key historical verification requires a newly governed RFC 3161 verification in the same process.
- HSM/KMS-backed signing and externally distributed signer certificates remain future deployment work.

## 12. Focused verification evidence

The implemented slice was executed in the reconstructed focused workspace after the final module split:

```text
8 passed
```

Focused compilation also passed. The checks cover public-key registration/replay/collision, retirement, database identity replacement, protected private-key loading, governed signing, offline verification, tamper refusal, RFC 3161 binding, historical retired-key verification, CLI path/key secrecy, atomic no-overwrite behavior, and canonical runtime caching.

This is not a complete exact-current repository pytest, Ruff, platform, container, HSM, external-PKI, or independent-process test matrix. Release readiness is not claimed.
