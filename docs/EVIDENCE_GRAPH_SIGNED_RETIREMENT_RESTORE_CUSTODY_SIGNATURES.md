# Ed25519 signatures for external restore custody manifests

Last updated: 2026-08-03

This runbook covers public-key signatures over already verified external restore chain-of-custody manifests.

The repository pins:

```text
cryptography==49.0.0
```

The implementation uses the package's Ed25519 signing and PEM serialization APIs. It does not generate keys, export private keys, contact a timestamp service, rotate keys, or integrate a hardware security module.

## 1. Provision keys outside the application

Supply an existing Ed25519 private key in unencrypted PKCS8 PEM format and its corresponding public key in PEM format.

On POSIX, the private-key file must not grant group or world permissions:

```bash
chmod 600 custody-ed25519-private.pem
```

The application refuses:

- symlink or reparse-point key paths;
- non-regular key files;
- broad POSIX private-key permissions;
- encrypted or malformed private keys;
- non-Ed25519 private or public keys.

Private-key bytes are read through a bounded descriptor-based path and are never included in output or operator summaries.

## 2. Sign a structurally verified manifest

First create and verify the external manifest described in `EVIDENCE_GRAPH_SIGNED_RETIREMENT_RESTORE_EXTERNAL_CUSTODY_EXPORT.md`.

Then sign it:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_export.py sign \
  evidence/external-chain.json \
  --output evidence/external-chain.ed25519.json \
  --key-id custody-ed25519-2026-01 \
  --private-key-path custody-ed25519-private.pem
```

The command:

1. descriptor-reads and structurally verifies the manifest;
2. validates strict custody chronology;
3. loads an Ed25519 private key;
4. derives the public key;
5. computes SHA-256 over the raw 32-byte public key;
6. signs canonical JSON containing the algorithm, key ID, public-key fingerprint, complete manifest, and envelope schema;
7. atomically creates the signed envelope without overwrite.

## 3. Verify with the public key

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_export.py verify-signature \
  evidence/external-chain.ed25519.json \
  --public-key-path custody-ed25519-public.pem \
  --expected-key-id custody-ed25519-2026-01 \
  --expected-public-key-sha256 PUBLIC_KEY_SHA256
```

Verification requires:

- strict envelope schema;
- canonical base64 encoding of the 64-byte Ed25519 signature;
- structurally valid embedded custody manifest;
- valid custody chronology;
- public-key fingerprint equal to the envelope fingerprint;
- optional exact key-ID pin;
- optional exact public-key-fingerprint pin;
- successful Ed25519 verification over canonical JSON.

Wrong keys, changed manifests, changed key IDs, changed fingerprints, or changed signatures fail closed.

## 4. Envelope contents

The envelope contains:

- `algorithm: ed25519`;
- explicit operator key ID;
- SHA-256 fingerprint of the raw public key;
- the complete external custody manifest;
- canonical base64 Ed25519 signature;
- schema and safety flags.

It does not contain:

- private-key material;
- private/public key paths;
- source text;
- raw actor IDs;
- restore or journal mutation actions.

The envelope is publicly verifiable by anyone possessing the public key.

## 5. Commands deliberately absent

The signature extension adds only:

```text
sign
verify-signature
```

It does not add:

- key generation;
- private-key export;
- key deletion;
- key rotation;
- manifest import;
- restore execution;
- journal mutation;
- artifact deletion;
- trusted timestamping.

## 6. Trust boundary

Ed25519 proves that the holder of the corresponding private key signed the exact canonical envelope payload. It does not prove:

- the scientific correctness of reviewed relations;
- the real-world identity of the key holder unless the public key is governed externally;
- when the signature was created;
- that a key was not compromised;
- hardware-backed key custody;
- transparency-log inclusion.

Operators must distribute and pin public keys through an independently governed channel.

## 7. Focused verification

Executed in a reconstructed dependency workspace using the committed signing design and the locally installed `cryptography` Ed25519 implementation:

```text
3 focused checks passed
```

Covered:

- Ed25519 sign/verify round trip;
- explicit key-ID and public-key-fingerprint pinning;
- atomic no-overwrite output;
- wrong-public-key refusal;
- signature tamper refusal;
- broad private-key permission refusal.

Repository-native contracts additionally cover invalid key material and secret-free CLI summaries. A complete unchanged current checkout was not available, so full exact-current pytest, coverage, Ruff, platform, container, and independent-process matrices remain open.

Release readiness is not claimed.
