# Encrypted migration rollback artifacts

Last updated: 2026-08-02

## Purpose and boundary

The rollback-artifact slice converts one non-mutating cutover preflight’s cryptographic rollback identity into a protected, durable, restorable **payload artifact**. It captures the complete privacy-finalized authoritative vector rows, sparse-field snapshot and durable generation metadata, validates them against the preflight, encrypts them with AES-256-GCM, and publishes ciphertext plus a path-free manifest.

This slice still does **not** restore the payload into authoritative stores and does **not** cut over a live generation. The operator surface supports capture, metadata status, authenticated verification and tightly constrained deletion only.

## Required key configuration

There is no generated key, default key or plaintext fallback. Capture and verification require both variables:

```dotenv
MIGRATION_ROLLBACK_KEY_ID=<operator-managed-key-version>
MIGRATION_ROLLBACK_KEY_B64=<canonical-base64-of-exactly-32-random-bytes>
```

The key ID is stored in the manifest; the key material is not. `MIGRATION_ROLLBACK_KEY_B64` must decode to exactly 32 bytes. Whitespace, malformed base64, missing variables and incorrect lengths fail closed.

Generate key material outside the repository and inject it through the deployment secret mechanism. Do not commit, log, print, include in support bundles or place it in `.env.example` with a real value.

Current limitations:

- environment-variable secret injection is supported;
- KMS/HSM/secret-manager envelope encryption is not yet implemented;
- online key rotation/re-encryption is not yet implemented;
- an artifact must be verified with the exact key ID and key material used to create it.

## Components

### Payload and key contracts

`tools/migration_rollback_artifact.py` provides:

- complete rollback payload reconstruction from an authoritative snapshot;
- exact revalidation against the cutover preflight’s vector, sparse and rollback digests;
- source generation sequence/profile/content/count validation;
- strict duplicate-key and NaN/Infinity JSON handling;
- `RollbackEncryptionKey`, with key bytes excluded from `repr`;
- strict environment key loading;
- a path-free `EncryptedRollbackManifest`;
- deterministic manifest artifact digest excluding creation time only.

The encrypted payload contains:

- task, preflight, owner and document identity;
- durable source-generation metadata;
- complete privacy-finalized vector row IDs, text and metadata;
- complete sparse snapshot fields, provenance and metadata.

It does not contain retained-source file paths, original unredacted bytes, API keys or model-provider secrets.

### AES-256-GCM store

`tools/migration_rollback_store.py` provides:

- AES-256-GCM authenticated encryption;
- a fresh 12-byte random nonce per new artifact;
- authenticated additional data binding the key ID, task, owner, document, preflight, source sequence/profile/content and vector/sparse rollback digests;
- manifest-last, same-filesystem staged publication;
- fsync where supported and best-effort `0700`/`0600` permissions;
- ciphertext, plaintext and authenticated-data SHA-256 checks;
- key-ID mismatch refusal before decryption;
- wrong-key and ciphertext/manifest tamper refusal;
- strict file type, size, symlink/reparse and root-identity checks;
- idempotent reuse of an already verified artifact for the same task/preflight;
- no plaintext fallback.

Each artifact is stored under:

```text
<MIGRATION_ROLLBACK_ROOT>/<task-id>/<preflight-digest>/
  ciphertext.bin
  manifest.json
```

The default root and payload ceiling are:

```dotenv
MIGRATION_ROLLBACK_ROOT=data/migration_rollbacks
MIGRATION_ROLLBACK_MAX_BYTES=536870912
```

### Runtime and operator surface

`tools/migration_rollback_runtime.py` provides a path-scoped process-local store factory.

`tools/migration_rollback_cli.py` and `scripts/migration_rollbacks.py` expose:

```bash
python -m tools.migration_rollback_cli capture <task-id>
python -m tools.migration_rollback_cli capture <task-id> \
  --preflight-digest <preflight-sha256>

python -m tools.migration_rollback_cli status <task-id>
python -m tools.migration_rollback_cli verify <task-id>

python -m tools.migration_rollback_cli remove <task-id> \
  --preflight-digest <preflight-sha256> \
  --confirm-task-id <same-task-id> \
  --confirm-preflight-digest <same-preflight-sha256>
```

`status` reads only the manifest and does not need the decryption key. `capture` and `verify` require the configured key, authenticate/decrypt only in process memory, revalidate the entire payload against the preflight, and return no vector text, sparse text, retained path or key material.

Successful capture and verification report:

```json
{
  "restore_performed": false,
  "mutation_performed": false
}
```

Artifact removal is the only exposed storage mutation. It requires the migration task to be failed or cancelled and requires exact task-ID and preflight-digest confirmation. Removal never performs restore.

## Manifest contents

The plaintext manifest contains only bounded metadata:

- task, owner and document IDs;
- preflight and rollback identity digests;
- source sequence/profile/content hash;
- vector and sparse snapshot digests;
- plaintext and ciphertext hashes and byte counts;
- AES-256-GCM algorithm identifier;
- operator key ID;
- nonce in base64;
- authenticated-data hash;
- creation timestamp and schema version.

The manifest contains neither rollback text nor key material.

## Integrity and confidentiality checks

Verification fails closed on:

- missing, malformed or non-canonical key configuration;
- key-ID mismatch;
- wrong key material or AES-GCM authentication failure;
- changed ciphertext size/hash;
- changed authenticated metadata hash;
- changed plaintext size/hash;
- invalid strict JSON;
- payload identity mismatch;
- generation, vector or sparse digest/count drift;
- symlink/reparse members;
- non-regular files;
- replaced storage roots;
- oversized files.

## Focused verification

The constrained local rollback-artifact harness passed **14 tests** covering:

- complete payload reconstruction and exact preflight revalidation;
- key length, canonical base64 and missing-key failure;
- key material exclusion from object representation;
- AES-GCM encryption/decryption round trip;
- absence of rollback text and retained paths from ciphertext inspection and manifests;
- idempotent repeated capture despite random nonces;
- wrong key ID and wrong key refusal;
- ciphertext and manifest tamper detection;
- root replacement and symlink refusal;
- path-scoped runtime caching;
- capture/status/verify privacy behavior;
- status without a decryption key;
- missing task/preflight bounded not-found behavior;
- failed/cancelled state and double-confirmation deletion requirements.

The combined focused migration promotion, benchmark, statistical, cutover-preflight and rollback-artifact suite passed **70 tests** before the bounded missing-preflight refinement and **71 tests** after it. Python compilation passed for the rollback modules. These are isolated harness results, not the complete exact-head repository matrix.

## Remaining before restore or cutover

1. Reconstruct validated public snapshot types from the decrypted payload without weakening their validators.
2. Restore and verify the payload into isolated temporary vector/sparse/generation stores before any live restore path exists.
3. Add a durable cutover journal with exclusive leases and idempotency keys.
4. Implement an atomic or compensating publication protocol that never exposes an unvalidated mixed generation to retrieval.
5. Validate the new authoritative generation before task commit.
6. Add automatic rollback using the encrypted artifact after every failed publication or validation phase.
7. Verify restored vector/sparse/generation identities against the preflight before releasing old state.
8. Integrate a production secret manager or KMS/HSM and implement governed key rotation/re-encryption.
9. Add retention, legal-hold, compaction and secure-deletion policy for encrypted artifacts and keys.
10. Inject crashes, disk errors, authentication failures and backend failures at every capture, publication, validation and rollback boundary.
11. Pass one unchanged exact-head Linux, Windows and container verification matrix.

## Permanent non-claims

- Encryption at rest is not a complete secret-management system.
- An environment-injected key is not equivalent to KMS/HSM governance.
- A successfully verified rollback artifact has not been restored.
- The existence of a rollback artifact does not authorize cutover.
- AES-GCM confidentiality does not remove the need for filesystem access control, secret rotation, retention policy and operational auditing.
- Release readiness and live cutover safety are not claimed.
