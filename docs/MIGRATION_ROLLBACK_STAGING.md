# Isolated rollback staging verification

Last updated: 2026-08-02

## Scope

The staging verifier proves that an encrypted rollback artifact can be:

1. authenticated and decrypted in memory;
2. validated against its cutover preflight;
3. reconstructed into the repository's public immutable vector, sparse and generation snapshot types;
4. written to a separate bounded process-local staging store;
5. read back as fresh deep-copied snapshots;
6. re-hashed to the exact vector, sparse and rollback identities captured by the preflight.

The staging store is intentionally non-authoritative. It does not receive a RAG layer, sparse database, generation database, authoritative coordinator or current-pointer object.

## Components

`tools/migration_rollback_staging.py` provides:

- `InMemoryRollbackStagingStore`;
- deterministic staging IDs bound to task, preflight and rollback identity;
- deep-copy reconstruction of nested vector/sparse metadata;
- idempotent identical staging and changed-snapshot collision refusal;
- bounded entry count;
- exact vector/sparse re-snapshot identity verification;
- source generation sequence/profile/content/count verification;
- timestamp-stable staging verification digests.

`tools/migration_rollback_staging_cli.py` and `scripts/migration_rollback_staging.py` provide one verify-only command:

```bash
python -m tools.migration_rollback_staging_cli <task-id>
python -m tools.migration_rollback_staging_cli <task-id> \
  --preflight-digest <preflight-sha256>
```

The command requires the configured AES-GCM rollback key, decrypts only in memory and reports:

```json
{
  "staging_verified": true,
  "staging_scope": "process_local_non_authoritative",
  "staging_mutation_performed": true,
  "authoritative_mutation_performed": false,
  "restore_performed": false,
  "cutover_performed": false
}
```

It never returns vector text, sparse text, retained paths or encryption key material.

## What this proves

The staging verifier proves:

- the ciphertext is decryptable under the expected key ID and key;
- the decrypted payload still matches the preflight;
- public immutable snapshot constructors accept the payload;
- an independent store can accept and return deep-copied snapshots;
- the staged snapshots retain exact vector/sparse/source-generation identities.

## What this does not prove

The in-memory staging store is not the production Chroma/vector backend, persistent sparse SQLite backend or durable generation store. Therefore this slice does not prove:

- production backend write compatibility;
- production embedding-function compatibility during vector restore;
- persistent sparse transaction behavior;
- generation-database publication behavior;
- atomic cross-store publication;
- crash recovery during live writes;
- rollback into authoritative state.

Those require explicit temporary backend adapters and fault injection before a live restore path can exist.

## Focused verification

The constrained local staging harness passed **7 tests** covering:

- exact preflight vector/sparse/source identity verification;
- deep-copy behavior for nested metadata;
- idempotent repeated staging;
- changed-snapshot collision refusal;
- changed preflight/generation refusal;
- bounded staging capacity and removal;
- path-free CLI output;
- bounded missing-key and missing-preflight behavior;
- explicit separation of staging and authoritative mutations.

These are focused isolated tests, not the full exact-head repository matrix.

## Next dependency

Before any live restore or cutover command can exist, the repository still requires:

1. temporary adapters backed by the real vector, sparse and generation implementations;
2. re-snapshot verification from those temporary persistent backends;
3. a durable cutover journal with exclusive leases and idempotency keys;
4. an atomic or compensating publication protocol;
5. automatic encrypted rollback after every failed publication/validation phase;
6. exact-head Linux, Windows, container and fault-injection verification.
