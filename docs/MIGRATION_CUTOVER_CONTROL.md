# Preparation-only migration cutover control

Last updated: 2026-08-02

## Purpose

The cutover-control slice creates one durable, leased, idempotent audit operation after every non-mutating prerequisite has passed:

- validated migration task;
- eligible paired-statistical promotion report;
- exact cutover preflight;
- authenticated encrypted rollback artifact;
- successful typed rollback reconstruction;
- successful isolated non-authoritative staging verification;
- unchanged current authoritative source generation.

It does not execute a cutover. The journal state machine intentionally contains only:

1. `planned`;
2. `running`;
3. `ready`;
4. `failed`;
5. `cancelled`.

There is no `executing`, `committed`, `published`, `rolled_back` or pointer-swap state.

## Immutable preparation identity

`tools/migration_cutover_control.py` defines `CutoverPreparation`. Its deterministic operation ID binds:

- task, owner and document identity;
- source sequence, profile and content hash;
- target profile;
- shadow validation digest;
- paired promotion report and benchmark fingerprints;
- cutover-preflight digest;
- rollback identity and encrypted rollback artifact digests;
- rollback key ID, but never key material;
- isolated-staging verification digest;
- target artifact identity;
- current vector and sparse snapshot digests;
- current and target row/generation counts.

The operation ID excludes only the preparation timestamp. Re-running unchanged preparation reuses the original operation instead of creating timestamp-only duplicates.

## Exact prerequisite checks

`build_cutover_preparation` requires exact agreement across the task, preflight, promotion report, encrypted rollback manifest and staging verification. It also re-reads the current generation and requires it to remain:

- `active` or `restored`;
- on the exact source sequence;
- on the exact source profile;
- on the exact source content hash;
- at the exact vector row count;
- at the exact sparse generation.

Any mismatch blocks preparation before a ready operation can exist.

## Durable leased journal

`tools/migration_cutover_journal.py` provides a path-safe SQLite journal with:

- immutable deterministic operation IDs;
- idempotent seeding;
- owner-scoped bounded listing;
- exclusive expiring worker leases;
- retry ceilings;
- expired-running reclamation;
- generic bounded failure types;
- ready state only after a second prerequisite resolution under the lease;
- cancellation restricted to planned or failed operations;
- symlink/reparse and database/parent identity defenses;
- strict duplicate-key and nonstandard-number preparation decoding.

A ready operation is terminal within this preparation journal. It cannot be claimed again or cancelled through the preparation surface.

## Double resolution under lease

`tools/migration_cutover_runtime.py` resolves all prerequisites twice:

1. resolve and seed the deterministic operation;
2. acquire the exclusive preparation lease;
3. resolve every prerequisite again;
4. require the second deterministic operation ID to equal the first;
5. mark ready only if nothing changed.

If any prerequisite changes while the lease is held, the operation records a generic failure type and remains non-executable.

The second resolution includes decrypting and authenticating the rollback artifact, typed reconstruction, isolated staging and current-generation verification.

## Operator surface

`tools/migration_cutover_control_cli.py` and `scripts/migration_cutover_control.py` expose only:

```bash
python -m tools.migration_cutover_control_cli prepare <task-id> \
  --worker-id cutover-preparer-1

python -m tools.migration_cutover_control_cli status <operation-id>

python -m tools.migration_cutover_control_cli list \
  --owner-id alice \
  --state ready \
  --limit 100

python -m tools.migration_cutover_control_cli cancel <operation-id> \
  --confirm-operation-id <same-operation-id>
```

Every successful operation payload reports:

```json
{
  "authoritative_mutation_performed": false,
  "restore_performed": false,
  "cutover_performed": false
}
```

The output contains digests, counts, bounded IDs, state and lease metadata. It contains no rollback text, sparse text, retained paths or encryption key material.

## Configuration

```dotenv
MIGRATION_CUTOVER_DB_PATH=data/migration_cutovers.sqlite3
```

The prepare command also requires the rollback encryption variables documented in `config/migration_rollback.env.example`.

## Focused verification

The constrained local cutover-control harness passed **13 tests** covering:

- deterministic preparation IDs across timestamp-only reruns;
- exact task/preflight/report/rollback/staging identity binding;
- blocked-report, stale-generation and mismatched-staging refusal;
- idempotent journal seed;
- exclusive leases and ready transition;
- terminal ready state;
- expired-running reclamation and retry ceilings;
- generic failure recording and planned/failed cancellation;
- owner-scoped listing;
- database identity replacement refusal;
- double resolution under lease;
- changed-prerequisite failure during preparation;
- invalid resolver refusal before journal mutation;
- privacy-safe prepare/status/list/cancel CLI behavior;
- bounded not-found and exact-confirmation behavior.

These are focused isolated tests, not the complete exact-head repository matrix.

## Safety boundary

A ready preparation means only that the immutable prerequisites agreed twice during one leased preparation interval. It does not reserve or freeze authoritative state after the lease ends. A future publication executor must revalidate the current generation while holding the same exclusive owner/document lock used for every authoritative write.

Before an execution state can be added, the repository still requires:

1. real-backend temporary vector/sparse/generation restore verification;
2. a publication/rollback journal designed around actual write phases;
3. atomic or compensating cross-store publication with retrieval visibility control;
4. automatic authenticated rollback on every failed phase;
5. exact post-publication and post-rollback generation validation;
6. production KMS/HSM or secret-manager integration and key rotation;
7. retention and secure deletion;
8. crash and backend-failure injection at every transition;
9. one unchanged exact-head Linux, Windows and container verification matrix.

## Permanent non-claims

- `ready` is not authorization.
- `ready` is not a lock on future source state.
- `ready` is not publication or restore.
- A deterministic operation ID is an idempotency identity, not distributed consensus.
- SQLite leases are single-host coordination, not a distributed exactly-once guarantee.
- Live cutover and release readiness are not claimed.
