# Non-mutating migration cutover preflight

Last updated: 2026-08-02

## Purpose

The cutover preflight is the final **non-mutating** control-plane stage before any future live migration publication can be designed. It binds:

- one validated migration task;
- one exact shadow artifact manifest;
- one current eligible paired promotion report;
- one unchanged authoritative source generation;
- one complete current vector rollback snapshot identity;
- one complete current sparse rollback snapshot identity.

The preflight stores only hashes, counts, sequences and fingerprints. It does not store rollback document text, sparse field text, retained-source paths or raw benchmark content.

No command in this slice can:

- approve a cutover;
- write shadow rows into the authoritative stores;
- replace vector or sparse state;
- change the durable current-generation pointer;
- mark a migration task committed;
- restore a rollback snapshot.

## Components

### Identity builder

`tools/migration_cutover_preflight.py` builds a validated `CutoverPreflight`.

A preflight requires:

- migration task state `validated`;
- current promotion report decision `eligible`;
- promotion policy ID `paired-promotion-v1`, proving that the report includes aggregate and paired statistical gates;
- exact task, owner, document, source-sequence, source-profile, target-profile and shadow-validation alignment across the journal, shadow and promotion report;
- live source generation state `active` or `restored`;
- unchanged source generation sequence, profile fingerprint and content hash;
- shadow content hash equal to the authoritative source content hash;
- complete vector rollback snapshot under the exact owner/document scope;
- vector rollback row count equal to the durable generation record;
- complete sparse rollback snapshot under the exact owner/document scope;
- sparse rollback generation and profile equal to the durable generation record/task.

The builder hashes the complete current vector rows and sparse field snapshot in memory, then retains only the digests.

### Persisted preflight fields

A persisted preflight contains:

- task, owner and document IDs;
- source sequence;
- source and target profile fingerprints;
- source content hash;
- shadow validation digest;
- promotion report digest;
- benchmark fingerprint;
- vector and sparse rollback snapshot digests;
- composite rollback identity digest;
- composite target artifact digest;
- source vector row count;
- source sparse generation and field count;
- target vector and sparse row counts;
- creation timestamp and schema version.

The `preflight_digest` excludes only the creation timestamp. Replanning an unchanged source/shadow/report combination reuses the original immutable record rather than creating timestamp-only audit churn.

### Append-only store

`tools/migration_cutover_preflight_store.py` provides:

- immutable preflights addressed by preflight digest;
- one atomic per-task `current.json` pointer;
- bounded history;
- strict duplicate-key and NaN/Infinity refusal;
- regular-file, symlink/reparse and root-identity defenses;
- idempotent reuse of an identical preflight;
- exact digest verification on read;
- removal only through the operator boundary after task-state checks.

The default storage root is:

```dotenv
MIGRATION_CUTOVER_PREFLIGHT_ROOT=data/migration_cutover_preflights
```

### Runtime and operator surface

`tools/migration_cutover_preflight_runtime.py` provides the path-scoped process-local store factory.

`tools/migration_cutover_preflight_cli.py` and `scripts/migration_cutover_preflights.py` provide only:

```bash
python -m tools.migration_cutover_preflight_cli plan <task-id>
python -m tools.migration_cutover_preflight_cli status <task-id>
python -m tools.migration_cutover_preflight_cli status <task-id> \
  --preflight-digest <preflight-sha256>
python -m tools.migration_cutover_preflight_cli history <task-id> --limit 100
python -m tools.migration_cutover_preflight_cli remove-task <task-id> \
  --confirm-task-id <same-task-id>
```

Every successful `plan`, `status` and `history` payload includes:

```json
{"mutation_performed": false}
```

Preflight removal requires the migration task to be `failed` or `cancelled` and requires exact task-ID confirmation.

## Rollback identity versus rollback artifact

The current slice creates a **rollback identity**, not a durable rollback artifact.

The rollback identity proves which complete vector and sparse snapshots were observed during preflight. It does not retain enough material to restore those snapshots after a later cutover.

Before live cutover can be implemented, the repository must add a private durable rollback-artifact store that:

- preserves the complete privacy-finalized vector and sparse snapshots;
- encrypts or otherwise protects sensitive indexed text at rest under an explicit secret/key-management policy;
- verifies stored artifact digests against the preflight rollback identity;
- keeps old authoritative state until rollback verification passes;
- applies bounded retention and secure deletion;
- never returns rollback text or paths through public/operator JSON.

## Focused verification

The constrained local cutover-preflight harness passed **15 tests** covering:

- exact source/shadow/promotion identity binding;
- requirement for an eligible paired statistical report;
- unchanged source generation sequence/profile/content;
- complete vector and sparse snapshot hashing;
- owner/document metadata isolation;
- vector row-count and sparse-generation agreement;
- timestamp-stable preflight digests;
- append-only history and atomic current pointer;
- idempotent unchanged replanning;
- report-change history;
- tamper detection;
- path-free persisted records and CLI output;
- symlink/reparse and replaced-root refusal;
- status/history/not-found semantics;
- exact-confirmation cleanup restricted to failed/cancelled tasks;
- explicit non-mutation output.

This is focused isolated verification, not a clean exact-head repository run.

## Remaining before live cutover

1. Execute paired benchmark fixtures against real current and shadow retrieval stacks.
2. Measure latency, memory, storage and provider billing under governed instrumentation.
3. Add a protected durable rollback-artifact store.
4. Add a cutover journal with exclusive leases and idempotency keys.
5. Add an atomic or compensating vector+sparse+generation publication protocol with no unvalidated mixed state exposed to retrieval.
6. Validate the newly published generation before marking the migration committed.
7. Add automatic rollback on every failed publication/validation phase.
8. Verify rollback against the stored rollback identity and source generation.
9. Add exact operator authorization and immutable audit records.
10. Add bounded retention/compaction for shadows, reports, preflights and rollback artifacts.
11. Inject crashes and backend failures before and after every write and pointer transition.
12. Pass one unchanged exact-head Linux, Windows and container verification matrix.

## Permanent non-claims

- A preflight is not approval.
- A rollback digest is not a restorable rollback artifact.
- An eligible promotion report is not universal scientific superiority.
- A preflight does not mutate or reserve the live generation.
- Source state may change after preflight; every future cutover must revalidate it under the same exclusive lock used for publication.
- Release readiness is not claimed.
