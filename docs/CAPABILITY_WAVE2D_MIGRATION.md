# Capability Wave 2D — profile migration control plane

Last updated: 2026-08-01

## Scope of this slice

This slice implements migration inventory, immutable task planning and a durable resumable journal. It intentionally does **not** execute reindexing or cut over a live document. Migration execution remains disabled until independent shadow stores and validation gates exist.

## Implemented components

- `tools/migration_types.py`
  - validated migration candidates and durable tasks;
  - bounded identifiers, exact integers, SHA-256 fingerprints and finite timestamps;
  - planned, running, validated, committed, failed and cancelled states.
- `tools/migration_planner.py`
  - compares durable current generations with a target embedding profile;
  - consults the retained-document registry without returning source paths;
  - classifies ready, already-current, source-unavailable, deleted and registry-inspection-failed documents;
  - derives deterministic task IDs from owner, document, source sequence and source/target fingerprints.
- `tools/migration_journal.py`
  - SQLite task journal with immutable task identity;
  - idempotent seeding;
  - bounded retry attempts;
  - expiring worker leases and renewal;
  - validation digests required before commit;
  - generic failure types rather than private exception messages;
  - cancellation restricted to planned or failed tasks;
  - symlink/reparse and database identity checks.
- `tools/migration_runtime.py`
  - path-keyed process-local journal factory.
- `tools/index_migration_cli.py` and `scripts/index_migrations.py`
  - inventory;
  - seed;
  - status filtering;
  - owner-verified cancellation;
  - bounded JSON without retained-source paths;
  - no execution or cutover command.

## State machine

1. `planned` — immutable task seeded from a current source generation.
2. `running` — one worker owns an unexpired lease and an incremented attempt.
3. `validated` — shadow output has a validation digest and remains leased.
4. `committed` — cutover completed after validation; lease cleared.
5. `failed` — generic failure type recorded; task may be retried within its attempt budget.
6. `cancelled` — operator-cancelled before active execution.

Expired running tasks may be reclaimed. Expired validated tasks preserve the validation digest so a new worker may finish cutover without repeating an already recorded validation step.

## Inventory privacy

The planner records only:

- owner and document IDs;
- current generation sequence;
- source and target profile fingerprints;
- canonical target profile alias;
- a retained-source availability boolean;
- eligibility and a bounded reason code.

Retained-source filesystem paths are never included in candidates, task rows or CLI JSON.

## Focused contracts

Tests cover:

- profile-drift classification;
- canonical profile aliases;
- stable task IDs;
- idempotent seed behavior;
- lease acquisition and renewal;
- validation-before-commit;
- retry ceilings;
- expired running and validated task recovery;
- cancellation restrictions;
- cross-owner cancellation refusal before mutation;
- database identity replacement;
- generic path-free CLI errors.

## Required next slice

- shadow vector and sparse stores isolated by migration task;
- retained-source reparse through the current privacy-finalized ingestion pipeline;
- target-profile encoder construction through an explicit adapter interface;
- validation of content hash, field/chunk counts, provenance, retrieval metrics and resource budgets;
- durable shadow artifact identity;
- atomic current-generation cutover;
- rollback references and bounded shadow retention;
- crash/fault injection at every transition;
- pause/resume/cancel semantics for active workers.

## Verification boundary

Source and focused contracts are committed. The complete exact-head Linux, Windows, container and release-lock matrix is not currently observable as green for the latest `main` SHA, so migration execution and release readiness are not claimed.
