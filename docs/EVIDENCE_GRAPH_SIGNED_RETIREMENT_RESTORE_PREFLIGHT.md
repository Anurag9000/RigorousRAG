# Signed retirement snapshot restore preflight

Last updated: 2026-08-02

The restore preflight compares a verified signed-retirement snapshot with an already initialized retirement database opened in SQLite read-only mode.

It performs no restore, insert, update, delete, schema creation or pointer mutation.

## 1. Command

```bash
python scripts/evidence_graph_set_signed_retirement_restore.py preflight \
  /secure/audit/retirements.json \
  --target-db-path /restore-target/evidence_graph_set_signed_retirements.sqlite3 \
  --limit 10000
```

The target database must already exist and contain the signed retirement journal schema. A missing, empty uninitialized or redirected path is refused.

## 2. Read-only target contract

The target view:

- requires a regular non-redirecting database file;
- records parent-directory and database-inode identity;
- opens SQLite with `mode=ro`;
- enables `PRAGMA query_only=ON`;
- checks that `evidence_graph_set_signed_retirements` exists;
- exposes only bounded owner-scoped listing;
- revalidates path identities before every connection;
- performs no schema initialization.

This is separate from the normal runtime factory, which is intentionally allowed to create an operational database when one does not exist.

## 3. Comparison model

For every retirement ID in the union of snapshot and target records, the preflight reports one status:

- `exact_match`;
- `missing_target_record`;
- `additional_target_record`;
- `immutable_collision`;
- `state_collision`.

Immutable comparison covers:

- retirement ID;
- owner ID;
- publication operation ID;
- graph-set key;
- signed candidate ID and digest;
- weaker candidate ID;
- signed authority digest;
- schema version.

State comparison covers the remaining durable fields, including phase, state, attempts, leases, final pointer observation, verification/failure metadata and timestamps.

## 4. Dispositions

### `empty_snapshot_no_restore`

The verified snapshot contains no records. No restore is meaningful.

### `empty_target_restore_candidate`

The snapshot contains records and the initialized target journal is empty.

This is the only disposition marked `eligible_for_future_restore=true`. It is planning evidence only; no restore command exists.

### `already_restored_exactly`

Every snapshot record exists in the target with exactly identical immutable and mutable state, and the target contains no additional records.

No action is needed.

### `target_nonterminal_refusal`

The target contains at least one `planned`, `running` or `failed` retirement. Restore or merge design must not overwrite active or recoverable work.

### `immutable_collision_refusal`

The same retirement ID maps to different immutable scope. This is treated as corruption, a cryptographic collision or an invalid target implementation.

### `state_collision_refusal`

The immutable retirement scope matches, but durable state differs. The preflight never chooses one state over another.

### `partial_restore_refusal`

The target contains an exact subset but is missing snapshot history. Partial insertion is not implemented.

### `target_additional_history_refusal`

The target contains records absent from the snapshot. The preflight refuses to treat the snapshot as a replacement or authoritative superset.

## 5. Safety and privacy

The report contains:

- owner and snapshot digest;
- record counts;
- exact/missing/additional/collision counts;
- nonterminal target count;
- per-retirement comparison status and state/phase;
- deterministic report digest;
- false mutation/restore/source-text flags.

It contains no source text, relation evidence, graph node text, query text or reviewer assertion secrets.

The target database remains byte-for-byte unchanged in focused CLI contracts.

## 6. Why restore is still absent

A safe restore executor still requires:

- explicit signed operator authorization;
- target emptiness and exclusive lock guarantees;
- a durable restore intent journal;
- transactionally bounded inserts;
- crash recovery between inserted records;
- exact replay and collision refusal;
- backup-before-restore;
- post-restore snapshot regeneration and digest comparison;
- multi-process exclusion;
- SQLite I/O, disk-full and process-kill fault injection;
- Windows and container filesystem matrices.

A restore-eligible preflight is not restore authorization.

## 7. Committed contracts

Contracts cover:

- empty initialized target eligibility;
- exact terminal idempotence;
- nonterminal target refusal;
- state collision refusal;
- partial target refusal;
- additional target history refusal;
- initialized-schema requirement;
- SQLite query-only write refusal;
- CLI byte-preserving target access;
- false restore/insert/mutation flags.

These newest restore-preflight contracts have not yet been run in a fresh exact-current complete repository checkout. Release readiness is not claimed.
