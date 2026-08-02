# Signed retirement snapshot restore execution

Last updated: 2026-08-02

This runbook covers the crash-recoverable restore of a verified signed-retirement snapshot into an already initialized, globally empty signed-retirement database.

The restore path is intentionally narrow. It does not overwrite, merge, delete, compact, or import nonterminal work.

## 1. Durable stores

The live retirement journal and restore-intent journal must be separate files:

```bash
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DB_PATH=data/evidence_graph_set_signed_retirements.sqlite3
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_RESTORE_DB_PATH=data/evidence_graph_set_signed_retirement_restores.sqlite3
```

The restore runtime rejects canonical-path equality and existing hard-link aliases between the restore-intent journal and:

- the explicit target retirement database;
- the configured retirement database;
- the authorization-only publication journal;
- the signed publication journal.

## 2. Preconditions

A restore can be seeded only when all of the following hold:

1. the snapshot passes descriptor-safe verification;
2. its checksum and record reconstruction are valid;
3. every record belongs to the snapshot owner;
4. every record is terminal: `completed` or `cancelled`;
5. the snapshot is non-empty;
6. the target path already exists;
7. the target contains the initialized signed-retirement schema;
8. the entire target table is empty, not merely empty for one owner;
9. the operator supplies the exact snapshot digest as confirmation;
10. no restore intent with the same immutable identity conflicts with stored state.

Snapshots containing `planned`, `running`, or `failed` work are refused because restoring them could resurrect executable or retryable operations.

## 3. Immutable restore identity

The deterministic restore ID commits:

- owner ID;
- snapshot digest;
- canonical target-path digest.

The durable row also binds the snapshot record count. Snapshot content or target-path drift after claiming causes a durable generic failure.

Raw snapshot paths and raw target paths are not stored in the restore-intent journal or emitted in status output.

## 4. Seed

```bash
python scripts/evidence_graph_set_signed_retirement_restore_execute.py seed \
  --snapshot /secure/backups/retirements.json \
  --target-db-path /restore/retirements.sqlite3 \
  --confirm-snapshot-digest SNAPSHOT_SHA256 \
  --max-attempts 3
```

Seed performs:

- descriptor-safe snapshot verification;
- exact digest confirmation before opening the restore-intent database;
- terminal-only validation;
- initialized-target validation;
- global-empty-target validation;
- one deterministic restore-intent insertion.

Seed does not insert target rows.

A target that already contains the exact snapshot cannot be used to create a new retroactive restore intent. Exact target replay is accepted only when the deterministic intent already exists.

## 5. Execute

```bash
python scripts/evidence_graph_set_signed_retirement_restore_execute.py execute RESTORE_ID \
  --snapshot /secure/backups/retirements.json \
  --target-db-path /restore/retirements.sqlite3 \
  --worker-id restore-worker-1 \
  --lease-seconds 60
```

The durable phases are:

```text
planned -> target_committed -> verified
```

Execution first claims an expiring exclusive restore lease. It then re-verifies the snapshot and exact target-path digest against the immutable intent.

### `planned`

The executor opens one `BEGIN IMMEDIATE` transaction on the target.

- If the target table is empty, every snapshot record is inserted in deterministic retirement-ID order and the transaction commits.
- If the target already equals the snapshot exactly, the executor treats it as crash-after-target-commit recovery and performs no target mutation.
- Partial, additional, immutable-collision, or state-collision history is refused.

The executor then records `target_committed` with a deterministic verification digest.

### `target_committed`

The executor opens another `BEGIN IMMEDIATE` transaction on the target, revalidates exact target equality, and keeps that write lock while marking the restore intent `verified` and `completed`.

This lock prevents a cooperating SQLite writer from adding or changing target history between final exact verification and durable restore-intent completion.

The target transaction performs no row mutation during this final phase.

## 6. Reconcile

```bash
python scripts/evidence_graph_set_signed_retirement_restore_execute.py reconcile-one \
  --owner-id alice \
  --snapshot /secure/backups/retirements.json \
  --target-db-path /restore/retirements.sqlite3 \
  --worker-id restore-worker-1 \
  --lease-seconds 60
```

Reconcile selects one planned or expired-running restore for the owner and executes it using one captured timestamp.

The supplied snapshot and target must match the selected operation. A worker should be configured for one snapshot/target scope at a time.

## 7. Crash recovery

### Crash before target commit

The target transaction rolls back. The intent remains recoverable from `planned` after lease expiry or explicit retry.

### Crash after target commit but before `target_committed`

The target contains the complete exact snapshot. Retry detects exact equality, does not insert duplicate rows, and advances the durable phase.

### Crash after `target_committed` but before completion

Retry preserves the phase, rechecks exact target history under a target write lock, and completes.

### Crash after restore-intent completion but before target lock release

The target rows were already committed in the earlier phase. The no-op final target transaction rolls back on process death; the completed intent and exact restored history remain valid.

## 8. Retry and cancellation

```bash
python scripts/evidence_graph_set_signed_retirement_restore_execute.py retry RESTORE_ID \
  --owner-id alice \
  --confirm-restore-id RESTORE_ID
```

Retry is allowed only for `failed` attempts below their attempt ceiling. It preserves `target_committed` so recovery never repeats target insertion unnecessarily.

```bash
python scripts/evidence_graph_set_signed_retirement_restore_execute.py cancel RESTORE_ID \
  --owner-id alice \
  --confirm-restore-id RESTORE_ID
```

Cancellation is allowed only while the restore is unstarted: state `planned` or `failed`, phase `planned`.

Once target history may have been committed, cancellation is refused. The operator must recover or investigate the exact target state.

Exact retry/cancel confirmation is checked before opening the restore-intent database and again inside the journal.

## 9. Read-only inspection

```bash
python scripts/evidence_graph_set_signed_retirement_restore_execute.py status RESTORE_ID
python scripts/evidence_graph_set_signed_retirement_restore_execute.py list \
  --owner-id alice
```

Status and list open only the restore-intent journal. They do not require the snapshot, target, graph stores, or publication stores.

Outputs contain only IDs, digests, counts, states, phases, lease fields, timestamps, and explicit safety flags.

## 10. Permanent exclusions

There is no command or internal branch for:

- overwriting an existing target record;
- merging a partial target;
- deleting additional target records;
- restoring nonterminal work;
- restoring an empty snapshot;
- changing the owner or target bound to an existing intent;
- relabeling pre-existing history as restored;
- using restore as publication, approval, or graph mutation.

## 11. Verification boundary

Executed in the reconstructed focused workspace:

```text
11 passed
```

Focused compilation also passed.

The executed checks cover restore identity, journal lifecycle, retry/cancel, lease reclaim, tamper and inode replacement, normal atomic restore, exact replay, both crash windows, terminal-only enforcement, no-merge behavior, post-claim scope drift, final target-lock refusal, runtime aliasing, lazy status/list, confirmation-before-open, and generic error output.

The repository also contains 17 restore-family repository-native contracts across journal, execution, recovery, runtime and CLI files. They have not yet been executed together from a fresh exact-current checkout.

Full repository pytest, coverage, Ruff, Windows, Docker/Compose, independent-process contention, process-kill, SQLite busy/WAL/I/O, and disk-full matrices remain open. Release readiness is not claimed.
