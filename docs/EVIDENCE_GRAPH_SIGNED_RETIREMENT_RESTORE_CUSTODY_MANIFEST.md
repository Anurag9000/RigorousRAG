# Custody-governed signed retirement restore workflow

Last updated: 2026-08-02

This runbook defines the durable chain of custody for terminal signed-retirement snapshot restores.

The custody workflow binds one verified pre-restore receipt and SQLite backup artifact to one deterministic restore intent before target work begins. After the restore completes, it binds one exact-comparison receipt to the same custody manifest.

## 1. Isolated custody database

Configure:

```bash
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_DB_PATH=data/evidence_graph_set_signed_retirement_custody.sqlite3
```

The custody database must not equal or hard-link to:

- the target signed-retirement database;
- the restore-intent journal;
- the legal-hold database;
- the authorization-only publication journal;
- the signed publication journal.

The documented environment fragment is available at:

```text
config/evidence_graph_restore_custody.env.example
```

## 2. Custody identity and state

The deterministic custody ID commits:

- owner ID;
- restore ID;
- pre-receipt digest;
- backup artifact SHA-256.

The monotonic state is:

```text
pre_bound -> post_bound
```

There is no deletion, reset, replacement, or reverse transition.

The complete manifest row is digest-reconstructed on every read. It contains only IDs, digests, byte counts, actor-binding provenance, states, and timestamps. Raw snapshot, receipt, backup, and target paths are not stored.

## 3. Required operator sequence

### Step 1 — create pre-restore backup evidence

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody.py pre-create \
  --snapshot /secure/backups/retirements.json \
  --target-db-path /restore/retirements.sqlite3 \
  --backup-output /secure/backups/empty-target.sqlite3 \
  --receipt-output /secure/backups/empty-target.receipt.json \
  --confirm-snapshot-digest SNAPSHOT_SHA256 \
  --actor-id operator-1
```

Verify it independently:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody.py pre-verify \
  --receipt /secure/backups/empty-target.receipt.json \
  --backup /secure/backups/empty-target.sqlite3
```

### Step 2 — seed and pre-bind through the governed entrypoint

Use the custody-preflighted operator entrypoint:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_governed.py seed \
  --snapshot /secure/backups/retirements.json \
  --target-db-path /restore/retirements.sqlite3 \
  --confirm-snapshot-digest SNAPSHOT_SHA256 \
  --pre-receipt /secure/backups/empty-target.receipt.json \
  --backup /secure/backups/empty-target.sqlite3 \
  --actor-id operator-1 \
  --max-attempts 3
```

Before restore-intent creation, the governed script:

1. verifies the pre receipt and backup;
2. computes the canonical target-path digest;
3. requires exact receipt-target alignment;
4. initializes the custody runtime with the explicit target path, refusing database aliasing.

The module CLI then:

1. verifies the exact snapshot digest;
2. loads the process-owned actor;
3. seeds the deterministic restore intent;
4. binds the verified receipt/backup to that restore in `pre_bound` state.

If manifest binding fails, the restore intent may remain planned but target history is unchanged. Execute and reconcile refuse it until exact custody is bound.

A separately seeded planned restore may be bound explicitly:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_manifest.py bind-pre RESTORE_ID \
  --confirm-restore-id RESTORE_ID \
  --pre-receipt /secure/backups/empty-target.receipt.json \
  --backup /secure/backups/empty-target.sqlite3 \
  --actor-id operator-1
```

Pre-binding is allowed only while the restore is `planned` or `failed` in phase `planned`. Once target work may have begun, retroactive pre-binding is refused.

## 4. Execute or reconcile with live custody evidence

Execute one restore:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_governed.py execute RESTORE_ID \
  --snapshot /secure/backups/retirements.json \
  --target-db-path /restore/retirements.sqlite3 \
  --pre-receipt /secure/backups/empty-target.receipt.json \
  --backup /secure/backups/empty-target.sqlite3 \
  --worker-id restore-worker-1 \
  --lease-seconds 60
```

Reconcile one claimable restore:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_governed.py reconcile-one \
  --owner-id alice \
  --snapshot /secure/backups/retirements.json \
  --target-db-path /restore/retirements.sqlite3 \
  --pre-receipt /secure/backups/empty-target.receipt.json \
  --backup /secure/backups/empty-target.sqlite3 \
  --worker-id restore-worker-1 \
  --lease-seconds 60
```

Before the restore lease is claimed or target state is mutated, canonical execution verifies:

- a custody manifest exists for the selected restore;
- owner, snapshot digest, and target-path digest match the restore intent;
- the supplied pre receipt has the exact bound receipt digest;
- the supplied backup has the exact bound SHA-256 and byte size;
- the custody database is not the restore target.

A changed, missing, or substituted receipt/backup fails before restore execution.

Completed restore replay remains custody-validated and target-read-only.

## 5. Create and bind post-restore comparison evidence

Create the exact comparison receipt:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody.py post-create RESTORE_ID \
  --confirm-restore-id RESTORE_ID \
  --snapshot /secure/backups/retirements.json \
  --target-db-path /restore/retirements.sqlite3 \
  --pre-receipt /secure/backups/empty-target.receipt.json \
  --backup /secure/backups/empty-target.sqlite3 \
  --receipt-output /secure/backups/restore-comparison.receipt.json \
  --actor-id auditor-1
```

Bind it to the durable custody manifest:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_manifest.py bind-post RESTORE_ID \
  --confirm-restore-id RESTORE_ID \
  --post-receipt /secure/backups/restore-comparison.receipt.json \
  --actor-id auditor-1
```

Post-binding requires:

- restore state `completed` and phase `verified`;
- exact restore, owner, snapshot, target, pre-receipt, backup, and target-verification digests;
- process-owned binding actor;
- monotonic transition from `pre_bound` to `post_bound`.

Exact replay preserves the original post-binding actor timestamp. Changed receipt or actor provenance is refused as a collision.

## 6. Inspect custody state

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_manifest.py status CUSTODY_ID
```

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_manifest.py status-for-restore RESTORE_ID
```

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_manifest.py list \
  --owner-id alice \
  --state pre_bound \
  --limit 100
```

Read commands load only the custody database and return no raw paths.

## 7. Replay and collision rules

### Exact pre-binding replay

A later invocation with the same restore, pre receipt, backup, and actor returns the stored manifest. The original `pre_bound_at` remains authoritative.

### Exact post-binding replay

A later invocation with the same post receipt and binding actor returns the stored manifest. The original `post_bound_at` remains authoritative.

### Refused collisions

The store refuses changes to:

- receipt or backup digest;
- backup size;
- owner, snapshot, target, or restore scope;
- pre/post binding actor identity, method, or binding digest;
- post target-verification digest;
- reconstructed manifest digest.

## 8. Commands intentionally absent

There is no custody-manifest command for:

- delete;
- overwrite;
- reset to pre-bound;
- replace receipt or backup;
- change actor provenance;
- execute or cancel a restore;
- alter target retirement history.

## 9. Verification boundary

The exact live custody-manifest modules and repository-native manifest test file were executed in the reconstructed dependency workspace:

```text
5 passed
```

The exact current governed-entrypoint and public-entrypoint custody tests were also executed together with the manifest tests and passed.

The earlier custody receipt SQLite slice passed:

```text
6 passed
```

Compilation passed for the exact custody contracts, receipt boundary, manifest store, replay boundary, runtime, and tests.

These focused executions are not the full exact-current repository suite. Complete pytest, coverage, Ruff, Windows, containers, independent-process races, disk-full/I/O failure injection, asymmetric signatures, and trusted timestamps remain open. Release readiness is not claimed.
