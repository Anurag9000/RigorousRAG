# Crash-recoverable restore-intent deletion execution

Last updated: 2026-08-04

This control plane logically removes one authorized terminal restore-intent row from the restore journal while preserving every legal-hold, custody, receipt, artifact, signature, signer-key, timestamp, authorization, deletion-attempt, marker, and tombstone record.

It does **not** securely erase SQLite pages, vacuum databases, delete custody evidence, remove source files, mutate restored target databases, or relabel historical provenance.

## Configuration

```bash
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DELETION_AUTH_DB_PATH=data/evidence_graph_set_signed_retirement_deletion_authorizations.sqlite3
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DELETION_DB_PATH=data/evidence_graph_set_signed_retirement_deletions.sqlite3
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_RESTORE_DB_PATH=data/evidence_graph_set_signed_retirement_restores.sqlite3
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_HOLD_DB_PATH=data/evidence_graph_set_signed_retirement_holds.sqlite3
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_DB_PATH=data/evidence_graph_set_signed_retirement_custody.sqlite3
```

The deletion-attempt journal must not equal or hard-link to any authorization, restore, retirement, hold, custody, custody-artifact, signer-key, authorization-only publication, or signed-publication database.

## Eligible restore records

Deletion execution accepts only terminal restore-intent records:

- `completed` with phase `verified`; or
- `cancelled` with phase `planned`.

A completed restore must have an exact `post_bound` custody manifest matching:

- owner ID;
- restore ID;
- snapshot digest;
- target-path digest;
- custody ID;
- custody-manifest digest.

A cancelled unstarted restore may have no custody manifest. If custody exists, it must match the immutable deletion scope exactly.

No planned, running, failed, partially committed, or scope-drifted restore can be seeded for deletion.

## Durable identities

The deletion ID commits:

- authorization ID and complete authorization digest;
- owner ID;
- restore ID;
- snapshot digest;
- target-path digest;
- complete restore-record digest;
- custody-manifest digest when custody exists.

Changing any governed identity produces a different deletion ID or fails scope validation.

## Durable phase model

```text
planned
→ marker_active
→ restore_deleted
→ verified
```

Attempt states are:

```text
planned | running | failed | completed | cancelled
```

The deletion-attempt journal provides:

- exclusive worker leases;
- expired-lease reclaim;
- attempt ceilings;
- phase-preserving failure and retry;
- exact owner/deletion confirmation;
- cancellation only before marker work starts;
- owner-scoped status, listing, and bounded reconciliation.

## Operator workflow

### 1. Produce and authorize a retention candidate

Follow `EVIDENCE_GRAPH_RESTORE_DELETION_AUTHORIZATION.md` and retain the resulting `AUTHORIZATION_ID` and `RESTORE_ID`.

### 2. Seed one deterministic deletion attempt

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletions.py \
  seed AUTHORIZATION_ID \
  --restore-id RESTORE_ID \
  --confirm-authorization-id AUTHORIZATION_ID \
  --confirm-restore-id RESTORE_ID \
  --max-attempts 3
```

Seeding repeats authorization preflight, validates terminal restore scope, validates custody, and writes only the deletion-attempt journal. It does not delete the restore row or consume the authorization.

Incorrect confirmation is rejected before any deletion store is opened.

### 3. Inspect the attempt

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletions.py \
  status DELETION_ID
```

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletions.py \
  list --owner-id alice --limit 100
```

These read commands open only the deletion-attempt journal.

### 4. Execute one attempt

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletions.py \
  execute DELETION_ID \
  --worker-id deletion-worker-1 \
  --lease-seconds 60
```

### 5. Reconcile the next claimable attempt

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletions.py \
  reconcile-one \
  --owner-id alice \
  --worker-id deletion-worker-1 \
  --lease-seconds 60
```

### 6. Retry a failed attempt

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletions.py \
  retry DELETION_ID \
  --owner-id alice \
  --confirm-deletion-id DELETION_ID
```

Retry preserves the last durable phase and is refused after the attempt ceiling.

### 7. Cancel only unstarted work

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletions.py \
  cancel DELETION_ID \
  --owner-id alice \
  --confirm-deletion-id DELETION_ID
```

Cancellation is permitted only while the attempt remains `planned` or `failed` at phase `planned`. Once marker work begins, cancellation is refused and recovery must complete or fail safely.

## Execution protocol

The worker performs the following bounded sequence:

1. Claim the deletion-attempt lease.
2. Reconstruct and verify the authorization record and digest.
3. Revalidate exact custody scope.
4. Create or recover the restore-database deletion marker.
5. Re-run current authorization, retention-candidate, and durable-hold preflight.
6. Reserve the authorization to exactly this deletion ID.
7. Recheck durable holds after reservation.
8. Delete the restore-intent row and commit its immutable tombstone in one restore-database transaction.
9. Record phase `restore_deleted` in the deletion-attempt journal.
10. Mark the authorization terminally `consumed`.
11. Revalidate restore-row absence, tombstone identity, and preserved custody evidence.
12. Complete the deletion attempt at phase `verified`.

The executor never deletes the authorization, hold, custody, receipt, artifact, signature, signer-key, timestamp, marker, or tombstone records.

## Legal-hold and deletion serialization

A simple “check holds twice” protocol is insufficient because a hold could begin before marker creation and commit after the worker’s last hold query.

The implementation therefore uses a durable hold-placement permit in the restore database:

1. Hold placement acquires a permit under `BEGIN IMMEDIATE`.
2. Marker creation also runs under `BEGIN IMMEDIATE` and refuses any active permit.
3. Once the marker is active, new permit acquisition is refused.
4. After the hold-store transaction commits, the exact permit is released.
5. A process death may leave the permit active; exact replay of the same deterministic hold recovers and releases it.
6. A different hold cannot take another hold’s active permit.

This serializes marker activation and hold placement without relying on timing or process-local locks.

A hold committed immediately before marker activation remains visible to execution preflight. A hold attempted after marker activation is refused before it can enter the hold store.

## Authorization reservation and consumption

The authorization store contains a separate one-authorization/one-deletion record:

```text
reserved → consumed
```

Rules:

- one authorization may bind only one deletion ID;
- a different deletion cannot reuse it;
- revocation is refused while it is reserved or consumed;
- reservation may be released only before row deletion when a last-moment legal hold blocks execution;
- after row deletion, consumption is terminal and idempotent.

## Atomic logical deletion and tombstone

The restore row and deletion marker change in one `BEGIN IMMEDIATE` transaction:

- the exact terminal restore row is revalidated by its complete record digest;
- the restore row is deleted;
- the marker changes from `active` to `deleted`;
- a deterministic tombstone digest is recorded.

The tombstone commits only governed identities and digests:

- deletion and authorization identities;
- authorization-consumption digest;
- owner, restore, snapshot, and target digests;
- complete deleted restore-record digest;
- custody ID and custody-manifest digest;
- deletion timestamp.

It contains no source text, raw filesystem path, receipt body, signature bytes, or private key material.

## Crash recovery

### Crash after marker creation

The marker remains active. A retry reclaims the deletion lease, verifies the same immutable marker scope, and continues.

### Crash after authorization reservation

The reservation remains bound to the same deletion ID. Exact retry reuses it. Another deletion and authorization revocation are refused.

### Crash after restore-row deletion before phase persistence

The restore row remains absent and the marker/tombstone remain committed. Retry detects the exact deleted marker, records the missing journal phase, consumes the authorization, and completes.

The restore row is never recreated as compensation.

### Crash after phase `restore_deleted` before completion

Retry revalidates the tombstone and custody, marks the authorization consumed if necessary, and completes.

### Hold appears before deletion

The worker releases the authorization reservation, marks its exact deletion marker `aborted`, and fails the attempt without deleting the restore row. After the hold is released, the same deletion can retry and reactivate the exact aborted marker over the unchanged restore record.

## Preserved evidence

Logical deletion retains:

- restore legal-hold history;
- deletion authorization and revocation history;
- authorization reservation/consumption history;
- pre/post custody receipts;
- custody manifests;
- custody-artifact attempt records;
- backup and receipt artifacts;
- external custody manifests and envelopes;
- HMAC, Ed25519, and RFC 3161 evidence;
- signer-key registration and retirement history;
- deletion-attempt history;
- hold-placement permit history;
- restore-database deletion marker and tombstone.

The target database restored by the original restore operation is not modified.

## Logical deletion is not secure erasure

This executor removes one logical row from the live restore-intent table. SQLite may retain historical bytes in database pages, WAL files, filesystem snapshots, storage-controller caches, backups, and physical media.

The executor deliberately does not:

- run `VACUUM`;
- enable or claim `secure_delete` guarantees;
- delete WAL or journal files manually;
- overwrite storage blocks;
- delete backups or custody artifacts;
- compact authorization, hold, custody, marker, tombstone, or deletion-attempt history;
- claim cryptographic erasure.

Secure deletion and database compaction require a separate governed policy and platform-specific evidence.

## Verification boundary

A reconstructed focused workspace executed the exact authorization, deletion contracts, authorization-consumption table, marker/tombstone mutation, deletion-attempt journal, permit-aware hold boundary, executor, runtime, and CLI with API-faithful stubs only for unrelated repository services.

Result:

```text
13 passed
```

The executed checks cover:

- deterministic terminal deletion identity;
- completed-custody and cancelled-no-custody rules;
- journal lifecycle, retry, cancellation, tamper, and database identity;
- authorization single-use reservation and consumption;
- normal atomic row deletion and tombstone completion;
- crash after authorization reservation;
- crash after row deletion before phase persistence;
- transient legal-hold abort and exact marker reactivation;
- active-marker refusal for new holds and revocation;
- hold-placement permit serialization and exact replay;
- confirmation-before-store and read-only CLI boundaries;
- generic, path-free recovery errors.

The repository contains 18 newly committed deletion-execution contracts. They have not been run together from a fresh exact-current checkout.

Still open:

- complete exact-current pytest, coverage, Ruff, and full-tree compilation;
- independent-process deletion/hold/revocation contention;
- real process-kill tests at every deletion phase;
- SQLite busy/locked, WAL, I/O-error, and disk-full injection;
- Windows, Docker/Compose, and network-filesystem matrices;
- stale permit operator audit and governed recovery tooling;
- secure deletion, compaction, and media-erasure policy.

Release readiness is not claimed.
