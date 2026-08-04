# Wave 5 status addendum — restore-intent deletion execution

Last updated: 2026-08-04

This addendum supersedes the “future deletion executor” boundary recorded in `WAVE5_STATUS_ADDENDUM_RESTORE_DELETION_AUTHORIZATION_2026-08-04.md`.

## Implemented

### Governed logical deletion scope

- [x] Deterministic deletion ID over authorization, restore, snapshot, target, deleted-record, and custody-manifest digests.
- [x] Terminal restore requirement: `completed/verified` or `cancelled/planned` only.
- [x] Mandatory exact `post_bound` custody for completed restores.
- [x] Cancelled unstarted restores may omit custody; unexpected custody must still match scope.
- [x] Separate deletion-attempt journal isolated from authorization, restore, hold, custody, signer, retirement, and publication databases.

### Lease journal and recovery

- [x] Exclusive worker leases and expired-lease reclaim.
- [x] Attempt ceilings and phase-preserving retry.
- [x] Exact owner/deletion confirmation.
- [x] Cancellation only before marker work begins.
- [x] Durable phases `planned → marker_active → restore_deleted → verified`.
- [x] Recovery after marker creation.
- [x] Recovery after authorization reservation.
- [x] Recovery after restore-row deletion before phase persistence.
- [x] Recovery after phase persistence before completion.
- [x] No compensation that recreates a deleted restore row.

### Authorization single use

- [x] One authorization can reserve only one deletion ID.
- [x] Reservation is integrity-bound and idempotent.
- [x] A different deletion cannot reuse the authorization.
- [x] Revocation is refused after reservation or consumption.
- [x] Reservation may be released only before deletion when a legal hold blocks execution.
- [x] Successful logical deletion marks authorization terminally consumed.

### Atomic restore-journal mutation

- [x] Active deletion marker stored in the restore database.
- [x] Exact complete restore-record digest revalidation under `BEGIN IMMEDIATE`.
- [x] Restore-row deletion and marker-to-tombstone transition in one transaction.
- [x] Deterministic text-free deletion tombstone.
- [x] Exact tombstone and row-absence verification before completion.
- [x] Restore target database remains unchanged.

### Legal-hold serialization

- [x] Durable hold-placement permit stored in the restore database.
- [x] Permit acquisition and marker activation serialized by the same restore-database write lock.
- [x] Active permit blocks deletion marker activation.
- [x] Active or deleted marker blocks new permit acquisition.
- [x] Exact same-hold replay recovers a permit left by process death.
- [x] Different holds cannot take another hold’s active permit.
- [x] Last-moment legal hold releases authorization reservation and aborts the exact marker without deleting the restore row.
- [x] Exact retry can reactivate the same aborted marker after the hold is released.

### Preserved evidence

The executor deletes none of the following:

- legal-hold history;
- authorization/revocation history;
- authorization reservation/consumption history;
- custody manifests;
- pre/post receipts;
- backup and receipt artifacts;
- external custody manifests and envelopes;
- HMAC, Ed25519, and RFC 3161 evidence;
- signer-key lifecycle records;
- deletion-attempt history;
- hold-permit history;
- deletion marker and tombstone.

### Operator commands

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletions.py seed ...
python scripts/evidence_graph_set_signed_retirement_restore_deletions.py status DELETION_ID
python scripts/evidence_graph_set_signed_retirement_restore_deletions.py list --owner-id OWNER
python scripts/evidence_graph_set_signed_retirement_restore_deletions.py execute DELETION_ID --worker-id WORKER
python scripts/evidence_graph_set_signed_retirement_restore_deletions.py reconcile-one --owner-id OWNER --worker-id WORKER
python scripts/evidence_graph_set_signed_retirement_restore_deletions.py retry ...
python scripts/evidence_graph_set_signed_retirement_restore_deletions.py cancel ...
```

Read commands open only the deletion-attempt journal. Incorrect seed/retry/cancel confirmations are rejected before durable-store creation.

## Verification evidence

A reconstructed focused workspace executed the exact authorization, current-candidate gate, authorization-consumption table, deletion contracts, marker/tombstone mutation, deletion-attempt journal, executor, permit-aware hold boundary, runtime, and CLI with API-faithful stubs only for unrelated repository services.

Result:

```text
13 passed
```

The repository contains 18 newly committed deletion-execution contracts covering:

- deterministic identity and terminal/custody rules;
- journal lifecycle, retry, cancel, tamper, and database identity;
- normal deletion and tombstone completion;
- authorization single-use behavior;
- both critical deletion crash windows;
- transient hold abort/reactivation;
- active-marker refusal for holds and revocation;
- hold-placement permit serialization and replay;
- confirmation-before-store and read-only CLI boundaries;
- generic path-free error output.

The 18 repository-native contracts have not been executed together from a fresh exact-current checkout.

## Still open

- [ ] Deletion-attempt operational audit and conservative retention planning.
- [ ] Stale hold-placement permit audit and exact governed recovery tooling.
- [ ] Independent-process deletion/hold/revocation contention tests.
- [ ] Process-kill injection at every marker, reservation, delete, consume, and completion phase.
- [ ] SQLite busy/locked, WAL, I/O-error, and disk-full injection.
- [ ] Windows, Docker/Compose, and network-filesystem matrices.
- [ ] Full exact-current pytest, coverage, Ruff, and full-tree compilation.
- [ ] Secure deletion and database compaction policy.
- [ ] Platform-specific SQLite page, WAL, backup, filesystem-snapshot, and physical-media erasure evidence.

## Permanent non-claims

- Retention candidacy is not deletion authorization.
- Deletion authorization is not deletion execution.
- Logical row deletion is not secure physical erasure.
- SQLite page reclamation, WAL erasure, backup deletion, and media overwriting are not performed.
- Custody and legal-hold evidence are preserved rather than erased.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
