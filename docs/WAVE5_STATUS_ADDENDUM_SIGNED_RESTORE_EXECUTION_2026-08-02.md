# Wave 5 status addendum — signed retirement restore execution

Last updated: 2026-08-02

This addendum supersedes the restore-execution item in `WAVE5_CURRENT_BACKLOG_2026-08-02.md` and the non-execution boundary in `EVIDENCE_GRAPH_SIGNED_RETIREMENT_RESTORE.md`.

## Implemented

### Immutable restore intent

- [x] Deterministic restore ID over owner, snapshot digest and canonical target-path digest.
- [x] Snapshot record count bound into immutable journal scope.
- [x] Append-only-scope SQLite restore-intent journal.
- [x] Expiring exclusive worker leases and lease reclaim.
- [x] Attempt ceilings, generic failure types, exact retry and exact cancellation.
- [x] Monotonic `planned -> target_committed -> verified` phases.
- [x] Retry preserves the committed-target phase.
- [x] Cancellation is impossible after target work begins.

### Isolated runtime

- [x] Dedicated `EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_RESTORE_DB_PATH`.
- [x] Canonical-path alias refusal.
- [x] Existing hard-link alias refusal.
- [x] Separation from the explicit target, retirement journal, and both publication journals.
- [x] Process-local canonical-path cache.

### Source and target policy

- [x] Descriptor-safe snapshot verification before execution.
- [x] Exact operator digest confirmation before opening the restore-intent database.
- [x] Non-empty snapshot requirement.
- [x] Terminal-only records: `completed` or `cancelled`.
- [x] Owner-scope validation for every record.
- [x] Already initialized target database requirement.
- [x] Globally empty target requirement for a new restore intent.
- [x] Retroactive intent creation over an already restored target refused.
- [x] Raw snapshot and target paths excluded from journal rows and output.

### Atomic target restore

- [x] One `BEGIN IMMEDIATE` target transaction.
- [x] Deterministic retirement-ID insertion order.
- [x] All rows inserted or the entire transaction rolled back.
- [x] Exact target accepted as idempotent crash replay.
- [x] Partial target refused.
- [x] Additional target history refused.
- [x] Immutable and mutable state collisions refused.
- [x] No overwrite, merge, delete or compaction branch.

### Final completion race closure

- [x] Exact target verification under `BEGIN IMMEDIATE`.
- [x] Restore-intent completion while the target write lock remains held.
- [x] External target writers cannot change history between final verification and completion.
- [x] Final target transaction performs no row mutation.

### Crash recovery

- [x] Crash before target commit leaves no target rows.
- [x] Crash after target commit before phase persistence recovers from exact target history.
- [x] Crash after `target_committed` before completion preserves phase and verifies before completion.
- [x] Post-claim snapshot or target-scope drift becomes a durable generic failure.
- [x] Completed restore replay is read-only.

### Operator surface

Implemented commands:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_execute.py seed ...
python scripts/evidence_graph_set_signed_retirement_restore_execute.py status ...
python scripts/evidence_graph_set_signed_retirement_restore_execute.py list ...
python scripts/evidence_graph_set_signed_retirement_restore_execute.py execute ...
python scripts/evidence_graph_set_signed_retirement_restore_execute.py reconcile-one ...
python scripts/evidence_graph_set_signed_retirement_restore_execute.py retry ...
python scripts/evidence_graph_set_signed_retirement_restore_execute.py cancel ...
```

Status and list load only the restore-intent journal. Success and failure output is text-free and path-free.

## Executed focused verification

The reconstructed restore workspace executed the actual new restore contracts, journal, mutation, executor, runtime and CLI with API-faithful stubs only for older retirement snapshot/journal dependencies.

Result:

```text
11 passed
```

Focused compilation passed.

The executed checks cover:

1. deterministic restore identity;
2. journal lifecycle and terminal completion;
3. phase-preserving retry and cancellation boundary;
4. expired lease reclaim;
5. row tamper and database inode replacement refusal;
6. normal atomic target restore;
7. crash after target commit;
8. crash after target-phase persistence;
9. nonterminal, partial and retroactive restore refusal;
10. post-claim target scope drift;
11. final target-lock refusal of concurrent additional history;
12. runtime path and hard-link alias refusal;
13. bad confirmation before journal creation;
14. lazy read-only status/list;
15. generic secret-free recovery output.

The repository contains 17 new repository-native restore contracts. Those tests have not yet been executed together from a fresh exact-current checkout.

## Still open

- [ ] Full exact-current repository pytest and coverage.
- [ ] Ruff and full-tree compilation from a fresh current checkout.
- [ ] Independent-process target and restore-intent contention.
- [ ] Process-kill injection with real subprocesses at both durable phases.
- [ ] SQLite busy/locked and WAL-mode tests.
- [ ] Filesystem I/O, fsync and disk-full injection.
- [ ] Windows permissions, hard-link and reparse-point matrices.
- [ ] Docker/Compose offline-target restore exercise.
- [ ] Mandatory backup-before-restore evidence.
- [ ] Post-restore comparison receipts and chain-of-custody manifests.
- [ ] Asymmetric snapshot signatures and trusted timestamps.
- [ ] Restore-intent operational audit, retention planning and legal holds.

## Permanent non-claims

- Restore does not publish or approve graph relations.
- Restore does not relabel authorization-only history as signed provenance.
- Snapshot checksums are not digital signatures.
- An empty-target restore is not a general merge or import facility.
- Focused reconstructed execution is not the complete release matrix.
- Release readiness is not claimed.
