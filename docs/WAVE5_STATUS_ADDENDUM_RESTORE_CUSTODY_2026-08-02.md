# Wave 5 status addendum — custody-governed restore execution

Last updated: 2026-08-02

This addendum supersedes the open custody-binding items in `WAVE5_CURRENT_BACKLOG_2026-08-02.md` and the optional-integration boundary in `EVIDENCE_GRAPH_SIGNED_RETIREMENT_RESTORE_CUSTODY.md`.

## Implemented

### Custody receipt artifacts

- [x] Process-owned pre-restore SQLite backup receipt.
- [x] Process-owned post-restore exact-comparison receipt.
- [x] Two-connection SQLite backup under a target write-reservation guard.
- [x] Mode-0600 temporary and final artifacts on POSIX.
- [x] Atomic hard-link no-overwrite publication.
- [x] Backup parent device/inode revalidation before publication.
- [x] Backup record-count, schema, SHA-256 and byte-size verification.
- [x] Descriptor-safe receipt and backup reads.
- [x] Strict duplicate-key JSON and deterministic receipt reconstruction.
- [x] Exact snapshot and target scope binding.
- [x] No raw paths, source text, or assertion secrets in receipts.

### Durable custody manifest

- [x] Isolated custody database runtime.
- [x] Deterministic custody ID over owner, restore, pre-receipt, and backup digest.
- [x] Complete manifest-row digest reconstruction.
- [x] Monotonic `pre_bound -> post_bound` state.
- [x] Process-owned actor provenance for both transitions.
- [x] Pre-binding allowed only before target work.
- [x] Post-binding allowed only after completed/verified restore.
- [x] Exact replay preserves original audit timestamps.
- [x] Changed evidence or actor provenance is a collision.
- [x] Custody database/target digest alias refusal.
- [x] Runtime canonical-path and existing hard-link alias refusal.
- [x] Read-only status, restore lookup and owner-scoped listing.
- [x] No delete, reset, overwrite, or receipt replacement operation.

### Canonical restore enforcement

The canonical operator entrypoint is:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_governed.py ...
```

- [x] Seed requires pre receipt and backup.
- [x] Seed verifies receipt-target path alignment before restore-intent creation.
- [x] Seed binds the pre evidence under the process-owned actor.
- [x] Execute requires the exact live bound receipt and backup.
- [x] Reconcile requires the exact live bound receipt and backup.
- [x] Receipt/backup substitution or tampering fails before restore claiming.
- [x] Public entrypoint initializes custody with the explicit target path before CLI dispatch.
- [x] Read-only restore status/list remain custody-file independent.
- [x] Low-level restore executor remains available for isolated unit/fault testing.

### Operator commands

Receipt creation and verification:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody.py pre-create ...
python scripts/evidence_graph_set_signed_retirement_restore_custody.py pre-verify ...
python scripts/evidence_graph_set_signed_retirement_restore_custody.py post-create ...
python scripts/evidence_graph_set_signed_retirement_restore_custody.py post-verify ...
```

Custody manifest:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_manifest.py bind-pre ...
python scripts/evidence_graph_set_signed_retirement_restore_custody_manifest.py bind-post ...
python scripts/evidence_graph_set_signed_retirement_restore_custody_manifest.py status ...
python scripts/evidence_graph_set_signed_retirement_restore_custody_manifest.py status-for-restore ...
python scripts/evidence_graph_set_signed_retirement_restore_custody_manifest.py list ...
```

Governed restore:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_governed.py seed ...
python scripts/evidence_graph_set_signed_retirement_restore_governed.py execute ...
python scripts/evidence_graph_set_signed_retirement_restore_governed.py reconcile-one ...
```

Compatibility status/list/retry/cancel remain available through the existing restore script. Target-mutating operator workflows should use the governed entrypoint.

## Configuration

A complete custody environment fragment is committed at:

```text
config/evidence_graph_restore_custody.env.example
```

The custody database variable is:

```bash
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_DB_PATH=data/evidence_graph_set_signed_retirement_custody.sqlite3
```

## Focused execution evidence

### Custody receipt SQLite slice

```text
6 passed
```

Covered nonblocking backup, empty-target enforcement, mode-0600/no-overwrite publication, artifact and receipt tamper refusal, completed restore scope, exact target comparison, target drift, and confirmation ordering.

### Exact live custody-manifest slice

The exact modules and repository-native test were downloaded from current `main` and executed in the reconstructed dependency workspace:

```text
5 passed
```

Covered SQLite manifest insertion/update, replay-stable timestamps, actor/evidence collision refusal, binding-before-target-work, live backup revalidation, post binding, row tamper, target alias, runtime isolation, and confirmation boundaries.

### Exact entrypoint slices

The exact current governed-entrypoint, public-entrypoint, and custody-manifest tests were executed together and passed.

Focused compilation passed for custody contracts, receipt implementation/boundary, manifest store, replay boundary, runtime, entrypoints, and tests.

## Still open

- [ ] Full exact-current repository pytest and coverage.
- [ ] Ruff and complete tree compilation from an unchanged current checkout.
- [ ] Independent-process custody binding and restore contention.
- [ ] Concurrent backup/receipt publication race injection.
- [ ] Durable orphan-artifact registry for a receipt-publication race.
- [ ] SQLite busy/locked, WAL, I/O and disk-full custody tests.
- [ ] Windows descriptor, permission, hard-link, and reparse matrices.
- [ ] Docker/Compose custody-governed restore exercise.
- [ ] Asymmetric receipt signatures and signer key IDs.
- [ ] Trusted timestamps and signer key rotation.
- [ ] Custody-manifest operational audit and conservative retention planning.
- [ ] Signed external chain-of-custody export.

## Permanent non-claims

- SHA-256 custody integrity is not an asymmetric digital signature.
- HMAC actor possession is not external IAM or legal authority.
- Custody evidence does not approve scientific relations.
- Custody binding does not permit overwrite, merge, or target deletion.
- Focused reconstructed execution is not the complete release matrix.
- Release readiness is not claimed.
