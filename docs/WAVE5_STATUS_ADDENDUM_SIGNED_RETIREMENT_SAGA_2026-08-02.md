# Wave 5 status addendum — crash-recoverable signed publication retirement

Last updated: 2026-08-02

This addendum supersedes the retirement-executor open items in `WAVE5_STATUS_ADDENDUM_SIGNED_TRANSITION_2026-08-02.md`.

## Implemented

### Third isolated retirement journal

- [x] Dedicated SQLite retirement journal.
- [x] Deterministic retirement identity bound to publication operation, candidates and signed authority.
- [x] Canonical-path separation from both publication journals.
- [x] Existing hard-link alias refusal.
- [x] Symlink/reparse path refusal.
- [x] Parent-directory and database-inode identity checks.
- [x] Owner-scoped bounded queue and listing.
- [x] Expiring exclusive leases.
- [x] Attempt ceilings.
- [x] Exact seed replay and scope-collision refusal.
- [x] Retry with phase preservation.
- [x] Cancellation only before durable pointer intent.

### Durable recovery phases

- [x] `planned`.
- [x] `pointer_restore_intent`.
- [x] `pointer_safe`.
- [x] `authorization_retired`.
- [x] terminal `verified`.
- [x] Failure state retains the exact recovery phase.
- [x] Reclaim after retirement-lease expiry.
- [x] Terminal verification digest.

### Weaker publication lease control

- [x] Deterministic weaker lease owner `signed-retirement:<RETIREMENT_ID>`.
- [x] Takeover only after the previous weaker lease expires.
- [x] Exact same-saga renewal.
- [x] Active unrelated worker refusal.
- [x] No weaker retry-count increment during retirement takeover.
- [x] Weaker cancellation only while the exact saga lease is live.
- [x] Idempotent already-cancelled replay.
- [x] Owner/key/candidate scope validation before mutation.

### Pointer safety

- [x] Durable intent before any pointer mutation.
- [x] Pre-intent pointer limited to signed or weaker candidate.
- [x] Compare-and-swap signed-pointer restoration from the exact weaker candidate.
- [x] Already-signed-pointer no-op path.
- [x] Post-intent external pointer preservation.
- [x] Pre-intent external pointer refusal.
- [x] Weaker-pointer reactivation refusal after pointer safety.
- [x] No code path restores the weaker candidate as compensation.
- [x] Final verification requires the weaker candidate not to be current.

### Signed candidate authority

- [x] Signed candidate ID/digest reload before execution.
- [x] Current member-generation and evidence-graph authority validation.
- [x] Exact signed authority-digest match.
- [x] Authority revalidation through terminal verification.
- [x] Late authority drift fails after weaker retirement without reviving weaker state.

### Crash recovery

- [x] Recovery after retirement claim.
- [x] Recovery after durable pointer intent.
- [x] Recovery after signed-pointer commit before pointer-phase persistence.
- [x] Recovery after weaker cancellation before retirement-phase persistence.
- [x] Recovery from `authorization_retired` before terminal verification.
- [x] Raw post-claim failures normalized into durable recovery state.
- [x] One-item reconciler with deterministic queue selection.

### Operator surface

```bash
python scripts/evidence_graph_set_signed_retirements.py seed OPERATION_ID \
  --owner-id OWNER \
  --confirm-operation-id OPERATION_ID

python scripts/evidence_graph_set_signed_retirements.py status RETIREMENT_ID
python scripts/evidence_graph_set_signed_retirements.py list --owner-id OWNER

python scripts/evidence_graph_set_signed_retirements.py execute RETIREMENT_ID \
  --worker-id WORKER \
  --lease-seconds 60

python scripts/evidence_graph_set_signed_retirements.py reconcile-one \
  --owner-id OWNER \
  --worker-id WORKER \
  --lease-seconds 60

python scripts/evidence_graph_set_signed_retirements.py retry RETIREMENT_ID \
  --owner-id OWNER \
  --confirm-retirement-id RETIREMENT_ID

python scripts/evidence_graph_set_signed_retirements.py cancel RETIREMENT_ID \
  --owner-id OWNER \
  --confirm-retirement-id RETIREMENT_ID
```

- [x] Seed/retry/cancel confirmation is checked before opening the retirement store.
- [x] Journal methods revalidate confirmation as defense in depth.
- [x] Status/list do not load graph, generation or publication stores.
- [x] Retry/cancel use only the retirement journal.
- [x] Success and failure outputs contain no source text or secret assertion material.
- [x] Outputs explicitly state that weaker-pointer restoration did not occur.

## Configuration

```bash
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DB_PATH=data/evidence_graph_set_signed_retirements.sqlite3
```

This path must be distinct from:

```bash
EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH
EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH
```

## Focused verification

Executed in a reconstructed workspace with the committed retirement contracts, journal, mutation boundary, executor, failure-normalizing boundary and runtime, plus API-faithful stubs only for unrelated repository services:

- **12/12** retirement-saga core checks passed;
- focused Python compilation passed.

The checks cover:

- deterministic identity;
- journal lifecycle and reclaim;
- retry phase preservation;
- exact weaker lease takeover;
- normal pointer restoration and retirement;
- already-signed-pointer retirement;
- crash after pointer commit;
- crash after weaker cancellation;
- pre-intent external-pointer refusal;
- post-intent external-pointer preservation;
- late authority drift;
- weaker lease renewal before cancellation;
- raw post-claim failure normalization;
- third-journal isolation.

Repository-native contracts have also been committed for the journal, mutation layer, runtime isolation, planning boundary, executor fault matrix, boundary and CLI. They have not yet been run as a complete exact-current repository suite.

## Still open

- [ ] Fresh exact-current complete repository pytest and coverage.
- [ ] Ruff and full-tree static verification.
- [ ] True multi-process contention using independent processes.
- [ ] Process-kill injection at every durable phase.
- [ ] SQLite busy/locked, I/O-error, WAL and disk-full injection.
- [ ] Windows filesystem and reparse-point matrix.
- [ ] Docker/Compose persistence and restart tests.
- [ ] Retirement-journal operational audit and retention planning.
- [ ] Backup/restore, legal-hold and signed audit export procedures.
- [ ] External IAM or asymmetric reviewer assertion integration.

## Permanent non-claims

- Retirement does not convert an authorization-only graph set into signed provenance.
- A signed candidate's authority digest does not establish scientific truth.
- Post-intent external pointers are preserved, not endorsed.
- Journal leases are not distributed consensus.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
