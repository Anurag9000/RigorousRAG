# Wave 5 status addendum — durable restore custody artifact publication

Last updated: 2026-08-03

This addendum closes durable tracking of backup/receipt publication races and adds non-destructive artifact operational visibility.

## Implemented

### Durable publication attempts

- [x] Deterministic text-free artifact identity.
- [x] Immutable owner/snapshot/target/output-path digest scope.
- [x] Isolated SQLite attempt journal.
- [x] Lease claim, expiry/reclaim, renewal, retry ceiling, exact retry, and exact cancellation.
- [x] Durable `publication_intent` before artifact creation.
- [x] Terminal transitions rejected before publication intent.
- [x] Normal verified-pair completion.
- [x] Recovery after both outputs exist but journal completion did not persist.
- [x] Backup-only orphan classification.
- [x] Receipt-only orphan classification.
- [x] Artifact-collision orphan classification.
- [x] No automatic artifact deletion or overwrite.
- [x] Completed-pair live revalidation on later execution.
- [x] Completed-pair deletion/tamper refusal without rewriting history.
- [x] Exact path-digest scope validation before lease claim.
- [x] Process-owned receipt actor provenance persisted only after pair verification.

### Operator and read-only boundaries

- [x] Confirmed seed, execute, publish, retry, and cancellation commands.
- [x] Snapshot confirmation before journal creation.
- [x] Artifact confirmation before actor or journal resolution.
- [x] Canonical/hard-link journal alias refusal.
- [x] Query-only SQLite status/list boundary.
- [x] Privacy-safe output without raw paths or receipt actor IDs.
- [x] Generic failure output without private error details.

### Operational audit and retention

- [x] Complete state/lease/orphan classification.
- [x] Derived deterministic restore IDs without schema migration.
- [x] Strict report/plan reconstruction.
- [x] Bounded-result and duplicate-ID refusal.
- [x] Active durable restore-hold integration.
- [x] Orphan evidence permanently excluded from retention candidates.
- [x] Completed pairs retained by default.
- [x] Latest-per-target protection.
- [x] No deletion command.

## Focused execution evidence

Executed in a reconstructed dependency workspace using exact committed artifact contracts, governed SQLite journal, recovery executor, runtime, and query-only view, with minimal stubs only for older unrelated repository services:

```text
19 focused checks passed
```

The executed checks cover:

- deterministic identity and schema reconstruction;
- phase-guarded lifecycle;
- lease reclaim and attempt ceilings;
- normal pair publication;
- crash-after-publication recovery;
- all three orphan dispositions;
- completed-pair live revalidation and tamper refusal;
- pre-claim scope mismatch;
- canonical and hard-link runtime isolation;
- query-only SQLite write refusal.

Focused compilation passed.

Repository-native CLI and operational audit/retention contracts are committed but have not been executed together from a complete exact-current checkout.

## Still open

- [ ] Complete exact-current repository pytest and coverage.
- [ ] Ruff and full-tree compilation from an unchanged current checkout.
- [ ] Independent-process artifact lease/output races.
- [ ] Process-kill injection before and after each file publication.
- [ ] SQLite locked/WAL/I/O/disk-full journal faults.
- [ ] Filesystem full, fsync, hard-link, and permission fault injection.
- [ ] Windows and container artifact persistence matrices.
- [ ] Signed external chain-of-custody export.
- [ ] Asymmetric signatures and trusted timestamps.
- [ ] Destructive-retention authorization and deletion journal.

## Permanent non-claims

- An orphan classification is not cleanup authorization.
- A completed journal row does not override later live-artifact verification failure.
- Integrity digests are not digital signatures.
- A retention candidate is not deletion authorization.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
