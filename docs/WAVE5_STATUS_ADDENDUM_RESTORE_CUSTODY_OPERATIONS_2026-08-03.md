# Wave 5 status addendum — restore custody operations

Last updated: 2026-08-03

This addendum closes the custody-manifest operational audit and retention-planning item recorded in `WAVE5_STATUS_ADDENDUM_RESTORE_CUSTODY_2026-08-02.md`.

## Implemented

- [x] Identity-pinned SQLite `mode=ro` custody view.
- [x] `PRAGMA query_only=ON` and already-initialized-schema requirement.
- [x] Owner-scoped custody audit.
- [x] Optional restore, snapshot, target-path-digest, and state filters.
- [x] `pre_bound_pending_post` and `post_bound_complete` classifications.
- [x] Deterministic classification counts and report digest.
- [x] Bounded-result and duplicate-ID refusal.
- [x] No raw paths, actor IDs, binding digests, source text, signatures, or keys.
- [x] Conservative custody retention planning.
- [x] `pre_bound` custody permanently excluded from candidates.
- [x] Completed `post_bound` custody retained by default.
- [x] Minimum-age and latest-per-target protection.
- [x] Explicit planning-only custody holds.
- [x] Integrity-backed active durable restore-hold integration.
- [x] No deletion, overwrite, merge, hold-release, or target-mutation command.

## Operator commands

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_operations.py audit \
  --owner-id OWNER \
  --custody-db-path CUSTODY_DB
```

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_operations.py retention-plan \
  --owner-id OWNER \
  --custody-db-path CUSTODY_DB \
  --durable-hold-db-path HOLD_DB
```

The retention command requires `--include-post-bound` before any completed custody record can become a planning candidate.

## Focused execution evidence

Executed in a reconstructed dependency workspace using the committed custody-operations core, query-only view, CLI, and API-faithful stubs only for unrelated services:

```text
6 passed
```

Covered:

1. custody-state classification;
2. digest-only filtering;
3. deterministic report reconstruction;
4. bounded-result and duplicate refusal;
5. incomplete-custody protection;
6. latest-per-target protection;
7. completed-by-default retention;
8. explicit and durable legal holds;
9. query-only write refusal;
10. uninitialized-schema refusal;
11. privacy-safe CLI output;
12. absence of a destructive command.

Focused compilation passed.

These checks do not constitute a complete exact-current repository run.

## Still open

- [ ] Full exact-current repository pytest and coverage.
- [ ] Ruff and complete tree compilation from an unchanged current checkout.
- [ ] Independent-process custody audit/binding contention.
- [ ] SQLite locked/WAL/I/O/disk-full custody tests.
- [ ] Windows and container custody matrices.
- [ ] Durable orphan-artifact registry for receipt-publication races.
- [ ] Signed external chain-of-custody export.
- [ ] Asymmetric custody signatures and trusted timestamps.
- [ ] Destructive-retention authorization and deletion journal.

## Permanent non-claims

- Custody integrity digests are not digital signatures.
- Durable holds prevent retention candidates; they do not authorize deletion.
- A retention plan is not deletion authorization.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
