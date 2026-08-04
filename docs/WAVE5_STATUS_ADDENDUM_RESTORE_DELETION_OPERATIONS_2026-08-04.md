# Wave 5 status addendum — restore-deletion operations and permit audit

Last updated: 2026-08-04

## Implemented

- [x] Owner-scoped deletion-attempt operational audit.
- [x] Planned/running/failed/completed/cancelled classification.
- [x] Active and expired lease distinction.
- [x] Reclaimable and exhausted attempt distinction.
- [x] Bounded completeness and duplicate-ID refusal.
- [x] Conservative deletion-attempt retention planning.
- [x] Completed deletions retained by default.
- [x] Newest terminal attempt protected per restore ID.
- [x] Operator-held deletion IDs protected.
- [x] No nonterminal or failed deletion attempt can become a retention candidate.
- [x] Read-only restore hold-placement permit audit.
- [x] Complete permit-digest reconstruction and tamper refusal.
- [x] Active-hold, released-hold, missing-hold, and released-history classifications.
- [x] Exact-hold-replay recommendation only for committed active holds.
- [x] Strict report/plan reconstruction for counts, ordering, safety flags, and digests.
- [x] Text-free and raw-path-free output.
- [x] No permit release, retry, cancellation, deletion, compaction, or repair command.

## Operator commands

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletion_operations.py \
  audit --owner-id OWNER
```

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletion_operations.py \
  retention-plan --owner-id OWNER
```

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletion_operations.py \
  permit-audit --owner-id OWNER
```

## Focused verification

A reconstructed focused workspace executed the exact new operations, permit-audit, reconstruction boundary, and CLI parser with an API-faithful stub only for the older restore journal.

```text
8 passed
```

The checks cover all operational and permit classifications, bounded/duplicate refusal, conservative retention, operator holds, permit tampering, report/plan tampering, and absence of mutating commands.

## Still open

- [ ] Governed recovery of an active permit whose hold is missing or already released.
- [ ] Independent-process permit/marker/hold contention.
- [ ] Process-kill injection around permit acquisition, hold commit, permit release, marker activation, and marker abort.
- [ ] SQLite busy/locked, WAL, I/O-error, and disk-full injection.
- [ ] Full exact-current repository pytest, coverage, Ruff, and full-tree compilation.
- [ ] Windows, Docker/Compose, and network-filesystem matrices.
- [ ] Secure deletion and compaction policy.

## Non-claims

- Operational classification is not execution authorization.
- A permit audit is not permit-release authorization.
- Retention candidates are not deletion authorization.
- No secure erasure or database compaction is performed.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
