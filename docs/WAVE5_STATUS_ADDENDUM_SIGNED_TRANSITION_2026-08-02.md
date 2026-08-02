# Wave 5 status addendum — signed publication transition and retirement preflight

Last updated: 2026-08-02

This addendum supersedes the transition and verification boundary in `WAVE5_STATUS_ADDENDUM_SIGNED_PUBLICATION_PATHS_2026-08-02.md`.

## Implemented

### Assurance-level journal isolation

- [x] Authorization-only and signed publication use separate durable SQLite journals.
- [x] Signed runtime rejects the same canonical path as the authorization-only journal.
- [x] Signed runtime rejects existing hard-link aliases.
- [x] The same deterministic logical operation can exist independently in both journals.
- [x] State changes in one journal do not change the matching operation in the other.
- [x] Signed CLI output identifies the isolated journal assurance boundary.

### Read-only transition audit

- [x] Owner-scoped and optional graph-set-scoped audit.
- [x] Bounded result limit with fail-closed completeness behavior.
- [x] Deterministic, text-free report digest.
- [x] Active-lease classification.
- [x] Expired-running recovery classification aligned with actual journal rules.
- [x] Planned/failed exact-cancel classification.
- [x] Completed authorization-only non-retroactivity classification.
- [x] Duplicate nonterminal detection.
- [x] Completed-signed weaker-duplicate detection.
- [x] Completed twins and cancelled/compensated no-action classification.
- [x] Reconstruction validation for counts, actions, unique operation IDs and report digest.
- [x] Read-only CLI and script entrypoint.

### Expired duplicate retirement preflight

- [x] Exact immutable-scope match across both journals.
- [x] Lease-expiry validation.
- [x] Completed/verified signed-attempt requirement.
- [x] Signed candidate ID/digest validation against immutable graph-set storage.
- [x] Signed candidate authority validation.
- [x] Current pointer inspection.
- [x] Journal-only retirement eligibility classification.
- [x] Signed-pointer-restoration-before-retirement classification.
- [x] Stale signed candidate refusal.
- [x] External pointer change refusal.
- [x] Digest-bound, text-free preflight report.
- [x] Read-only CLI and script entrypoint.

## Operator commands

```bash
python scripts/evidence_graph_set_signed_transition.py audit \
  --owner-id OWNER \
  --graph-set-key GRAPH_SET_KEY
```

```bash
python scripts/evidence_graph_set_signed_retirement.py preflight OPERATION_ID \
  --owner-id OWNER
```

Neither command changes journals, graph-set pointers, graph versions or source data.

## Verification

Executed in reconstructed focused workspaces with live signed logic and minimal stubs only for unrelated repository services:

- **12/12** signed actor runtime checks passed;
- **26/26** signed publication, journal-isolation and transition-audit checks passed;
- **7/7** retirement-preflight checks passed;
- **33/33** publication/transition/preflight checks passed together;
- Python compilation passed for the focused modules.

Two semantic defects were found by review and corrected before this status was recorded:

1. a completed signed operation originally masked a retryable weaker duplicate as non-actionable;
2. an expired `running` attempt was initially described as cancellable even though the journal permits cancellation only for `planned` or `failed` states.

The corrected audit distinguishes active leases, expired recovery, direct cancellation, and retirement preflight.

## Still open

- [ ] A lease-claimed retirement executor with explicit confirmation.
- [ ] Compare-and-swap restoration of the signed pointer before weaker-journal retirement.
- [ ] Crash recovery between pointer restoration and journal retirement.
- [ ] Real SQLite execution of the newest cross-mode same-operation isolation contract in an exact-current checkout.
- [ ] Real multi-process signed/authorization-only contention.
- [ ] Process-kill, disk-full and SQLite write-failure injection.
- [ ] Full exact-current repository pytest, coverage and Ruff.
- [ ] Windows and container matrices.
- [ ] Signed audit export, backup/restore and legal-hold procedures.

## Permanent non-claims

- A transition report is not migration authorization.
- Preflight eligibility is not execution.
- An authorization-only completed graph set cannot be relabeled as signed provenance.
- Journal isolation is not distributed consensus.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
