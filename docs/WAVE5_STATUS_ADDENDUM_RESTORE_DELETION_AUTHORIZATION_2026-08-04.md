# Wave 5 status addendum — restore-intent deletion authorization

Last updated: 2026-08-04

This addendum records the exact boundary implemented after the restore legal-hold, custody, artifact, external-signature, and RFC 3161 control planes.

## Implemented

### Integrity-backed authorization history

- [x] Deterministic authorization identity over owner, restore, snapshot, target, historical retention-plan digest, policy digest, and operator idempotency key.
- [x] Complete-row integrity digest in a separate SQLite table.
- [x] Process-owned actor provenance for authorization and revocation.
- [x] Bounded authorization expiry of at most 31 days.
- [x] Monotonic `authorized → revoked` transition with no reactivation.
- [x] Exact idempotent replay and actor-provenance collision refusal.
- [x] Parent/database identity verification and redirect refusal.
- [x] Isolated authorization database with canonical-path and hard-link alias refusal against restore, retirement, hold, custody, custody-artifact, and publication journals.

### Historical and current candidate validation

- [x] Exact reproduction of the supplied retention plan at its supplied generation timestamp.
- [x] Future retention-plan timestamp refusal.
- [x] Durable legal-hold refusal.
- [x] Owner, restore, snapshot, and target-path scope validation.
- [x] Second current-state retention-plan evaluation immediately before authorization.
- [x] Current-candidate and current-scope requirement.
- [x] Completed restores retained unless the policy explicitly includes them.

### Read-only execution preflight

- [x] Authorization status and expiry revalidation.
- [x] Restore existence and immutable scope revalidation.
- [x] Current durable legal-hold revalidation.
- [x] Current retention candidacy revalidation.
- [x] Deterministic report digest reconstruction.
- [x] Text-free and raw-path-free output.
- [x] No restore, hold, custody, target, or authorization mutation.

Preflight dispositions are:

- `authorized_candidate_current`;
- `authorization_revoked`;
- `authorization_expired`;
- `restore_missing`;
- `restore_scope_changed`;
- `durable_legal_hold_active`;
- `no_longer_retention_candidate`.

Only `authorized_candidate_current` is marked eligible for a future deletion executor.

### Operator surface

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletion_authorizations.py authorize ...
python scripts/evidence_graph_set_signed_retirement_restore_deletion_authorizations.py status AUTHORIZATION_ID
python scripts/evidence_graph_set_signed_retirement_restore_deletion_authorizations.py list --owner-id OWNER
python scripts/evidence_graph_set_signed_retirement_restore_deletion_authorizations.py preflight AUTHORIZATION_ID
python scripts/evidence_graph_set_signed_retirement_restore_deletion_authorizations.py revoke ...
```

There is intentionally no `delete`, `execute`, `purge`, `vacuum`, `compact`, or secure-erasure command.

## Configuration

```bash
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DELETION_AUTH_DB_PATH=data/evidence_graph_set_signed_retirement_deletion_authorizations.sqlite3
```

The existing process-owned actor configuration is reused. Caller-supplied actor text alone is never accepted as identity proof.

## Focused verification evidence

A reconstructed workspace executed the exact new authorization, current-state gate, runtime, CLI, SQLite integrity, revocation, and preflight implementation with API-faithful stubs only for older repository services.

Result:

```text
8 passed
```

The checks cover:

1. deterministic identity and exact replay;
2. actor-provenance collision refusal;
3. active legal-hold refusal;
4. historical plan digest and candidate validation;
5. current-candidate and future-plan refusal;
6. exact monotonic revocation;
7. complete-row tamper and database replacement refusal;
8. every preflight disposition;
9. report-digest reconstruction;
10. runtime path-alias refusal;
11. confirmation before store creation;
12. read-only command isolation;
13. absence of a delete command.

This is focused reconstructed evidence, not a complete exact-current repository test run.

## Still open

- [ ] A separate lease-based deletion-attempt journal.
- [ ] Atomic authorization consumption or terminal execution binding.
- [ ] Exact revalidation under the restore-journal write lock.
- [ ] Mandatory preservation and revalidation of legal-hold, custody, receipt, artifact, signature, signer-key, and timestamp evidence.
- [ ] Immutable deletion tombstones containing only governed identities and digests.
- [ ] Crash recovery after restore-row deletion but before deletion-attempt completion.
- [ ] Independent-process deletion/hold/authorization contention tests.
- [ ] SQLite busy, WAL, I/O-error, disk-full, and process-kill injection.
- [ ] Secure deletion and compaction policy.
- [ ] Full exact-current pytest, coverage, Ruff, Windows, and container matrices.

## Permanent non-claims

- Retention candidacy is not deletion authorization.
- Deletion authorization is not deletion execution.
- Preflight eligibility is not authorization consumption.
- Authorization does not bypass legal holds or custody preservation.
- Authorization does not prove secure physical erasure.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
