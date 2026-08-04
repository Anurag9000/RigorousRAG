# Wave 5 status addendum — governed hold-placement permit recovery

Last updated: 2026-08-04

This addendum closes the current Wave 5 item for governed recovery of active restore hold-placement permits that have no committed active original hold.

## Implemented

- [x] Deterministic recovery identity bound to owner, restore, original hold and active-permit digest.
- [x] Process-owned actor binding for every mutating recovery.
- [x] Exact hold-ID and permit-digest confirmation.
- [x] Configurable age gate with a 60-second minimum and 3,600-second default.
- [x] Complete active-permit digest verification before mutation.
- [x] Restore owner and existence revalidation.
- [x] Active/deleted deletion-marker refusal.
- [x] Active-original-hold refusal with exact hold-replay requirement.
- [x] Released-original-hold stale-permit cleanup.
- [x] Missing-original-hold quarantine before permit release.
- [x] Integrity-backed active quarantine hold with deterministic scope and reason.
- [x] Fresh short-lived actor assertion replay of an already committed quarantine hold.
- [x] Permit release and immutable recovery receipt in one restore-database transaction.
- [x] Idempotent recovery receipt replay.
- [x] Strict receipt reconstruction and tamper refusal.
- [x] Read-only receipt status and owner-scoped listing.
- [x] Bad confirmation rejection before store loading.
- [x] Text-free, raw-path-free CLI output.

## Operator commands

Audit the permit before recovery:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletion_operations.py \
  permit-audit \
  --owner-id OWNER
```

Recover one confirmed stale permit:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_hold_permit_recovery.py \
  recover HOLD_ID \
  --owner-id OWNER \
  --confirm-hold-id HOLD_ID \
  --confirm-permit-digest PERMIT_DIGEST \
  --minimum-age-seconds 3600 \
  --actor-id ACTOR_ID
```

Inspect immutable recovery evidence:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_hold_permit_recovery.py \
  status RECOVERY_ID
```

```bash
python scripts/evidence_graph_set_signed_retirement_restore_hold_permit_recovery.py \
  list --owner-id OWNER
```

## Verification evidence

Executed in focused reconstructed workspaces using the committed recovery logic, API-faithful older service boundaries, and real SQLite transactions:

```text
7 core recovery checks passed
1 fresh-actor quarantine replay check passed
3 CLI boundary checks passed
```

Aggregate focused result: **11/11**.

Focused Python compilation passed.

Repository-native contracts committed for this slice:

- five core recovery tests;
- one fresh-actor quarantine replay test;
- three CLI boundary tests.

These repository-native tests have not been executed as part of a fresh exact-current complete repository checkout.

## Permanent boundaries

- An active original hold is never released by permit recovery.
- A missing original hold always creates an active quarantine before permit release.
- A quarantine hold requires separate review and explicit release.
- Recovery does not delete restore history or authorize deletion.
- Recovery does not mutate graph, document, citation, source or restore-target state.
- A focused reconstructed harness is not the complete release matrix.
- Release readiness is not claimed.

## Still open

- Full exact-current repository pytest, coverage, Ruff and full-tree compilation.
- Independent-process hold placement/recovery/deletion contention.
- Process-kill injection between quarantine commit and permit/receipt commit.
- SQLite busy/locked, WAL, I/O-error and disk-full injection.
- Windows path, reparse-point and permission matrices.
- Docker/Compose persistence and restart verification.
- HSM/KMS-backed actor and signer operations.
