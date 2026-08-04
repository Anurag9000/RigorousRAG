# Governed recovery of restore hold-placement permits

Last updated: 2026-08-04

Restore legal-hold placement is serialized against restore-intent deletion by a permit stored in the restore-intent database. A process can terminate after acquiring that permit and before the legal-hold transaction commits or the permit is released. This runbook describes the governed recovery path for that abandoned state.

Recovery never deletes a restore intent, creates a deletion authorization, executes a restore, changes graph state, or removes legal-hold history.

## 1. Audit permits first

Run the existing read-only permit audit:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletion_operations.py \
  permit-audit \
  --owner-id OWNER \
  --limit 10000
```

The relevant classifications are:

- `active_permit_with_active_hold`: the legal hold exists and is active. Do not use permit recovery. Replay the exact original hold placement so the normal boundary releases the permit.
- `active_permit_with_released_hold`: the hold exists but was subsequently released. The governed recovery command may clean up the stale permit without creating a quarantine hold.
- `active_permit_without_hold_record`: no committed hold exists for the active permit. The governed recovery command creates an active quarantine hold before releasing the permit.
- `released_permit_history`: no recovery is required.

The audit is text-free and path-free. It verifies the complete permit digest before classification.

## 2. Recovery prerequisites

Recovery requires all of the following:

1. the permit remains `active`, or an exact governed receipt proves it was already recovered;
2. the exact hold ID is confirmed;
3. the exact original active-permit digest is confirmed on first execution and every replay;
4. the permit is older than the configured minimum age before first execution;
5. the restore intent still exists under the same owner;
6. no active or completed deletion marker controls the restore;
7. a process-owned reviewer actor binding is configured;
8. the hold, permit and any existing recovery-receipt rows pass their integrity checks.

The default minimum age is 3,600 seconds. The command refuses values below 60 seconds.

## 3. Recover an abandoned permit

```bash
python scripts/evidence_graph_set_signed_retirement_restore_hold_permit_recovery.py \
  recover HOLD_ID \
  --owner-id OWNER \
  --confirm-hold-id HOLD_ID \
  --confirm-permit-digest ORIGINAL_PERMIT_DIGEST \
  --minimum-age-seconds 3600 \
  --actor-id ACTOR_ID
```

`ACTOR_ID` must match the process-owned actor resolved from the configured environment, descriptor file, or short-lived signed assertion.

Bad hold confirmation is rejected before durable stores are opened. A completed recovery replay with a different permit digest is also refused.

## 4. Missing-hold quarantine rule

An active permit without a committed original hold is not released directly. Recovery first commits an integrity-backed active quarantine hold:

- deterministic key: `permit-recovery-<ORIGINAL_HOLD_ID>`;
- reason code: `abandoned_hold_placement_permit`;
- same owner and restore intent as the permit;
- process-owned actor provenance;
- active legal-hold status.

Only after the quarantine hold is durable does recovery release the stale permit and write the immutable recovery receipt.

This ordering ensures that a delayed original hold replay cannot expose the restore intent to deletion between permit release and hold recovery. The quarantine hold remains active until an operator reviews and explicitly releases it through the ordinary legal-hold command family.

A crash after quarantine creation is replayable with a fresh actor assertion. Replay matches the quarantine's deterministic owner, restore, key, reason, integrity record, and active status; it does not require the new short-lived assertion to reproduce the original assertion digest or timestamp.

## 5. Active and released original holds

### Active original hold

Recovery refuses with an exact-replay requirement. Use the same original hold key, reason and actor scope through the normal hold-placement path. That path recognizes the committed hold and releases its active permit.

### Released original hold

Recovery may transition the stale permit to `released` without creating a quarantine hold. The released hold remains immutable history.

## 6. Recovery receipt

Every successful mutation produces an immutable, digest-verified receipt binding:

- deterministic recovery ID;
- owner, restore and original hold IDs;
- original active-permit digest;
- released-permit digest;
- recovery classification;
- quarantine hold ID and digest when applicable;
- process-owned actor identity, method and binding digest;
- recovery timestamp.

The permit transition and receipt insertion commit in one restore-database transaction. Exact replay requires the same original permit digest, returns the existing receipt, and performs no mutation.

Inspect receipts without opening the hold store:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_hold_permit_recovery.py \
  status RECOVERY_ID
```

```bash
python scripts/evidence_graph_set_signed_retirement_restore_hold_permit_recovery.py \
  list \
  --owner-id OWNER \
  --limit 100
```

Status and list commands are read-only and do not create recovery tables when none exist.

## 7. Safety boundaries

Recovery does not:

- delete restore-intent rows;
- release an active original legal hold;
- bypass deletion markers;
- create deletion authorization;
- consume deletion authorization;
- overwrite or merge restore history;
- mutate graph, citation, source or document state;
- return source text or raw filesystem paths.

The quarantine hold is a protection mechanism, not deletion or restore authorization.

## 8. Verification boundary

Focused reconstructed execution using the committed recovery logic and real SQLite transactions passed:

- **7/7** core recovery checks;
- **1/1** fresh-actor quarantine crash-replay check;
- **1/1** wrong-digest completed-replay refusal check;
- **3/3** CLI boundary checks;
- focused Python compilation.

Aggregate focused result: **12/12**.

The checks cover missing-hold quarantine, active-hold refusal, released-hold cleanup, age and marker refusal, exact replay, exact replay confirmation, receipt tamper refusal, fresh signed-actor replay, pre-store confirmation, process-owned actor dispatch, and read-only status/list isolation.

This is not a complete exact-current repository pytest or platform matrix. Release readiness is not claimed.
