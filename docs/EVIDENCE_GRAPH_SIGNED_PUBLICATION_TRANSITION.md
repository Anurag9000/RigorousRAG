# Transitioning publication attempts to the isolated signed journal

Last updated: 2026-08-02

Signed actor-use publication and authorization-only publication use separate durable journals:

```bash
EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH=data/evidence_graph_set_publications.sqlite3
EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH=data/evidence_graph_set_signed_publications.sqlite3
```

Expired authorization-only duplicates are retired through a third isolated saga journal:

```bash
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DB_PATH=data/evidence_graph_set_signed_retirements.sqlite3
```

All three paths must resolve to distinct files and must not be hard-link aliases.

The separation prevents a signed command from resuming a candidate created before signed actor-use metadata was added. It also prevents retirement recovery state from being mistaken for either publication assurance level.

## 1. Audit authorization-only attempts

Run the read-only transition audit:

```bash
python scripts/evidence_graph_set_signed_transition.py audit \
  --owner-id alice \
  --graph-set-key systematic-review-2026 \
  --limit 1000
```

The audit reads both publication journals and emits a deterministic text-free report containing:

- logical operation ID;
- graph-set key;
- authorization-only state and phase;
- expected-current and candidate set IDs;
- lease state and expiry;
- matching signed-attempt state and phase;
- one bounded action classification;
- report digest.

It performs no pointer, graph-set or journal mutation.

The audit refuses a result set that reaches its configured bound because report completeness cannot be established. Narrow the graph-set key or use a larger bounded limit.

## 2. Transition actions

### `cancel_authorization_only_then_reseed_signed`

The weaker attempt is `planned` or `failed`. These are directly cancellable through the existing exact-confirmation authorization-only command:

```bash
python scripts/evidence_graph_set_publication.py cancel OPERATION_ID \
  --owner-id alice \
  --confirm-operation-id OPERATION_ID
```

Then seed a signed publication using the actual current pointer expectation.

### `wait_for_authorization_only_lease`

The weaker attempt is running under an active lease. Do not steal the lease, edit the database or execute a competing signed transition. Wait for completion or lease expiry and audit again.

### `reconcile_expired_authorization_only_attempt_before_transition`

The weaker attempt is running with an expired lease and no completed signed duplicate has been established. Recover the authorization-only attempt through its normal durable executor before deciding whether a signed replacement is required.

### `do_not_claim_signed_provenance_reseed_with_current_pointer_if_needed`

The authorization-only publication completed. It cannot be relabeled as signed provenance. A new signed publication version is required when signed relation metadata is needed.

### `resolve_duplicate_nonterminal_attempts`

Both journals contain nonterminal attempts for the same operation. Do not execute both. Inspect their exact leases, phases and pointer expectations, then resolve one assurance path deliberately.

### `cancel_authorization_only_duplicate_after_signed_completion`

The signed attempt completed and the weaker duplicate is `planned` or `failed`. Cancel the weaker duplicate with exact operation-ID confirmation. No retirement saga is required.

### `wait_for_authorization_only_lease_then_retire_duplicate`

The signed attempt completed, but the weaker duplicate has an active lease. Wait for lease expiry or worker completion, then audit again.

### `preflight_expired_authorization_only_duplicate_retirement`

The signed attempt completed and the weaker running lease expired. Run the exact read-only retirement preflight.

### `retire_completed_signed_duplicate_authorization_attempt`

A matching signed attempt completed, while the weaker authorization-only duplicate remains capable of retry. Inspect both states and use direct cancellation when the weaker state permits it; use the retirement preflight and saga only for expired `running` duplicates.

### `signed_attempt_already_completed`

Both matching attempts are terminal and the signed attempt completed. Retain immutable history; no transition is required.

### `no_signed_transition_required`

The weaker attempt is already safely terminal and no signed transition action remains.

## 3. Read-only retirement preflight

Inspect one expired running duplicate:

```bash
python scripts/evidence_graph_set_signed_retirement.py preflight OPERATION_ID \
  --owner-id alice
```

The preflight revalidates:

- exact owner, operation, graph-set key, proposals and expected pointer across both journals;
- expired weaker running lease;
- completed and verified signed attempt;
- signed candidate ID and digest;
- current signed-candidate authority;
- current pointer.

Possible eligible results:

### `retire_expired_journal_only`

The authoritative signed candidate is already current. The weaker row can be retired without pointer mutation.

### `restore_signed_pointer_then_retire`

The weaker candidate is current while the signed candidate remains authoritative. The retirement saga must compare-and-swap the pointer back to the signed candidate before retiring the weaker row.

Noneligible results include:

- active weaker lease;
- incomplete signed publication;
- stale signed candidate;
- unrelated current pointer;
- authorization-only row no longer in the expected running state.

The preflight remains read-only and digest-bound.

## 4. Crash-recoverable retirement saga

For an eligible preflight, seed a durable retirement:

```bash
python scripts/evidence_graph_set_signed_retirements.py seed OPERATION_ID \
  --owner-id alice \
  --confirm-operation-id OPERATION_ID
```

Execute one retirement:

```bash
python scripts/evidence_graph_set_signed_retirements.py execute RETIREMENT_ID \
  --worker-id retirement-worker-1 \
  --lease-seconds 60
```

Or reconcile the next claimable retirement:

```bash
python scripts/evidence_graph_set_signed_retirements.py reconcile-one \
  --owner-id alice \
  --worker-id retirement-worker-1 \
  --lease-seconds 60
```

The saga is documented in detail in:

```text
docs/EVIDENCE_GRAPH_SIGNED_PUBLICATION_RETIREMENT_SAGA.md
```

Its durable phases are:

```text
planned
  → pointer_restore_intent
  → pointer_safe
  → authorization_retired
  → verified
```

Core guarantees:

- durable intent precedes pointer mutation;
- the weaker expired lease is taken over under the exact retirement identity;
- retry count on the weaker publication is not incremented;
- the signed pointer is restored only by compare-and-swap;
- a newer post-intent external pointer is preserved;
- the weaker candidate is never restored as compensation;
- only the exact saga lease may cancel the weaker row;
- crashes after pointer commit or weaker cancellation are replayable;
- signed authority is revalidated through terminal verification.

## 5. No automatic journal migration

The retirement saga does not copy authorization-only rows into the signed journal. Copying a row would not rebuild signed actor-use relation metadata and could cross an assurance boundary.

A signed graph-set version must be created by the signed publication path. The retirement saga only neutralizes an expired weaker duplicate after a completed signed version already exists.

Immutable graph-set versions and earlier journal history remain retained.

## 6. Verification evidence

Focused reconstructed workspaces currently provide:

- **12/12** signed assertion, actor-binding and actor-use checks;
- **33/33** signed publication, journal-isolation, transition and preflight checks;
- **12/12** retirement-saga core checks;
- focused Python compilation for the new retirement contracts, journal, mutation, executor, boundary and runtime.

Retirement coverage includes deterministic identity, lease reclaim, retry phase preservation, exact weaker-lease takeover, pointer compare-and-swap, two critical crash windows, external-pointer rules, late authority drift and post-claim failure normalization.

These are focused reconstructed checks with API-faithful stubs for unrelated services, not a fresh exact-current complete repository run. Full pytest, coverage, Ruff, Windows, containers, true multi-process process-kill tests and SQLite I/O/disk-full fault injection remain open. Release readiness is not claimed.
