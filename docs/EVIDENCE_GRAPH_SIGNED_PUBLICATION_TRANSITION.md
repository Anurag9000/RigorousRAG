# Transitioning publication attempts to the isolated signed journal

Last updated: 2026-08-02

Signed actor-use publication uses a dedicated phase journal:

```bash
EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH=data/evidence_graph_set_publications.sqlite3
EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH=data/evidence_graph_set_signed_publications.sqlite3
```

The separation prevents a signed command from resuming a candidate created by the authorization-only command family before signed actor-use metadata was added.

This document describes inspection and planning. The audit and preflight commands are read-only. They do not migrate attempts, cancel leases, change pointers, delete candidates or publish graph sets.

## 1. Audit authorization-only attempts

```bash
python scripts/evidence_graph_set_signed_transition.py audit \
  --owner-id alice \
  --graph-set-key systematic-review-2026 \
  --limit 1000
```

The command reads both journals and emits a deterministic, text-free report containing:

- logical operation ID;
- graph-set key;
- authorization-only state and phase;
- expected-current and candidate set IDs;
- lease status and expiry;
- whether a matching signed attempt exists;
- signed state and phase when present;
- one bounded action classification;
- a report digest.

It never returns relation evidence, source paths, queries, assertion bodies, signatures, keys or document text.

The command fails closed when either journal returns exactly the configured result limit because completeness cannot be established. Narrow the graph-set key or increase the bounded limit.

## 2. Audit action meanings

### `cancel_authorization_only_then_reseed_signed`

The authorization-only attempt is `planned` or `failed`. These are the states accepted by the existing exact-confirmation `cancel` command.

After checking the current graph-set pointer:

```bash
python scripts/evidence_graph_set_publication.py cancel OPERATION_ID \
  --owner-id alice \
  --confirm-operation-id OPERATION_ID
```

Then seed a new signed-journal attempt with an explicit pointer expectation.

### `wait_for_authorization_only_lease`

The weaker attempt is running under an unexpired lease. Do not cancel, retry, re-seed or modify its database. Wait for the worker to finish or the lease to expire, then audit again.

### `reconcile_expired_authorization_only_attempt_before_transition`

The weaker attempt remains `running`, but its lease expired. The journal does not permit direct cancellation of a running attempt. Inspect its status and use the authorization-only recovery path to resolve its pointer/candidate state before re-auditing:

```bash
python scripts/evidence_graph_set_publication.py status OPERATION_ID
python scripts/evidence_graph_set_publication.py execute OPERATION_ID \
  --worker-id recovery-worker \
  --lease-seconds 60
```

Do not run recovery blindly when a stronger signed attempt is already completed; use the duplicate-retirement preflight described below.

### `do_not_claim_signed_provenance_reseed_with_current_pointer_if_needed`

The authorization-only publication already completed. It must not be relabeled as signed provenance. The immutable graph-set version remains what it was.

When a signed-provenance replacement is required, create a new signed attempt using the actual current set ID as the explicit expectation. This creates a new reviewed graph-set version whose relation metadata contains the signed actor-use aggregate.

### `resolve_duplicate_nonterminal_attempts`

Both journals contain nonterminal attempts for the same logical operation. Do not execute both. Inspect both exact statuses, active leases and the graph-set pointer. Resolve the weaker attempt first, then retain or re-seed only the signed operation.

### `cancel_authorization_only_duplicate_after_signed_completion`

The signed attempt completed and the weaker duplicate is `planned` or `failed`. Cancel the weaker duplicate with exact operation-ID confirmation. No pointer change is required.

### `wait_for_authorization_only_lease_then_retire_duplicate`

The signed attempt completed, but the weaker duplicate still has an active running lease. Wait for expiry or worker completion. Never modify the journal database directly.

### `preflight_expired_authorization_only_duplicate_retirement`

The signed attempt completed and the weaker running lease expired. Run the read-only retirement preflight before deciding how to retire the weaker journal record.

### `signed_attempt_already_completed`

Both matching attempts are terminal and the signed attempt completed. No signed transition is required. Retain immutable history.

### `no_signed_transition_required`

The authorization-only attempt is already cancelled or compensated and no matching signed action is needed.

## 3. Retirement preflight for expired weaker duplicates

```bash
python scripts/evidence_graph_set_signed_retirement.py preflight OPERATION_ID \
  --owner-id alice
```

The preflight revalidates:

- identical operation ID, owner, graph-set key, proposal set and expected pointer across the two journals;
- expired authorization-only running lease;
- completed and verified signed attempt;
- exact signed candidate ID and digest in the immutable graph-set store;
- current authority of the signed candidate;
- current graph-set pointer.

The report is read-only and digest-bound.

### `retire_expired_journal_only`

The signed candidate is authoritative and already current. The weaker candidate is not current. The journal record is eligible for a future retirement operation without pointer mutation.

### `restore_signed_pointer_then_retire`

The weaker candidate is currently pointed to, while the completed signed candidate remains authoritative. A future retirement executor must restore the signed candidate with compare-and-swap before retiring the weaker journal record.

The current preflight does **not** perform either action.

### `wait_for_authorization_only_lease`

The lease is not expired. No retirement is eligible.

### `signed_attempt_not_completed`

The matching signed attempt is not completed and verified. No signed retirement is eligible.

### `signed_candidate_not_authoritative`

The signed candidate no longer matches current member generations/graphs. Refuse retirement and investigate graph regeneration or a new signed publication.

### `external_pointer_change_refusal`

The current pointer is neither the signed candidate nor the weaker candidate expected by this duplicate pair. Do not overwrite it. Investigate the external publication first.

### `authorization_attempt_not_running`

The operation no longer matches the expired-running preflight contract. Re-run the transition audit and follow the state-specific action.

## 4. Why there is no automatic migration

Automatic copying of journal rows would not rebuild relation metadata. Automatic replay could also cross an assurance boundary or overwrite a newer pointer.

A safe transition therefore requires:

1. exact state and lease inspection;
2. explicit current/no-current pointer expectation;
3. signed authorization and actor-use validation;
4. candidate reconstruction under the signed path;
5. compare-and-swap pointer activation;
6. post-activation authority verification;
7. immutable retention of earlier candidates and terminal records.

The audit and preflight deliberately stop before mutation.

## 5. Verification evidence

In reconstructed focused workspaces using the live signed modules and minimal stubs only for unrelated repository services:

- **12/12** signed assertion, actor-binding and actor-use checks passed;
- **26/26** signed publication, journal isolation and transition-audit checks passed;
- **7/7** duplicate-retirement preflight checks passed;
- **33/33** signed publication/transition/preflight checks passed together;
- Python compilation passed for the focused modules.

The checks cover report-digest reconstruction, active and expired leases, completed-signed weaker duplicates, result-limit refusal, pointer classifications, signed candidate authority, external pointer refusal and text-free CLI output.

This is not a complete exact-current repository test run. Full pytest, coverage, Ruff, Windows, containers, real multi-process contention, process-kill, disk-full and SQLite write-failure injection remain open. Release readiness is not claimed.
