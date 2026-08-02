# Signed retirement operational audit and retention planning

Last updated: 2026-08-02

This runbook covers read-only operational visibility for the crash-recoverable signed publication retirement journal.

It does not expose a deletion, purge, retry, cancellation, pointer or publication mutation command.

## 1. Audit retirement work

```bash
python scripts/evidence_graph_set_signed_retirement_operations.py audit \
  --owner-id alice \
  --limit 1000
```

Optionally narrow the report to one logical publication operation:

```bash
python scripts/evidence_graph_set_signed_retirement_operations.py audit \
  --owner-id alice \
  --publication-operation-id OPERATION_ID \
  --limit 1000
```

The report includes only retirement/publication IDs, graph-set keys, state, phase, attempt counts, lease-presence flags, lease expiry, generic failure type, timestamps and classifications.

It does not return:

- source text;
- relation evidence;
- graph node text;
- reviewer assertion bodies;
- signatures or keys;
- queries or provider responses;
- filesystem source paths.

The audit fails closed when the returned row count reaches the configured limit because completeness cannot be established.

## 2. Operational classifications

### `planned_ready`

The retirement has not been claimed and is eligible for normal execution, subject to its attempt ceiling and fresh dependency validation.

### `running_active`

A worker holds an unexpired retirement lease. Do not run a competing executor for the same retirement.

### `running_expired_reclaimable`

The retirement lease expired and the attempt count remains below its ceiling. `execute` or `reconcile-one` may reclaim it while preserving the recorded recovery phase.

### `running_expired_exhausted`

The retirement lease expired, but the attempt ceiling has been reached. Automatic reconciliation cannot claim it. Preserve the record and investigate the failure history and external state.

### `failed_retryable`

The retirement recorded a generic failure and remains below its attempt ceiling. An exact-confirmation retry may return it to `planned` while preserving phase.

### `failed_exhausted`

The retirement failed at its attempt ceiling. It remains durable evidence and is never a retention candidate.

### `completed`

The weaker publication row was retired and terminal verification completed. Completed records are retained by default.

### `cancelled`

The retirement was cancelled before durable pointer intent. It may be considered by conservative retention planning only when it is old, is not held and is not the newest terminal record for its logical publication operation.

## 3. Produce a retention plan

```bash
python scripts/evidence_graph_set_signed_retirement_operations.py retention-plan \
  --owner-id alice \
  --minimum-age-days 180 \
  --retain-latest-per-operation 1 \
  --limit 10000
```

The default policy:

- never considers planned, running or failed records;
- retains completed records;
- retains the newest terminal record for every publication operation;
- retains records younger than the minimum age;
- marks only old, non-held, non-latest cancelled duplicates as candidates;
- performs no deletion.

Completed records may be included only through an explicit planning flag:

```bash
python scripts/evidence_graph_set_signed_retirement_operations.py retention-plan \
  --owner-id alice \
  --minimum-age-days 365 \
  --retain-latest-per-operation 1 \
  --include-completed
```

Even with `--include-completed`, the newest terminal record per operation remains protected.

## 4. Legal holds

Pass one or more exact retirement IDs:

```bash
python scripts/evidence_graph_set_signed_retirement_operations.py retention-plan \
  --owner-id alice \
  --held-retirement-id RETIREMENT_ID_1 \
  --held-retirement-id RETIREMENT_ID_2
```

Held records are never candidates, regardless of age, state or completed-retirement inclusion.

The hold list is an input to the plan only. This command does not create a durable legal-hold registry. Durable hold governance remains separate future work.

## 5. Retention reasons

Each terminal record receives one reason:

- `legal_hold`;
- `latest_terminal_for_operation`;
- `younger_than_minimum_age`;
- `completed_retirements_retained_by_default`;
- `old_terminal_duplicate_candidate`;
- `not_retention_candidate`.

The plan contains a deterministic digest over its scope, policy inputs and ordered items.

## 6. No deletion path

There is intentionally no command such as `delete`, `purge`, `apply-plan` or `vacuum-records`.

Before destructive retention is considered, the repository still requires:

- durable legal-hold storage and authorization;
- signed export of records selected for deletion;
- backup and restore validation;
- referential checks against incident/audit evidence;
- exact operator approval and two-person review policy;
- crash-safe deletion journal;
- retention execution fault injection;
- Windows and container filesystem tests.

A retention candidate is planning information, not deletion authorization.

## 7. Verification boundary

Committed focused contracts cover:

- every state and lease classification;
- retryable and exhausted failures;
- bounded-result refusal;
- duplicate-ID refusal;
- latest-terminal protection;
- legal holds;
- completed-record default retention;
- explicit completed inclusion;
- old terminal duplicate candidates;
- CLI text-free and non-mutating flags.

These newest operational contracts have not yet been run in a fresh exact-current complete repository checkout. Release readiness is not claimed.
