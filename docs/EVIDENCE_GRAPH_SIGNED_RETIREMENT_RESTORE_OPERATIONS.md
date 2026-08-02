# Signed retirement restore operations and retention planning

Last updated: 2026-08-02

This runbook covers read-only operational inspection of crash-recoverable signed-retirement restore intents and conservative retention planning.

Neither command mutates a restore intent, target retirement database, snapshot, graph store, publication store, or filesystem object. There is no delete command.

## 1. Audit restore queues

```bash
python scripts/evidence_graph_set_signed_retirement_restore_operations.py audit \
  --owner-id alice \
  --limit 1000
```

Optional filters:

```bash
--state failed
--snapshot-digest SNAPSHOT_SHA256
--target-path-digest TARGET_PATH_SHA256
```

The audit reads only the restore-intent journal configured by:

```bash
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_RESTORE_DB_PATH
```

It does not open the source snapshot or target retirement database.

## 2. Audit classifications

### `planned_ready`

The restore has not been claimed and can be executed or cancelled according to the normal restore-intent command contract.

### `running_active`

A worker owns an unexpired lease. Do not retry, cancel, or start a competing restore.

### `running_expired_reclaimable`

The lease expired and the attempt ceiling is not exhausted. The normal execute or reconcile path may reclaim the exact durable phase.

### `running_expired_exhausted`

The lease expired but the attempt count reached the configured ceiling. Manual investigation is required. Retention planning never treats this history as disposable.

### `failed_retryable`

The restore failed below its attempt ceiling. Review the durable phase and exact target state before using the explicit retry command.

### `failed_exhausted`

The restore failed at its attempt ceiling. Preserve the intent and target evidence for investigation.

### `completed`

The restore reached `verified`. Completed restore history is retained by default.

### `cancelled`

The restore was cancelled before target work began.

## 3. Privacy and boundedness

Audit output includes only:

- restore ID;
- snapshot digest;
- canonical target-path digest;
- snapshot record count;
- state and phase;
- attempt count and ceiling;
- lease-owner presence, not the worker identity;
- lease expiry and classification;
- target verification digest;
- generic failure type;
- timestamps;
- deterministic report digest.

Raw snapshot paths and target paths are never returned.

The audit fails closed when the journal returns exactly the configured result limit because completeness cannot be established. Narrow the state/digest scope or increase the bounded limit.

## 4. Retention planning

```bash
python scripts/evidence_graph_set_signed_retirement_restore_operations.py retention-plan \
  --owner-id alice \
  --minimum-age-seconds 15552000 \
  --retain-latest-per-target 1 \
  --limit 10000
```

Optional controls:

```bash
--include-completed
--hold-restore-id RESTORE_ID
```

Multiple `--hold-restore-id` arguments are accepted within a bounded limit.

## 5. Conservative retention rules

The planner never selects:

- `planned` restore intents;
- active or expired `running` intents;
- retryable or exhausted `failed` intents;
- terminal records younger than the minimum age;
- held restore IDs;
- the configured newest terminal records for each target-path digest;
- completed restores unless `--include-completed` is explicitly supplied.

By default, only old duplicate cancelled history can become a candidate.

With `--include-completed`, an older completed restore may become a candidate only when it is old enough, not held, and not among the newest protected terminal records for that target.

A retention candidate is planning information only. It does not authorize deletion, secure erasure, vacuuming, compaction, or removal of a snapshot or target database.

## 6. Legal holds

The current hold input is explicit and per-plan. A listed restore ID is always protected and receives reason `legal_hold`.

This is not yet a durable legal-hold registry or an authorization system. Persistent legal-hold governance remains open.

## 7. Commands intentionally absent

There is no operations command for:

- delete;
- purge;
- vacuum;
- compact;
- retry;
- cancel;
- execute;
- restore;
- overwrite;
- merge.

Mutation remains confined to the separately documented restore executor with exact confirmations and crash recovery.

## 8. Verification boundary

Executed in the reconstructed focused workspace:

```text
4 passed
```

Focused compilation passed.

The executed checks cover all eight classifications, snapshot/target digest filters, result-limit refusal, duplicate-ID refusal, legal holds, per-target latest protection, completed-by-default retention, optional completed candidates, path-free CLI output, and the absence of destructive subcommands.

The repository contains five repository-native operations contracts. They have not yet been executed together in a fresh exact-current repository checkout.

Full repository pytest, coverage, Ruff, independent-process contention, Windows, and container verification remain open. Release readiness is not claimed.
