# Restore-deletion operations, retention planning, and permit audit

Last updated: 2026-08-04

This surface provides read-only visibility into restore-deletion attempts and hold-placement permits. It cannot execute, retry, cancel, release, delete, compact, vacuum, or mutate any record.

## Commands

### Deletion queue audit

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletion_operations.py \
  audit --owner-id alice --limit 1000
```

The report classifies every returned deletion attempt as:

- `planned_ready`;
- `running_active`;
- `running_expired_reclaimable`;
- `running_expired_exhausted`;
- `failed_retryable`;
- `failed_exhausted`;
- `completed`;
- `cancelled`.

Output contains only governed IDs/digests, state, phase, attempt counts, lease presence/expiry, marker/tombstone/custody-manifest digests, generic failure type, and timestamps. It never returns source text or raw filesystem paths.

The command fails closed if the bounded result limit is reached or duplicate deletion IDs are returned.

### Conservative deletion-attempt retention plan

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletion_operations.py \
  retention-plan \
  --owner-id alice \
  --minimum-age-seconds 31536000 \
  --retain-latest-per-restore 1 \
  --limit 10000
```

Optional controls:

```bash
--include-completed
--hold-deletion-id DELETION_ID
```

Retention rules are intentionally conservative:

- planned, running, and failed attempts are never candidates;
- completed attempts are retained by default;
- at least the newest terminal attempt per restore ID is protected;
- operator-held deletion IDs are protected;
- only sufficiently old terminal duplicates can become planning candidates;
- the plan never deletes or compacts anything.

`--include-completed` changes planning eligibility only. It does not authorize deletion of completed deletion-attempt evidence.

### Hold-placement permit audit

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletion_operations.py \
  permit-audit --owner-id alice --limit 10000
```

The audit reconstructs and verifies each permit digest from the restore database, checks the corresponding legal-hold record, inspects any deletion marker, and returns one classification:

- `active_permit_with_active_hold` — the hold committed but permit release may have been interrupted; exact replay of the same deterministic hold is the supported recovery path;
- `active_permit_with_released_hold` — the hold is released but its placement permit remains active; governed recovery tooling is required;
- `active_permit_without_hold_record` — permit acquisition occurred but no hold record is visible; investigate an interrupted hold transaction before any release;
- `released_permit_history` — immutable permit history requiring no action.

Only `active_permit_with_active_hold` is marked `exact_hold_replay_recommended: true`. The audit does not release any permit.

## Report integrity

The canonical CLI routes all three reports through a strict reconstruction boundary that verifies:

- owner and timestamp normalization;
- item counts;
- unique deterministic item ordering;
- classification counts;
- retention candidate counts;
- safety flags;
- complete deterministic report or plan digest.

Changing a count, item, safety flag, classification, candidate bit, or digest causes reconstruction failure.

## Non-capabilities

The command family has no:

- `release-permit`;
- `retry`;
- `cancel`;
- `delete`;
- `purge`;
- `vacuum`;
- `compact`;
- secure-erasure operation.

Operational classifications are not execution instructions. Use the dedicated deletion executor for deletion-attempt recovery, and exact hold replay only for the specifically classified committed-hold permit case.

Active permits without a committed active hold require a future governed recovery protocol that proves the hold-store outcome and unchanged restore/deletion scope before releasing the permit. Direct SQLite edits are prohibited.

## Verification boundary

A reconstructed workspace executed the exact deletion-operations, permit-audit, strict reconstruction boundary, and CLI parser with an API-faithful restore-journal stub only for older repository services.

Result:

```text
8 passed
```

The checks cover:

1. all eight deletion queue classifications;
2. bounded-result and duplicate-ID refusal;
3. latest-terminal, completed-by-default, minimum-age, and operator-hold retention rules;
4. retention-plan digest reconstruction;
5. all four permit classifications;
6. permit-digest tamper refusal;
7. permit report safety/digest reconstruction;
8. absence of mutating CLI commands.

This is focused reconstructed evidence, not a complete exact-current repository run. Full pytest, coverage, Ruff, Windows, containers, independent-process contention, and SQLite/process-kill fault injection remain open. Release readiness is not claimed.
