# Durable legal holds for signed retirement restore intents

Last updated: 2026-08-02

This runbook covers integrity-backed, process-owned legal holds over signed-retirement restore intents.

A legal hold protects one restore intent from retention candidacy. It does not execute, retry, cancel, overwrite, merge, restore, or delete anything.

## 1. Configuration

```bash
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_HOLD_DB_PATH=data/evidence_graph_set_signed_retirement_holds.sqlite3
```

The hold database must not equal or hard-link to:

- the restore-intent journal;
- the signed-retirement journal;
- the authorization-only publication journal;
- the signed publication journal.

Hold mutation uses the existing process-owned reviewer actor configuration:

```bash
EVIDENCE_GRAPH_REVIEW_ACTOR_ID=
EVIDENCE_GRAPH_REVIEW_ACTOR_ID_PATH=
EVIDENCE_GRAPH_REVIEW_ACTOR_ASSERTION_PATH=
EVIDENCE_GRAPH_REVIEW_ACTOR_HMAC_KEY_PATH=
EVIDENCE_GRAPH_REVIEW_ACTOR_EXPECTED_ISSUER=
```

Exactly one actor source must be configured. Direct environment/file modes and short-lived HMAC assertions retain their existing trust boundaries.

## 2. Hold identity

Each hold episode requires an operator-supplied stable `hold_key`.

The deterministic hold ID commits:

- owner ID;
- restore ID;
- hold key.

Examples of hold keys:

```text
litigation-2026-001
regulator-request-2026-08
incident-review-042
```

A released hold episode can never be reactivated. A later hold on the same restore must use a new hold key.

The reason is a bounded `reason_code`, not free-form text. Example codes:

```text
litigation
regulatory
incident_investigation
records_request
internal_audit
```

## 3. Place a hold

```bash
python scripts/evidence_graph_set_signed_retirement_restore_holds.py place RESTORE_ID \
  --owner-id alice \
  --confirm-restore-id RESTORE_ID \
  --hold-key litigation-2026-001 \
  --reason-code litigation \
  --actor-id reviewer-1
```

`--actor-id` is optional for operator clarity, but when supplied it must exactly match the process-owned actor binding.

Placement requires:

1. exact restore-ID confirmation before opening the hold store;
2. an existing restore intent;
3. exact owner scope;
4. a valid process-owned actor binding;
5. a non-expired signed actor assertion when HMAC mode is used;
6. deterministic hold scope and reason/actor identity compatibility with any replay.

Replaying the same hold key, reason, owner, restore, and creation actor returns the stored record. Changing the reason or creation actor under the same deterministic hold ID is refused as a collision.

Placement never changes the restore intent.

## 4. Complete-row integrity

The compatibility hold table is paired with a companion integrity table.

Every active or released hold has a SHA-256 `hold_digest` over:

- immutable hold scope;
- reason code;
- active/released status;
- creation actor ID, method and binding digest;
- creation timestamp;
- release actor ID, method and binding digest;
- release timestamp;
- schema version.

Creation inserts the hold row and companion digest in one transaction.

Release updates the hold row and companion digest in one transaction.

Every canonical get/list/replay verifies the digest. Missing integrity records fail closed. A database writer that recomputes the digest with an unsupported actor method is still refused by the governed schema boundary.

Canonical operations use `GovernedSignedRetirementRestoreHoldStore`. The lower-level compatibility table implementation is not the operator boundary.

## 5. Release a hold

```bash
python scripts/evidence_graph_set_signed_retirement_restore_holds.py release HOLD_ID \
  --owner-id alice \
  --confirm-hold-id HOLD_ID \
  --actor-id reviewer-2
```

Release requires:

- exact hold-ID confirmation before opening the hold store;
- exact owner scope;
- a valid process-owned actor binding;
- a non-expired signed assertion when HMAC mode is used.

The transition is monotonic:

```text
active -> released
```

There is no reverse transition. Replaying release returns the existing released record and preserves its original release actor and timestamp.

## 6. Read-only inspection

```bash
python scripts/evidence_graph_set_signed_retirement_restore_holds.py status HOLD_ID
```

```bash
python scripts/evidence_graph_set_signed_retirement_restore_holds.py list \
  --owner-id alice \
  --restore-id RESTORE_ID \
  --status active \
  --limit 100
```

```bash
python scripts/evidence_graph_set_signed_retirement_restore_holds.py active-restore-ids \
  --owner-id alice \
  --limit 10000
```

Read commands do not load the restore-intent journal or target retirement database.

Output contains IDs, reason codes, actor binding methods/digests, timestamps, status and record digest. It contains no source text or raw database paths.

## 7. Retention-plan integration

A restore retention plan can consume active durable holds through a SQLite read-only view:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_operations.py retention-plan \
  --owner-id alice \
  --durable-hold-db-path data/evidence_graph_set_signed_retirement_holds.sqlite3 \
  --minimum-age-seconds 15552000 \
  --limit 10000
```

The hold database is opened with SQLite `mode=ro` and `query_only=ON`. Every active hold is integrity- and schema-verified.

Durable hold IDs are unioned with optional one-off inputs:

```bash
--hold-restore-id RESTORE_ID
```

The plan reports `durable_hold_count` and `explicit_hold_count`, but does not return the hold database path.

Neither the hold store nor restore journal is mutated.

## 8. Commands intentionally absent

There is no legal-hold command for:

- delete;
- purge;
- overwrite;
- reactivate;
- edit reason;
- edit actor provenance;
- retry or cancel a restore;
- execute a restore;
- remove target history.

A released episode remains immutable evidence. A later hold requires a new hold key.

## 9. Verification boundary

Executed in the reconstructed focused workspace:

```text
5 passed
```

Focused compilation passed.

The executed checks cover deterministic placement, idempotent replay, collision refusal, monotonic release, non-reactivation, active-ID listing, complete-row tamper detection, missing-integrity refusal, unsupported actor-method refusal, owner scope, exact confirmation, database identity replacement, SQLite read-only hold resolution, and retention protection without hold-store mutation.

The repository contains ten new legal-hold/read-only integration contracts. They have not yet been executed together from a fresh exact-current checkout.

Full repository pytest, coverage, Ruff, Windows, containers, independent-process contention, and external IAM remain open. Release readiness is not claimed.
