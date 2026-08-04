# Restore-intent retention deletion authorization

Last updated: 2026-08-04

This control plane converts one exact restore-intent retention candidate into an expiring, process-owned authorization record. It does **not** delete a restore intent, custody evidence, hold record, artifact, graph set, source file, or database row.

## Safety boundary

Deletion authorization is deliberately separated from retention planning and from any future deletion executor.

An authorization is accepted only when all of the following are true:

1. the restore ID and confirmation ID are identical;
2. the restore belongs to the requested owner;
3. the supplied retention-plan digest is reproduced at its supplied generation timestamp;
4. that historical plan marks exactly one matching restore record as a retention candidate;
5. the restore record, plan item, snapshot digest and target-path digest agree exactly;
6. no active durable legal hold protects the restore;
7. the plan timestamp is not in the future;
8. a second current-state retention plan still marks the restore as a candidate;
9. the configured process-owned actor matches any explicit `--actor-id`;
10. the actor binding is not expired;
11. the authorization has an operator-provided idempotency key and bounded expiry.

The authorization remains only evidence that a qualified actor approved one exact candidate under one exact policy. It is not permission to skip a future deletion preflight, legal-hold check, custody check, backup requirement, or crash-recoverable deletion journal.

## Configuration

```bash
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DELETION_AUTH_DB_PATH=data/evidence_graph_set_signed_retirement_deletion_authorizations.sqlite3
```

The runtime refuses canonical-path or hard-link aliasing with configured restore, retirement, legal-hold, custody, custody-artifact, authorization-only publication, and signed-publication databases.

Authorization and revocation use the existing process-owned actor configuration:

```bash
EVIDENCE_GRAPH_REVIEW_ACTOR_ID=
EVIDENCE_GRAPH_REVIEW_ACTOR_ID_PATH=
EVIDENCE_GRAPH_REVIEW_ACTOR_ASSERTION_PATH=
EVIDENCE_GRAPH_REVIEW_ACTOR_HMAC_KEY_PATH=
EVIDENCE_GRAPH_REVIEW_ACTOR_EXPECTED_ISSUER=
```

Exactly one actor source must be configured. Caller-supplied actor text alone is never sufficient.

## 1. Produce the exact retention plan

Use the restore operations command and include the durable legal-hold database:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_operations.py \
  retention-plan \
  --owner-id alice \
  --minimum-age-seconds 15552000 \
  --retain-latest-per-target 1 \
  --durable-hold-db-path data/evidence_graph_set_signed_retirement_holds.sqlite3 \
  --limit 10000
```

Keep the following output fields together:

- `generated_at`;
- `plan_digest`;
- `minimum_age_seconds`;
- `retain_latest_per_target`;
- `include_completed`;
- the selected item’s restore, snapshot, and target-path digests.

Do not authorize a record that is not explicitly marked `retention_candidate: true`.

## 2. Authorize one candidate

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletion_authorizations.py \
  authorize RESTORE_ID \
  --owner-id alice \
  --confirm-restore-id RESTORE_ID \
  --plan-digest PLAN_DIGEST \
  --plan-generated-at PLAN_GENERATED_AT \
  --authorization-key TICKET_OR_CASE_ID \
  --minimum-age-seconds 15552000 \
  --retain-latest-per-target 1 \
  --expires-in-seconds 86400 \
  --actor-id operator-1
```

Add `--include-completed` only when the source plan used that policy. Completed restores remain protected by default.

The authorization ID commits:

- owner ID;
- restore ID;
- snapshot digest;
- target-path digest;
- historical plan digest;
- deletion-policy digest;
- operator idempotency key.

The complete mutable authorization row is protected by a separate integrity digest. Replaying the same exact command returns the original record. Reusing the same immutable identity with different actor provenance fails as an identity collision.

Authorization expiry is limited to at most 31 days. Shorter windows are preferred.

## 3. Inspect authorization history

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletion_authorizations.py \
  status AUTHORIZATION_ID
```

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletion_authorizations.py \
  list --owner-id alice --restore-id RESTORE_ID --limit 100
```

These commands open only the authorization store. They do not load or mutate the restore journal, legal-hold store, custody stores, target database, or source data.

## 4. Revalidate immediately before any future executor

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletion_authorizations.py \
  preflight AUTHORIZATION_ID --limit 10000
```

Preflight recomputes current state and returns one disposition:

- `authorized_candidate_current` — authorization is active, unexpired, scope-exact, not held, and still a current retention candidate;
- `authorization_revoked`;
- `authorization_expired`;
- `restore_missing`;
- `restore_scope_changed`;
- `durable_legal_hold_active`;
- `no_longer_retention_candidate`.

Only `authorized_candidate_current` is marked `eligible_for_future_deletion_executor: true`. Even that result performs no deletion and creates no deletion attempt.

The report is digest-bound and contains no raw filesystem paths or source text.

## 5. Revoke an authorization

```bash
python scripts/evidence_graph_set_signed_retirement_restore_deletion_authorizations.py \
  revoke AUTHORIZATION_ID \
  --owner-id alice \
  --confirm-authorization-id AUTHORIZATION_ID \
  --actor-id operator-2
```

Revocation is monotonic. A revoked authorization cannot be reactivated. An exact replay returns the original revoked record and does not replace its first release actor or timestamp.

## Durable-state behavior

The authorization database contains:

- one immutable authorization-scope row;
- a monotonic `authorized` or `revoked` status;
- process-owned actor provenance for authorization and optional revocation;
- bounded authorization expiry;
- a complete-row integrity digest.

The store refuses:

- redirecting path components;
- database or parent inode replacement;
- missing or changed integrity rows;
- invalid boolean/database encodings;
- unsupported actor binding methods;
- owner/snapshot/target scope drift;
- active durable holds;
- mismatched or future retention plans;
- plans that are no longer current candidates.

## Explicit non-capabilities

There is no `delete`, `execute`, `purge`, `vacuum`, `compact`, or secure-erasure command in this control plane.

It does not:

- delete restore-intent history;
- delete custody manifests, receipts, artifacts, holds, signatures, or signer records;
- consume an authorization;
- bypass legal holds;
- mutate target retirement databases;
- prove secure physical deletion;
- provide distributed consensus.

A future deletion executor must use a separate lease-based journal, atomically consume an active authorization, re-run this preflight, preserve mandatory custody/legal-hold evidence, record exact deleted identities, and recover every crash window. That executor remains intentionally unimplemented.

## Verification boundary

A reconstructed focused workspace executed the exact new authorization, current-state boundary, runtime, CLI, SQLite integrity and preflight logic with API-faithful stubs only for older repository services:

```text
8 passed
```

Covered behavior includes deterministic identity, idempotent replay, actor collision refusal, active-hold refusal, historical and current-plan checks, future-plan refusal, exact monotonic revocation, complete-row tamper detection, all preflight dispositions, database replacement refusal, runtime path alias refusal, confirmation-before-store behavior, read-only command isolation, report-digest reconstruction, and the absence of a delete command.

This is focused reconstructed evidence, not a complete exact-current repository run. Full pytest, coverage, Ruff, Windows, containers, independent-process contention, SQLite I/O failures, and process-kill testing remain open. Release readiness is not claimed.
