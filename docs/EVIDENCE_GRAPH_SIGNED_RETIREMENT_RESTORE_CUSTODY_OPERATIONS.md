# Restore custody operational audit and retention planning

Last updated: 2026-08-03

This runbook covers the read-only operational surface for durable signed-retirement restore custody manifests.

Custody manifests bind one restore intent to its pre-restore backup evidence and, after successful restore verification, its post-restore comparison evidence. Operational inspection must not create or mutate custody state, legal holds, restore intents, target databases, receipts, or backups.

## 1. Read-only custody database boundary

The operations CLI opens the custody database through SQLite `mode=ro` with `PRAGMA query_only=ON`.

It requires an already initialized custody database and validates:

- the custody path contains no symlink or reparse-point traversal;
- the parent directory remains the same device/inode;
- the database remains the same regular-file device/inode;
- the custody table is already initialized;
- every returned row reconstructs as a valid deterministic custody manifest.

The audit path therefore does not initialize a missing database or create tables.

## 2. Operational audit

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_operations.py audit \
  --owner-id alice \
  --custody-db-path data/evidence_graph_set_signed_retirement_custody.sqlite3 \
  --limit 1000
```

Optional digest-only filters:

```bash
--restore-id RESTORE_ID
--snapshot-digest SNAPSHOT_DIGEST
--target-path-digest TARGET_PATH_DIGEST
--state pre_bound
--state post_bound
```

The two classifications are:

- `pre_bound_pending_post`: pre-restore custody is bound, but the completed restore/post-comparison evidence has not yet been bound;
- `post_bound_complete`: pre- and post-restore custody evidence are both durably bound.

The report contains only:

- custody, restore, snapshot, and target-path digests;
- custody state and classification;
- backup byte count;
- pre/post binding timestamps and binding methods;
- post-receipt presence;
- deterministic counts and report digest.

It does not return raw target, receipt, backup, or database paths; actor IDs; actor-binding digests; source text; signatures; or keys.

The command fails closed when the bounded result limit is reached because report completeness cannot be established.

## 3. Conservative retention plan

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_operations.py retention-plan \
  --owner-id alice \
  --custody-db-path data/evidence_graph_set_signed_retirement_custody.sqlite3 \
  --minimum-age-seconds 31536000 \
  --retain-latest-per-target 1 \
  --limit 10000
```

Default behavior produces no deletion candidates for completed custody records. To evaluate old duplicate `post_bound` records, the operator must explicitly add:

```bash
--include-post-bound
```

Even then, a record is only a planning candidate when all of the following are true:

1. it is `post_bound`;
2. it is older than the configured minimum age;
3. it is not protected by an explicit custody hold;
4. its restore is not protected by an active durable legal hold;
5. it is not among the newest retained terminal custody records for its target-path digest.

`pre_bound` custody is never a retention candidate because it represents incomplete chain-of-custody history.

Completed custody remains retained by default because a post-bound manifest is the durable link between restore execution and its pre/post evidence.

## 4. Durable restore legal holds

The CLI automatically reads the configured durable restore-hold database when this variable is set:

```bash
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_HOLD_DB_PATH=data/evidence_graph_set_signed_retirement_holds.sqlite3
```

A path may also be supplied explicitly:

```bash
--durable-hold-db-path data/evidence_graph_set_signed_retirement_holds.sqlite3
```

Every custody manifest associated with an actively held `restore_id` is protected.

Emergency planning-only protection may be added with:

```bash
--hold-custody-id CUSTODY_ID
```

Explicit custody holds are not durable legal-hold records. They affect only the generated plan.

The output reports only the counts of explicit custody holds and active durable restore holds. It does not expose hold reasons, actor identities, or hold provenance.

## 5. Safety boundary

The operations script has only:

```text
audit
retention-plan
```

It has no delete, compact, release-hold, overwrite, receipt-replacement, restore, retry, or target-mutation command.

Every output explicitly records:

```text
custody_store_mutation_performed: false
hold_store_mutation_performed: false
restore_mutation_performed: false
target_mutation_performed: false
deletion_performed: false
source_text_returned: false
raw_path_returned: false
```

A retention candidate is not deletion authorization.

## 6. Focused verification boundary

Executed in a reconstructed dependency workspace using the committed custody-operations implementation and API-faithful stubs only for unrelated repository services:

```text
6 passed
```

The focused checks cover:

- `pre_bound` and `post_bound` classification;
- digest-only filters;
- deterministic report and plan reconstruction;
- bounded-result and duplicate-ID refusal;
- incomplete-custody protection;
- latest-per-target protection;
- completed-by-default retention;
- explicit and durable hold protection;
- SQLite query-only write refusal;
- missing-schema refusal;
- privacy-safe CLI output;
- absence of a destructive CLI command.

Focused compilation passed for the new operations core, read-only custody view, CLI, script, and tests.

This is not a full exact-current repository test run. Complete pytest, coverage, Ruff, full-tree compilation, independent-process contention, Windows, and container matrices remain open. Release readiness is not claimed.
