# Durable pre-restore custody artifact publication

Last updated: 2026-08-03

This runbook covers durable publication of the pre-restore SQLite backup artifact and its matching custody receipt.

The low-level receipt implementation publishes the backup before the JSON receipt. A process failure or concurrent publication race can therefore leave one output without its pair. The durable artifact journal records intent before either output is created and classifies every observed outcome without overwriting or deleting evidence.

## 1. Configuration

```bash
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_ARTIFACT_DB_PATH=data/evidence_graph_set_signed_retirement_custody_artifacts.sqlite3
```

This database must not equal or hard-link to:

- the backup or receipt output;
- the restore target;
- the custody-manifest database;
- the legal-hold database;
- the restore-intent or retirement journals;
- either publication journal.

## 2. Durable identity

One artifact attempt is identified by a deterministic digest over:

- authenticated owner ID;
- terminal snapshot digest;
- target-path digest;
- backup-output-path digest;
- receipt-output-path digest.

Only digests are stored. The journal never stores raw target, backup, receipt, or snapshot paths.

The receipt actor is not accepted from the command line as authority. Execution resolves the existing process-owned reviewer actor binding and records the actor ID, binding method, and binding digest only after a valid pair is verified.

## 3. State machine

```text
planned
  -> running/publication_intent
  -> completed/verified
  -> orphaned/observed
  -> failed
  -> cancelled
```

Terminal orphan dispositions are:

- `backup_without_receipt`;
- `receipt_without_backup`;
- `artifact_collision`.

`orphaned` means immutable evidence remains at one or both operator-selected paths and automatic publication must stop. It does not mean the files were deleted.

## 4. Seed before publication

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_artifacts.py seed \
  --snapshot retirement-snapshot.json \
  --target-db-path data/restore-target.sqlite3 \
  --backup-output evidence/pre-restore.sqlite3 \
  --receipt-output evidence/pre-restore-receipt.json \
  --confirm-snapshot-digest SNAPSHOT_DIGEST
```

Seeding requires:

- a descriptor-verified terminal-only snapshot;
- an initialized target path;
- distinct target, backup, and receipt paths;
- absent backup and receipt outputs;
- exact snapshot-digest confirmation.

The confirmation check occurs before opening or creating the artifact journal.

## 5. Execute or recover

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_artifacts.py execute ARTIFACT_ID \
  --confirm-artifact-id ARTIFACT_ID \
  --snapshot retirement-snapshot.json \
  --target-db-path data/restore-target.sqlite3 \
  --backup-output evidence/pre-restore.sqlite3 \
  --receipt-output evidence/pre-restore-receipt.json \
  --worker-id custody-worker \
  --lease-seconds 60
```

The supplied paths are reduced to digests and must exactly match the durable intent before a lease is claimed.

Execution:

1. claims or reclaims the attempt under an expiring lease;
2. records `publication_intent` durably;
3. inspects any existing outputs before creating files;
4. verifies an existing exact pair and completes idempotently;
5. otherwise invokes the hardened no-overwrite backup/receipt publisher;
6. verifies the live pair and records hashes plus actor provenance;
7. never removes or replaces an existing output.

A completed attempt is not trusted blindly. Every later execute call revalidates the current backup and receipt against the completed journal record. Missing or tampered artifacts fail closed while the immutable journal history remains completed.

## 6. One-command publication

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_artifacts.py publish \
  --snapshot retirement-snapshot.json \
  --target-db-path data/restore-target.sqlite3 \
  --backup-output evidence/pre-restore.sqlite3 \
  --receipt-output evidence/pre-restore-receipt.json \
  --confirm-snapshot-digest SNAPSHOT_DIGEST \
  --worker-id custody-worker
```

`publish` performs seed followed by execute. For recovery after interruption, use the persisted artifact ID with `execute`; do not seed a second operation over existing outputs.

## 7. Crash and race behavior

### Both outputs valid

If the process dies after both outputs are published but before journal completion, the next execute verifies the pair and completes the original attempt.

### Backup only

A backup without a receipt becomes terminal `orphaned/backup_without_receipt`. The file remains untouched for operator investigation.

### Receipt only

A receipt without its backup becomes terminal `orphaned/receipt_without_backup`.

### Both outputs but not one exact pair

The attempt becomes terminal `orphaned/artifact_collision`. Neither output is overwritten or deleted.

This conservative policy avoids manufacturing a receipt after an ambiguous crash and preserves every observable artifact for chain-of-custody review.

## 8. Status and exact journal controls

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_artifacts.py status ARTIFACT_ID
python scripts/evidence_graph_set_signed_retirement_restore_custody_artifacts.py list --owner-id alice
```

Failed attempts may be returned to the queue only with exact confirmation:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_artifacts.py retry ARTIFACT_ID \
  --owner-id alice \
  --confirm-artifact-id ARTIFACT_ID
```

Cancellation is allowed only before durable publication intent:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_artifacts.py cancel ARTIFACT_ID \
  --owner-id alice \
  --confirm-artifact-id ARTIFACT_ID
```

No command deletes or overwrites backup or receipt files.

## 9. Privacy and safety fields

Operator output contains only IDs, digests, byte counts, generic states/failures, lease presence, timestamps, and binding method/digest. Raw paths and receipt actor IDs are not returned.

Execution results permanently report:

```text
artifact_deletion_performed: false
artifact_overwrite_performed: false
source_text_returned: false
raw_path_returned: false
```

## 10. Verification boundary

Repository-native contracts are committed for:

- deterministic identity and row reconstruction;
- lease claim/reclaim, retry ceilings, exact cancellation, and database identity;
- phase-guarded completed/orphan transitions;
- normal pair publication;
- crash after pair publication before journal completion;
- backup-only, receipt-only, and collision classification;
- completed-pair live revalidation and tamper refusal;
- scope mismatch before claim;
- runtime canonical-path and hard-link alias refusal;
- confirmation-before-journal/actor resolution;
- privacy-safe generic CLI errors.

A complete exact-current repository checkout is still unavailable, so these repository-native tests have not yet been executed together. They must not be presented as full-suite evidence.

Release readiness is not claimed.
