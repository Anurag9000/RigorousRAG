# Signed retirement restore custody receipts

Last updated: 2026-08-02

This runbook covers process-owned, digest-reconstructed custody receipts around signed-retirement snapshot restore.

Two receipt types exist:

1. a pre-restore SQLite backup receipt for the initialized empty target; and
2. a post-restore comparison receipt binding a completed restore to exact target history and the pre-restore backup.

Receipt operations do not mutate restore intents or retirement rows. The pre-create command creates a separate SQLite backup artifact and receipt file; post-create creates only a receipt file.

## 1. Actor identity

Receipt creation uses the existing process-owned reviewer actor boundary:

```bash
EVIDENCE_GRAPH_REVIEW_ACTOR_ID=
EVIDENCE_GRAPH_REVIEW_ACTOR_ID_PATH=
EVIDENCE_GRAPH_REVIEW_ACTOR_ASSERTION_PATH=
EVIDENCE_GRAPH_REVIEW_ACTOR_HMAC_KEY_PATH=
EVIDENCE_GRAPH_REVIEW_ACTOR_EXPECTED_ISSUER=
```

Supported binding methods are:

- `process_environment`;
- `descriptor_file`;
- `hmac_assertion`.

The actor binding proves control of the configured process identity or shared HMAC key. It does not prove scientific correctness, legal authority, or an external trusted timestamp.

## 2. Create the pre-restore backup and receipt

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody.py pre-create \
  --snapshot /secure/backups/retirements.json \
  --target-db-path /restore/retirements.sqlite3 \
  --backup-output /secure/backups/empty-target.sqlite3 \
  --receipt-output /secure/backups/empty-target.receipt.json \
  --confirm-snapshot-digest SNAPSHOT_SHA256 \
  --actor-id operator-1
```

The exact snapshot digest is checked before loading the actor.

The target must:

- already exist;
- contain the initialized signed-retirement schema;
- be globally empty;
- be a regular, non-redirecting SQLite file.

The backup and receipt output paths must be distinct and absent.

## 3. SQLite backup protocol

The canonical backup boundary uses two source connections:

1. a guard connection obtains `BEGIN IMMEDIATE`, confirms the target is globally empty, and captures the target schema digest;
2. a separate SQLite read-only connection performs `Connection.backup()` into a pre-created mode-0600 temporary artifact.

Using a separate source connection is required because invoking `backup()` from the same connection holding `BEGIN IMMEDIATE` can block.

The temporary artifact is created with:

- `O_EXCL`;
- `O_NOFOLLOW` where available;
- mode `0600`.

Before publication, the backup directory's device/inode identity is revalidated. The artifact is published using an atomic hard link that refuses an existing or concurrently appearing destination, followed by directory `fsync`.

The published backup is reopened and checked for:

- exact empty record count;
- exact schema digest;
- SHA-256 file digest;
- bounded file size.

The receipt is then published atomically with no overwrite.

A concurrent receipt-destination race can leave a valid but unreferenced backup artifact. It is deliberately not auto-deleted without a durable artifact-pair journal, because automatic cleanup could remove valid custody evidence. Operators should retain or investigate such artifacts.

## 4. Pre-restore receipt contents

The pre receipt binds:

- owner ID;
- source snapshot digest;
- canonical target-path digest;
- backup file SHA-256 and byte size;
- target and backup schema digests;
- target and backup record counts, both required to be zero;
- actor ID, binding method and binding digest;
- creation timestamp;
- deterministic receipt digest.

It contains no source text, assertion secret, raw target path, or raw snapshot path.

## 5. Verify pre-restore evidence

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody.py pre-verify \
  --receipt /secure/backups/empty-target.receipt.json \
  --backup /secure/backups/empty-target.sqlite3
```

Verification is actor-free and restore-journal-free. It performs descriptor-safe bounded reads, strict duplicate-key JSON parsing, receipt reconstruction, file SHA-256 verification, and SQLite schema/record-count verification.

Receipt or artifact tampering fails closed.

## 6. Create a post-restore comparison receipt

After the restore intent is completed and verified:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody.py post-create RESTORE_ID \
  --confirm-restore-id RESTORE_ID \
  --snapshot /secure/backups/retirements.json \
  --target-db-path /restore/retirements.sqlite3 \
  --pre-receipt /secure/backups/empty-target.receipt.json \
  --backup /secure/backups/empty-target.sqlite3 \
  --receipt-output /secure/backups/restore-comparison.receipt.json \
  --actor-id auditor-1
```

Post-create requires:

- exact restore-ID confirmation before actor loading;
- a valid pre receipt and matching backup artifact;
- a restore intent in `completed/verified` state;
- exact owner, snapshot digest and target-path digest alignment;
- exact target retirement history equal to the snapshot;
- a valid process-owned actor binding.

## 7. Post-restore receipt contents

The post receipt binds:

- owner and restore IDs;
- snapshot and target-path digests;
- pre-restore receipt digest;
- backup artifact SHA-256;
- exact target verification digest;
- target record count;
- comparison actor ID, method and binding digest;
- comparison timestamp;
- deterministic receipt digest.

It does not contain raw paths or source text.

## 8. Verify a post-restore receipt

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody.py post-verify \
  --receipt /secure/backups/restore-comparison.receipt.json
```

This reconstructs receipt integrity offline. To re-establish current live target equivalence, run post-create again with a new no-overwrite output after verifying the current target.

## 9. Commands intentionally absent

Custody commands do not provide:

- restore execution;
- overwrite or merge;
- target deletion;
- artifact deletion;
- receipt replacement;
- secret/key export;
- legal-hold release;
- trusted timestamp issuance.

## 10. Current integration boundary

Custody receipts are implemented and verifiable, but the low-level restore executor is not yet cryptographically or durably bound to a specific pre-receipt digest and backup artifact.

Therefore:

- operators can create and verify custody evidence today;
- the current restore CLI does not yet require the receipt/backup arguments for every execute or reconcile action;
- mandatory backup-before-restore enforcement requires a durable custody registry or an extension of the immutable restore intent;
- post-receipt creation remains a separate explicit operator step.

This distinction prevents optional evidence tooling from being mislabeled as enforced execution governance.

## 11. Verification boundary

Executed in the reconstructed focused SQLite workspace:

```text
6 passed
```

Focused compilation passed.

The executed checks cover:

- nonblocking two-connection SQLite backup;
- target write-reservation guarding;
- mode-0600 and no-overwrite artifacts;
- empty-target enforcement;
- backup/receipt destination collision refusal;
- receipt and backup tamper refusal;
- completed restore-scope binding;
- exact target comparison;
- target-drift refusal;
- confirmation before actor loading;
- actor-free pre verification.

The repository contains six custody repository-native contracts. They have not yet been executed together from a fresh exact-current checkout.

Full repository pytest, coverage, Ruff, Windows, containers, independent-process contention, filesystem-failure injection, asymmetric signatures and trusted timestamps remain open. Release readiness is not claimed.
