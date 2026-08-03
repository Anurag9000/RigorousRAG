# External restore chain-of-custody export

Last updated: 2026-08-03

This runbook covers deterministic external manifests for completed signed-retirement restores and an optional HMAC-authenticated envelope.

The export is evidence-only. It does not import snapshots, execute restores, mutate journals, release legal holds, overwrite custody records, or delete artifacts.

## 1. Export eligibility

A manifest is produced only when all of the following are true:

1. the supplied snapshot is descriptor-verified and terminal-only;
2. the restore ID exactly matches the snapshot owner/digest and target-path digest;
3. the durable restore intent is `completed/verified`;
4. the live target remains an exact restored copy with the recorded verification digest;
5. the supplied pre-restore backup and receipt verify together;
6. the supplied post-restore receipt verifies and matches the current target;
7. exactly one `post_bound` custody manifest matches the restore and both live receipts;
8. at least one completed artifact-journal record matches the live backup/receipt pair;
9. exactly one matching artifact intent binds the supplied live backup and pre-receipt path digests;
10. chronology is:

```text
artifact pair completion
<= pre-bound custody
<= restore completion
<= post-bound custody
<= export generation
```

Incomplete, orphaned-only, stale-target, path-divergent, or scope-divergent chains fail closed.

## 2. Export a deterministic manifest

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_export.py export \
  --restore-id RESTORE_ID \
  --snapshot retirement-snapshot.json \
  --target-db-path data/restore-target.sqlite3 \
  --backup-path evidence/pre-restore.sqlite3 \
  --pre-receipt-path evidence/pre-restore-receipt.json \
  --post-receipt-path evidence/post-restore-receipt.json \
  --output evidence/external-chain.json
```

Optional explicit read-only store paths:

```bash
--restore-db-path RESTORE_INTENT_DB
--custody-db-path CUSTODY_DB
--artifact-db-path ARTIFACT_DB
--hold-db-path HOLD_DB
```

When the hold database is supplied or configured, the manifest records one of:

- `active`;
- `inactive`.

When no hold source is supplied, it records `not_checked`. Hold status is never inferred.

The output is atomically created without overwrite and is mode `0600` on POSIX through the repository's manifest publication primitive.

## 3. Manifest contents

The manifest contains only validated identifiers and evidence:

- owner and deterministic restore ID;
- snapshot and target-path digests;
- restore verification digest and completion time;
- custody ID and manifest digest;
- pre/post receipt digests;
- backup SHA-256 and byte count;
- actor-ID SHA-256 digests, never raw actor IDs;
- actor-binding methods and binding digests;
- custody timestamps;
- exact completed artifact intent IDs and output-path digests;
- legal-hold status;
- deterministic chain digest.

It does not contain:

- raw snapshot, target, backup, receipt, journal, or key paths;
- source or document text;
- raw actor IDs;
- HMAC key material;
- assertion bodies, signatures, or private error text.

## 4. Offline structural verification

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_export.py verify \
  evidence/external-chain.json
```

Offline verification uses descriptor-based bounded reads, duplicate-key refusal, strict schema reconstruction, chronology validation, and deterministic chain-digest validation.

It does not load live journals or mutate any state.

Structural verification proves internal integrity of the exported JSON. It does not prove who produced the file.

## 5. Optional HMAC authentication

Create a protected key file containing at least 32 bytes. On POSIX, it must not grant group or world permissions.

```bash
chmod 600 custody-hmac.key
```

Authenticate a previously verified manifest:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_export.py authenticate \
  evidence/external-chain.json \
  --output evidence/external-chain.auth.json \
  --key-id custody-key-2026-01 \
  --key-path custody-hmac.key
```

Verify the authenticated envelope:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_export.py verify-authenticated \
  evidence/external-chain.auth.json \
  --key-path custody-hmac.key \
  --expected-key-id custody-key-2026-01
```

The envelope uses canonical JSON and HMAC-SHA256 with constant-time tag comparison. It embeds the complete structurally verified manifest and publishes no key bytes.

HMAC demonstrates possession of a shared secret. It is **not** an asymmetric public signature, hardware-backed attestation, trusted timestamp, or non-repudiation mechanism.

## 6. Commands deliberately absent

The script has only:

```text
export
verify
authenticate
verify-authenticated
```

It has no import, restore, replay, merge, overwrite, delete, compact, hold-release, or journal-mutation command.

Every summary reports:

```text
contains_source_text: false
contains_assertion_secrets: false
contains_raw_paths: false
mutation_performed: false
restore_performed: false
import_performed: false
deletion_performed: false
```

## 7. Verification boundary

Repository-native contracts are committed for:

- complete-chain construction;
- completed restore, post-bound custody, and completed artifact requirements;
- live target and receipt revalidation;
- exact live artifact-path binding;
- strict chronology;
- actor-ID reduction;
- legal-hold status;
- deterministic reconstruction and chain-digest tamper refusal;
- atomic no-overwrite export;
- duplicate-key refusal;
- HMAC round trip, wrong-key refusal, key-ID pinning, weak-key refusal, broad-permission refusal, and tag tampering;
- offline CLI verification without live stores;
- path-, actor-, and key-secret-free summaries.

A reconstructed focused harness passed:

```text
4 focused checks passed
```

The harness covered complete-chain construction, incomplete/path-divergent refusal, atomic export and tamper detection, and HMAC round-trip/wrong-key refusal. It used reconstructed dependencies rather than a complete unchanged repository checkout.

Complete exact-current pytest, coverage, Ruff, full-tree compilation, independent-process export races, Windows/container matrices, asymmetric signatures, trusted timestamps, and hardware-backed keys remain open. Release readiness is not claimed.
