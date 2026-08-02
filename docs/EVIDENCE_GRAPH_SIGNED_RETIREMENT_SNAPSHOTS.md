# Signed retirement snapshot export and verification

Last updated: 2026-08-02

Signed retirement snapshots provide deterministic, text-free audit exports of one owner's retirement journal.

They do not provide restore, import, merge, deletion or migration behavior.

## 1. Export

```bash
python scripts/evidence_graph_set_signed_retirement_snapshot.py export \
  --owner-id alice \
  --output /secure/audit/retirements-2026-08-02.json \
  --limit 10000
```

The export command:

- reads only the owner-scoped retirement journal;
- fails closed when the result reaches the configured bound;
- orders records by retirement ID;
- reconstructs every retirement attempt through its strict contract;
- computes a SHA-256 digest over canonical JSON content;
- writes a mode-0600 temporary inode;
- fsyncs the complete payload;
- publishes through an atomic hard link;
- refuses an existing or concurrently appearing output;
- fsyncs the output directory;
- never changes journal or graph-set state.

The output path may not traverse symlinks or reparse points.

## 2. Verify

```bash
python scripts/evidence_graph_set_signed_retirement_snapshot.py verify \
  /secure/audit/retirements-2026-08-02.json
```

Verification does not load the live retirement journal.

The canonical verification boundary:

- opens the snapshot through a descriptor;
- uses `O_NOFOLLOW` when available;
- requires a regular file;
- enforces a bounded byte size;
- reads exactly the descriptor's initial size;
- detects growth during the read;
- rechecks device, inode and size after reading;
- parses strict UTF-8 JSON;
- rejects duplicate object keys;
- rejects NaN and infinity;
- requires the exact versioned schema;
- reconstructs every retirement attempt;
- checks owner scope, ordering, unique IDs and record count;
- recomputes the snapshot digest;
- validates all safety flags.

Descriptor reading means replacement of the pathname after opening cannot redirect verification to a different inode.

## 3. Snapshot contents

The version-1 snapshot contains:

```text
schema_version
owner_id
generated_at
record_count
records
snapshot_digest
contains_source_text=false
contains_assertion_secrets=false
journal_mutation_performed=false
```

Each record contains the durable retirement fields already stored in the journal:

- retirement and publication operation IDs;
- graph-set key;
- signed and weaker candidate IDs/digests;
- signed authority digest;
- state and recovery phase;
- attempt ceiling and count;
- lease owner/expiry;
- final pointer observation;
- verification digest;
- generic failure type;
- timestamps;
- schema version.

It contains no:

- source document text;
- graph node or relation evidence text;
- query text;
- reviewer assertion body;
- HMAC signature or key;
- source file path;
- provider response.

## 4. Integrity scope

The SHA-256 snapshot digest detects accidental or malicious content modification without the digest being recomputed.

It is not a digital signature and does not establish who created the file. A party able to rewrite the entire snapshot can recompute an unsigned digest.

Still open for stronger audit export:

- asymmetric or hardware-backed signatures;
- signer key IDs and rotation;
- trusted timestamps;
- external transparency logs;
- remote immutable object storage;
- signed chain-of-custody manifests.

## 5. No restore path

There is intentionally no command such as:

```text
restore
import
merge
apply
replace-journal
```

Restoration requires a separate design covering:

- target-journal identity and emptiness/precondition checks;
- collision handling;
- monotonic state constraints;
- attempt and lease safety;
- database transactions and crash recovery;
- signed operator authorization;
- backup-before-restore;
- post-restore full verification;
- multi-process exclusion;
- Windows and container filesystem behavior.

A valid snapshot is audit evidence, not restoration authorization.

## 6. Focused contracts

Committed contracts cover:

- deterministic snapshot construction;
- owner scope and ordered records;
- atomic no-overwrite output;
- mode-0600 output on POSIX;
- successful export/verify round trip;
- checksum tamper refusal;
- duplicate-key refusal;
- path-redirection refusal;
- bounded-result refusal;
- verification without a live journal;
- file growth during descriptor read;
- pathname replacement after descriptor acquisition.

The newest snapshot contracts have not yet been run in a fresh exact-current complete repository checkout. Release readiness is not claimed.
