# Custody signer public-key registry and rotation

Last updated: 2026-08-03

This runbook governs Ed25519 public identities used to sign external restore chain-of-custody manifests.

The registry stores only public evidence:

- owner scope;
- key ID;
- issuer;
- algorithm;
- SHA-256 fingerprint of the raw Ed25519 public key;
- active or retired state;
- process-owned actor-binding provenance;
- registration and retirement timestamps;
- deterministic record digest.

It never stores private-key bytes, public-key bytes, PEM files, or filesystem paths.

## 1. Configuration

```bash
EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_DB_PATH=data/evidence_graph_set_signed_retirement_custody_signers.sqlite3
```

The runtime refuses canonical-path or hard-link aliasing with the restore, retirement, hold, custody, artifact, or publication journals configured in the same process.

Signer administration uses the existing process-owned reviewer actor binding. Naming an actor on the command line is not sufficient authority; any explicit actor ID must match the configured process actor.

## 2. Register a public key

Compute the expected SHA-256 fingerprint of the raw Ed25519 public key through a reviewed provisioning workflow, then register it with exact confirmation:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signers.py register \
  --owner-id alice \
  --key-id custody-ed25519-2026-01 \
  --issuer lab-security \
  --public-key-path custody-ed25519-public.pem \
  --confirm-public-key-sha256 PUBLIC_KEY_SHA256
```

Registration:

1. descriptor-reads a non-redirecting PEM public key;
2. requires Ed25519 type;
3. hashes the raw 32-byte public key;
4. checks exact fingerprint confirmation before actor or registry mutation;
5. records the active public identity and process actor binding;
6. returns an existing exact record idempotently;
7. refuses key-ID or per-owner fingerprint collisions.

The same public key cannot be assigned multiple key IDs for one owner. Rotation uses a genuinely new Ed25519 key and a new key ID.

## 3. Deliberate overlap during rotation

Register the successor key before retiring the predecessor:

```text
key-2026-01 active
key-2026-02 active
```

Multiple active keys are permitted so operators can distribute and pin the successor public key before removing the predecessor from new-signature eligibility.

The registry does not automatically select a key, generate a key, or decide overlap duration.

## 4. Governed signing

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signers.py sign-governed \
  evidence/external-chain.json \
  --owner-id alice \
  --key-id custody-ed25519-2026-02 \
  --private-key-path custody-ed25519-2026-02-private.pem \
  --output evidence/external-chain.ed25519.json
```

Before publishing an envelope, the command requires:

- active registry state;
- manifest owner equal to registry owner;
- private key type Ed25519;
- derived public-key fingerprint equal to the registered fingerprint;
- exact registered key ID.

A retired key cannot create a new governed signature.

Private-key material and paths are never added to the registry or operator output.

## 5. Retire a key

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signers.py retire \
  --owner-id alice \
  --key-id custody-ed25519-2026-01 \
  --confirm-key-id custody-ed25519-2026-01
```

Retirement is monotonic and requires exact key-ID confirmation plus a process-owned actor binding. The record is never deleted.

An exact replay by the same actor binding is idempotent. A different actor cannot overwrite the original retirement provenance.

## 6. Historical verification

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signers.py verify-registered \
  evidence/external-chain.ed25519.json \
  --owner-id alice \
  --key-id custody-ed25519-2026-01 \
  --public-key-path custody-ed25519-2026-01-public.pem
```

Retired records remain usable for historical cryptographic verification. Output explicitly distinguishes:

```text
signature_valid: true
historical_verification_allowed: true
eligible_for_new_signatures: false
```

The supplied public key must match the registry fingerprint and the signed envelope fingerprint.

Because current envelopes do not carry a trusted timestamp, retirement cannot prove whether a historical signature was created before or after the retirement event. The registry therefore supports verification and current signing eligibility, not trusted-time adjudication.

## 7. Read-only inspection

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signers.py status \
  --owner-id alice \
  --key-id custody-ed25519-2026-01

python scripts/evidence_graph_set_signed_retirement_restore_custody_signers.py list \
  --owner-id alice \
  --state active
```

Status and list use a SQLite `mode=ro`, `query_only=ON` view and require an initialized registry. They do not create a missing database.

Summaries omit registration/retirement actor IDs and all key paths. They include binding methods and binding digests for audit correlation.

## 8. Commands deliberately absent

The registry has no:

- private-key import or storage;
- key generation;
- public/private key deletion;
- automatic key selection;
- automatic retirement;
- envelope deletion;
- trusted timestamping;
- HSM/KMS operation;
- transparency-log publication.

## 9. Verification boundary

Repository-native contracts are committed for:

- idempotent registration;
- key-ID and fingerprint collision refusal;
- multiple active-key overlap;
- exact monotonic retirement;
- retirement provenance replay rules;
- record and database tamper refusal;
- query-only status/list;
- confirmation before actor/registry resolution;
- active-only governed signing;
- private-key fingerprint matching;
- retired-key historical verification;
- actor/path/private-material-free output.

A complete unchanged current checkout remains unavailable. Full exact-current pytest, coverage, Ruff, platform/container matrices, independent-process rotation contention, trusted timestamps, issuer overlap policy, and hardware-backed keys remain open.

Release readiness is not claimed.
