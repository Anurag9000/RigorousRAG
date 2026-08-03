# One-operation signed custody signer administration

Last updated: 2026-08-03

This runbook covers short-lived signed actor assertions used to register or retire one Ed25519 custody signer key.

Direct process-owned administration and signed assertion administration are intentionally separate:

```text
direct environment/file actor
  -> evidence_graph_set_signed_retirement_restore_custody_signers_governed.py

signed expiring assertion
  -> evidence_graph_set_signed_retirement_restore_custody_signer_admin_governed.py
```

A signed assertion cannot be reused for multiple signer changes.

## 1. Configuration

```bash
EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_DB_PATH=data/evidence_graph_set_signed_retirement_custody_signers.sqlite3
EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_ADMIN_USE_DB_PATH=data/evidence_graph_set_signed_retirement_custody_signer_admin_uses.sqlite3
```

The administration-use database must not equal or hard-link to the signer registry or any publication, retirement, restore, hold, custody, or artifact journal.

Configure the existing signed reviewer-actor source:

```bash
EVIDENCE_GRAPH_REVIEW_ACTOR_ASSERTION_PATH=review-actor.json
EVIDENCE_GRAPH_REVIEW_ACTOR_HMAC_KEY_PATH=review-actor.key
EVIDENCE_GRAPH_REVIEW_ACTOR_EXPECTED_ISSUER=review-issuer
```

The actor loader must provide:

- a non-direct binding method;
- actor ID;
- binding digest;
- assertion digest or equivalent binding digest;
- issuer;
- exact expiry.

Direct environment/file and command-line identities are refused on this command family.

## 2. Register one key with one assertion

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signer_admin_governed.py register-signed \
  --owner-id alice \
  --key-id custody-ed25519-2026-02 \
  --issuer lab-security \
  --public-key-path custody-ed25519-2026-02-public.pem \
  --confirm-public-key-sha256 PUBLIC_KEY_SHA256
```

The command:

1. validates the Ed25519 public key and exact fingerprint confirmation;
2. resolves and validates the expiring signed actor binding;
3. derives a deterministic action digest over owner, key ID, issuer, algorithm, and public fingerprint;
4. reserves the binding digest for exactly that action;
5. registers the key idempotently;
6. commits the reservation.

The reservation is written before the registry mutation.

## 3. Retire one key with one assertion

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signer_admin_governed.py retire-signed \
  --owner-id alice \
  --key-id custody-ed25519-2026-01 \
  --confirm-key-id custody-ed25519-2026-01
```

The retirement action digest binds the key's immutable registration scope:

- owner;
- key ID;
- issuer;
- algorithm;
- public-key fingerprint;
- original registration binding digest.

The assertion use is reserved before retirement and committed afterward.

## 4. Crash recovery and anti-backfill behavior

### Crash after reservation, before registry mutation

Re-run the exact command with the same signed assertion. The same reservation is reused, the registry action executes idempotently, and the use becomes committed.

### Crash after registry mutation, before use commit

Re-run the exact command with the same assertion. The command requires the pre-existing reservation, validates the exact existing registry action, then commits the use.

### Existing action without prior reservation

The command refuses. A signed assertion cannot be attached retroactively to an earlier direct registration or retirement.

### Same assertion used for another action

The administration-use database has a unique binding-digest constraint. Reusing the assertion for another owner, action, key ID, or action digest fails closed.

## 5. Read-only use status

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signer_admin_governed.py status USE_ID \
  --admin-use-db-path data/evidence_graph_set_signed_retirement_custody_signer_admin_uses.sqlite3
```

Status uses SQLite `mode=ro` and `query_only=ON`. It requires an initialized database and does not create one.

Output contains:

- use ID;
- assertion issuer and expiry;
- binding method;
- owner, action, key ID, and action digest;
- reserved/committed state and timestamps.

It omits:

- assertion body;
- assertion signature;
- actor ID;
- key paths;
- private-key material.

## 6. Credential compatibility boundary

The command accepts any non-direct actor binding only when issuer and expiry provenance are present. This avoids hard-coding one assertion implementation name while still refusing:

```text
process_environment
descriptor_file
command_line
```

Unknown future methods without issuer and expiry fail closed.

## 7. Commands deliberately absent

The signed-administration family has only:

```text
register-signed
retire-signed
status
```

It has no key generation, key import, private-key storage, key deletion, automatic rotation, assertion minting, assertion deletion, or registry compaction command.

## 8. Verification boundary

Repository-native contracts are committed for:

- deterministic use identity;
- assertion expiry and scope validation;
- one binding digest per action;
- exact monotonic reservation commit;
- database identity and payload tamper refusal;
- confirmation before actor/store resolution;
- reservation before registry mutation;
- crash recovery after pre-existing exact actions;
- retroactive-backfill refusal;
- signed retirement confirmation and commit;
- generic signed-method provenance handling;
- direct/command-line refusal;
- query-only status.

These contracts have not been executed together from a complete unchanged current checkout. Full pytest, coverage, Ruff, independent-process contention, process-kill injection, Windows/container matrices, and real signed-assertion provisioning exercises remain open.

Release readiness is not claimed.
