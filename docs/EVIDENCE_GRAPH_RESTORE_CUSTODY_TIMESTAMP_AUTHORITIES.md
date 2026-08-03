# Custody timestamp authority attestations

Last updated: 2026-08-03

This subsystem adds a publicly verifiable Ed25519 authority attestation over an exact signed restore chain-of-custody envelope.

It provides an **authority-asserted time**. It is deliberately not described as an RFC 3161 token, a hardware-clock proof, or proof that an external clock source is trustworthy.

## Security and evidence boundary

Each attestation binds:

- owner ID;
- timestamp authority ID and key ID;
- authority public-key SHA-256 fingerprint;
- complete canonical custody-envelope SHA-256 digest;
- custody-manifest digest;
- custody chain digest;
- asserted timestamp;
- random nonce digest;
- deterministic attestation serial;
- Ed25519 signature.

The serial is recomputed from every signed scope field. Modifying the owner, authority, key, custody evidence, asserted time, or nonce digest invalidates reconstruction before cryptographic verification.

The receipt explicitly contains:

```text
rfc3161_token: false
hardware_clock_proven: false
```

## Registry configuration

Use the dedicated configuration example:

```text
config/evidence_graph_custody_timestamp_authority.env.example
```

The primary database variable is:

```bash
EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_AUTHORITY_DB_PATH=data/evidence_graph_set_signed_retirement_custody_timestamp_authorities.sqlite3
```

The registry must not equal or hard-link to custody signer, signer-administration, custody, artifact, hold, restore, retirement, or publication databases.

Private authority keys are never stored in the registry.

## Register an authority key

First calculate the raw Ed25519 public-key fingerprint used by the existing custody key tooling, then register with exact confirmation:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamps.py register \
  --owner-id alice \
  --authority-id institutional-tsa \
  --key-id institutional-tsa-2026-01 \
  --public-key-path institutional-tsa-2026-01.public.pem \
  --confirm-public-key-sha256 PUBLIC_KEY_SHA256
```

Registration requires the configured process-owned or signed review actor. The stored record contains actor binding method and binding digest, but operator summaries omit the actor ID and all key paths.

Exact registration replay with the same owner, authority, key, fingerprint, and actor binding returns the original immutable registration timestamp. Scope divergence fails closed.

## Issue a governed attestation

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamps.py issue-governed \
  evidence/restore-chain.ed25519.json \
  --custody-signer-public-key-path custody-signer.public.pem \
  --owner-id alice \
  --authority-id institutional-tsa \
  --key-id institutional-tsa-2026-01 \
  --authority-private-key-path institutional-tsa-2026-01.private.pem \
  --output evidence/restore-chain.timestamp.json
```

Issuance requires:

1. a cryptographically valid signed custody envelope;
2. envelope owner equal to the requested owner;
3. an active exact authority registry record;
4. private-key-derived fingerprint equal to the registry fingerprint;
5. asserted time not earlier than custody-manifest generation;
6. asserted time not earlier than authority registration;
7. atomic no-overwrite output publication.

The nonce is generated from operating-system randomness. Only its SHA-256 digest is stored.

## Retire an authority key

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamps.py retire \
  --owner-id alice \
  --authority-id institutional-tsa \
  --key-id institutional-tsa-2026-01 \
  --confirm-key-id institutional-tsa-2026-01
```

Retirement is monotonic. A retired key cannot create new attestations.

Historical attestations remain governance-valid only when:

```text
registered_at <= asserted_at <= retired_at
```

An attestation asserted before registration or after retirement is refused even when its Ed25519 signature is mathematically valid.

## Historical verification

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamps.py verify-governed \
  evidence/restore-chain.timestamp.json \
  --signed-envelope-path evidence/restore-chain.ed25519.json \
  --custody-signer-public-key-path custody-signer.public.pem \
  --authority-public-key-path institutional-tsa-2026-01.public.pem \
  --owner-id alice \
  --authority-id institutional-tsa \
  --key-id institutional-tsa-2026-01
```

Verification uses a query-only registry connection and does not initialize a missing registry.

It revalidates:

- the custody signer signature;
- complete envelope digest;
- custody-manifest and chain digests;
- timestamp authority signature;
- owner, authority, key, and fingerprint expectations;
- future-time ceiling;
- registry registration and retirement chronology.

## Read-only inspection

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamps.py status \
  --owner-id alice \
  --authority-id institutional-tsa \
  --key-id institutional-tsa-2026-01
```

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamps.py list \
  --owner-id alice \
  --state active
```

Status, list, and historical verification are query-only. They do not mutate registry state or key material.

## Verification boundary

Focused reconstructed execution passed:

```text
8 passed
```

Covered:

- attestation round trip and atomic no-overwrite output;
- exact custody-envelope, manifest, and chain binding;
- wrong authority key, signature/field tamper, future-time, and chronology refusal;
- deterministic serial and nonce-digest behavior;
- authority registration replay and scope collision;
- active issuance and retired historical verification;
- database payload and file-identity tamper refusal;
- query-only registry write refusal;
- runtime database-alias refusal and caching;
- confirmation before actor/store resolution;
- path, actor-ID, and private-key-free operator output.

Eight repository-native test functions are committed. They have not been executed together from a complete unchanged current repository checkout.

## Permanent non-claims

- The receipt is not an RFC 3161 timestamp token.
- The receipt does not prove a hardware-backed clock.
- The receipt proves authority-key possession and an asserted time, not scientific correctness.
- Registry governance does not establish external institutional identity without governed key distribution.
- Historical verification does not independently prove that the authority clock was accurate.
- Full exact-current pytest, CI, platform, and fault-injection success is not claimed.
- Release readiness is not claimed.
