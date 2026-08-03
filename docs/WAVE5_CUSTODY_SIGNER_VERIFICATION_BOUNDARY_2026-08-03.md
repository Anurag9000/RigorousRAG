# Wave 5 custody signer verification boundary

Last updated: 2026-08-03

This ledger records the exact evidence boundary for external custody export, HMAC/Ed25519 authentication, signer registry governance, direct and signed administration, rotation planning, compliance auditing, and enforcement-aware signing.

## Executed reconstructed evidence

### External custody export

```text
4 focused checks passed
```

Covered:

- complete-chain construction and privacy reduction;
- incomplete, stale, and live-path-divergent refusal;
- atomic no-overwrite export and digest tamper refusal;
- HMAC round trip, wrong-key refusal, and key-ID pinning.

### Ed25519 custody signatures

```text
3 focused checks passed
```

Covered:

- sign/verify round trip;
- key-ID and public-key-fingerprint pinning;
- atomic no-overwrite output;
- wrong-public-key and signature-tamper refusal;
- protected private-key permissions.

### Public signer registry

```text
3 focused checks passed
```

Covered:

- exact idempotent registration;
- unique per-owner public fingerprints;
- multiple active-key overlap;
- monotonic retirement and replay behavior;
- record and database tamper refusal.

Total executed reconstructed signer-family checks:

```text
10 passed
```

These runs used reconstructed dependency workspaces, not a complete unchanged current repository checkout.

## Committed repository-native contracts

### External custody export and HMAC: 7

- complete deterministic chain and privacy reduction;
- incomplete/stale/path-divergent refusal;
- atomic export and structural tamper refusal;
- HMAC authentication and key controls;
- offline verification without live stores;
- path/actor/secret-free export summary;
- authenticated verification summary.

### Ed25519 envelopes: 4

- public-key sign/verify and no-overwrite;
- wrong-key, fingerprint, key-ID, and signature tamper refusal;
- private-key permissions/type refusal;
- offline secret-free CLI behavior.

### Public signer registry and direct administration: 11

- registry registration/collision contracts;
- overlap and monotonic retirement;
- record/database tamper refusal;
- query-only registry;
- confirmation-before-actor behavior;
- active-only governed signing;
- retired-key historical verification;
- privacy-safe status/list;
- direct process environment/file actor acceptance;
- non-direct/future actor refusal;
- direct-boundary installation.

### Rotation assessment: 4

- deterministic policy and tamper refusal;
- empty/fully-retired global initial-key action;
- age/overlap/count/issuer classifications;
- bounded and non-mutating CLI behavior.

### Signed one-operation administration: 12

- deterministic use identity and expiry;
- one binding digest per action;
- exact monotonic commit;
- database/payload tamper refusal;
- confirmation-before-actor/store;
- reservation before registration;
- retroactive registration backfill refusal;
- exact crash recovery after registration;
- signed retirement confirmation and commit;
- generic expiring credential provenance;
- direct/command-line credential refusal;
- query-only use status.

### Compliance and enforcement: 7

- direct and committed-signed compliance;
- missing/reserved/scope-mismatched use detection;
- registration plus retirement compliance;
- duplicate and bounded refusal;
- noncompliant active-key signing refusal;
- compliant manifest/private-key enforcement;
- historical verification with optional governance requirement.

Total committed signer-family test functions represented above:

```text
45 contracts
```

The 45 contracts have not been executed together from a complete exact-current checkout.

## Current canonical operator paths

```text
external custody export:
  evidence_graph_set_signed_retirement_restore_custody_export.py

direct signer administration:
  evidence_graph_set_signed_retirement_restore_custody_signers_governed.py

signed one-operation administration:
  evidence_graph_set_signed_retirement_restore_custody_signer_admin_governed.py

rotation assessment:
  evidence_graph_set_signed_retirement_restore_custody_signer_operations.py

compliance audit:
  evidence_graph_set_signed_retirement_restore_custody_signer_compliance.py

enforcement-aware signing:
  evidence_graph_set_signed_retirement_restore_custody_signer_enforced.py
```

Historical compatibility scripts are not the canonical production path.

## Exact-current verification still required

- complete repository pytest and coverage;
- Ruff and full-tree compilation;
- production package-import and CLI smoke tests;
- exact dependency installation with `cryptography==49.0.0`;
- independent-process registration/retirement/reservation contention;
- process-kill injection before and after registry/use commits;
- SQLite busy/locked, WAL, I/O, and disk-full injection;
- filesystem-full/fsync/output-race injection;
- Windows key-permission and reparse-point behavior;
- Docker/Compose persistence and restart;
- trusted timestamp and hardware-backed key integration.

## Non-claims

- No full exact-current pytest or CI success is claimed.
- No public-key signature proves scientific correctness.
- Registry compliance does not establish external identity without governed key distribution.
- Historical signature verification does not prove signing time.
- Rotation actions are planning information, not mutation authorization.
- Focused reconstructed checks are not the complete release matrix.
- Release readiness is not claimed.
