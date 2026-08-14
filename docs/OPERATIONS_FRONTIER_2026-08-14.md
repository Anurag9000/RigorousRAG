# Operational frontier implementation — 2026-08-14

This tranche follows the exact-head green checkpoint `134e27eb7c78647254e71ffa27e38366b3c31abe`.
It adds repository-contained controls that can be exercised deterministically without pretending
that external infrastructure has been certified.

## Implemented

### Feedback lineage and promotion

- Privacy-preserving active-learning batch manifests built only from existing feedback hashes,
  metadata, kinds, weights, and subject IDs.
- Order-independent batch fingerprints and owner-bound batch IDs.
- Explicit candidate/baseline version binding.
- Quality, p95 latency, estimated-cost, sample-size, and negative-feedback-coverage gates.
- Deterministic promotion decision IDs and reason codes suitable for append-only journals.

### SLO and telemetry foundation

- Stage observations with trace IDs, duration, success, token count, estimated cost, and bounded
  attributes.
- Context-manager instrumentation that records both success and exception paths.
- Sliding request-window availability and latency SLO evaluation.
- Explicit error-budget consumption and remaining-budget calculation.
- JSONL export and a callback sink designed to bridge the canonical observation into
  OpenTelemetry/OTLP or another exporter without making telemetry a mandatory runtime dependency.

### Distributed coordination

- Common lease coordinator protocol.
- In-memory fencing coordinator for deterministic tests/local execution.
- Durable SQL compare-and-swap lease coordinator with monotonically increasing fencing tokens;
  SQLite is directly exercised and PostgreSQL-style placeholders/DB-API connections are supported.
- redis-py-compatible distributed lease provider using `SET NX PX` plus compare-and-set Lua
  renewal/release, with monotonically increasing fencing tokens.
- Opaque worker identity generation.

### Software supply chain

- Deterministic release provenance binding commit, dependency lock, SBOM, artifact, image, workflow,
  and run identity.
- Deterministic component inventory suitable as a canonical SBOM input.
- Vulnerability-finding normalization into severity budgets.
- Ed25519 detached signature verification using the repository's pinned `cryptography` dependency.
- Fail-closed release policy for signature, SBOM, hashed lock, critical findings, and high findings.

### Backup, restore, canary, rollback

- Checksum-manifested file backup with generation and optional external encryption-key identity.
- Strict backup-name validation and duplicate-name refusal.
- Tamper-detecting verification before restore.
- Staged restore followed by atomic per-file replacement.
- Canary decision gates over error rate, p95 latency ratio, and quality delta.
- Explicit promote/rollback decision and deterministic reason codes.

## Verification added

`tests/unit/test_operational_frontier.py` covers:

1. deterministic/order-independent feedback batch identity;
2. successful promotion when quality/resource/coverage gates pass;
3. fail-closed promotion on insufficient feedback;
4. telemetry success and exception recording;
5. SLO/error-budget calculation;
6. in-memory fencing after lease expiry;
7. durable SQL fencing across two coordinator instances;
8. supply-chain fail-closed and pass cases;
9. deterministic component inventory;
10. checksum-verified backup/restore;
11. tamper refusal;
12. canary rollback on error/latency/quality regression;
13. vulnerability normalization and Ed25519 verification/tamper refusal.

## Deliberately not claimed as complete

The code above supplies production-oriented contracts, but the following require real external
systems or operator credentials before they can be called certified:

- Redis cluster failover, network partition, eviction, and clock/TTL behavior under load.
- PostgreSQL multi-replica transaction/fencing behavior and failover.
- A real OpenTelemetry collector/OTLP backend, dashboards, alert routing, and retention policy.
- Actual SBOM translation to SPDX/CycloneDX, registry signing, keyless signing or KMS/HSM signing,
  and a selected vulnerability scanner wired to CI/CD.
- Cloud/object-store backups, encrypted snapshot transport, cross-region replication, legal hold,
  retention, secure deletion, RPO/RTO drills, and whole-service restore.
- Canary traffic splitting at an ingress/service-mesh layer and automatic deployment rollback.

These remain external certification work, not hidden TODOs in the repository implementation.
