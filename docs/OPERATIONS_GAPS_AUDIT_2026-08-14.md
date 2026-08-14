# Operations frontier audit — 2026-08-14

This note records repository-contained controls added after the exact-head SLO and continual-promotion tranche. It deliberately separates deterministic control semantics from external infrastructure certification.

## Closed in this tranche

### Distributed coordination contract

- Monotonic fencing tokens for stale-writer rejection.
- Lease acquisition, renewal, release, expiry and ownership validation.
- At-least-once queue reference semantics.
- Idempotent enqueue keys.
- Visibility timeouts and redelivery.
- Retry delay, acknowledgement, negative acknowledgement and dead-letter handling.
- Protocol boundaries for production lease and durable-queue implementations.

The in-memory implementation is a reference/contract implementation. It is not evidence of multi-node durability or network-partition safety.

### Software supply-chain control plane

- Deterministic SHA-256 artifact manifests.
- Deterministic component/SBOM identity.
- Build provenance records binding repository revision, builder, build configuration, input manifest and output digest.
- Output/provenance verification.
- Normalized vulnerability records and fail-closed severity policy.
- Trivy and pip-audit shaped result adapters.
- Signer/verifier protocol and a local HMAC integrity reference implementation.

The HMAC reference signer is not a replacement for Sigstore, cosign, cloud KMS or HSM-backed release identity.

### Disaster recovery and progressive delivery

- Content-addressed backup manifests and integrity verification.
- Explicit RPO/RTO objectives and restore-rehearsal gates.
- Required-artifact completeness and integrity gates.
- Canary sample, error-rate, p95-latency and quality thresholds.
- Deterministic promote/hold/rollback decisions.
- Idempotent rollback action identities and sequential rollback state transitions.

These controls model and gate recovery/deployment evidence. They do not claim that cross-region restoration or an orchestrator rollback has been executed.

## Already present before this tranche

The current main line already contains privacy-preserving feedback batch lineage and promotion gates, continual-learning safeguards, and multi-window SLO burn-rate evaluation. Those mechanisms were preserved rather than duplicated.

## Next repository-contained implementation frontier

1. Production adapters for the lease and queue protocols, with opt-in integration/contract tests for Redis/Postgres/etcd and one durable broker.
2. CI wiring that emits an SBOM and provenance artifact, runs a real vulnerability scanner, verifies dependency locks, and blocks protected releases on policy violations.
3. Sigstore/cosign or KMS-backed signing adapter and attestation verification at promotion/deployment boundaries.
4. Backup catalog integration across retained source, vector, BM25, graph, model/adapter and policy-state artifacts, including restore ordering and dependency validation.
5. Canary/rollback integration with deployment and migration journals so a failed canary can select a known-good artifact set without bypassing evidence gates.
6. OpenTelemetry exporter bridge, Prometheus-compatible metrics and alert-routing adapters where the existing local telemetry/SLO layer stops.
7. Chaos/fault-injection contract suites for lease loss, duplicate delivery, partial restore, stale snapshots, scanner outages and rollback interruption.

## External certification that remains after repository controls

- Multi-replica fencing under process death, clock skew and network partition.
- Durable-broker redelivery/DLQ behavior under broker restart and partial outage.
- Cloud KMS/HSM rotation, separation of duties and key-revocation exercises.
- Real Sigstore/cosign/OIDC identity and OCI attestation verification.
- Cross-region backup/restore drills with measured RPO/RTO.
- Kubernetes or equivalent canary/rollback exercises.
- Production OpenTelemetry collector, Prometheus/Grafana and paging-route validation.
- Long-duration load, soak and chaos testing on representative CPU/GPU/storage/network capacity.

No item in the external-certification list should be reported as completed solely because the repository now contains its control contract.
