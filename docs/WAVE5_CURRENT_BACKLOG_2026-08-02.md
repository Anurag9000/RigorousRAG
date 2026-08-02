# Wave 5 current implementation backlog

Last updated: 2026-08-02

This file supersedes the Wave 5 checkbox section in `docs/TODO.md`. The older file remains useful as historical planning context but no longer reflects the current evidence-graph implementation.

## Completed foundation

- [x] Generation-scoped typed evidence graphs.
- [x] Deterministic structural graph construction and tombstones.
- [x] Immutable graph versions and current pointers.
- [x] Exact authoritative graph readers.
- [x] Durable graph-generation jobs and reconciliation.
- [x] Job leases, retries, cancellation, audit and retention planning.
- [x] Cross-document graph-set versions with exact member-generation provenance.
- [x] Explicit relation proposals and terminal human review decisions.
- [x] Reviewer policy, scoped grants, expiry and proposer/reviewer separation.
- [x] Deterministic authorization receipts and crash recovery.
- [x] Process-owned and HMAC-signed reviewer actor bindings.
- [x] One signed assertion per deterministic terminal decision.
- [x] Governed relation publication with authorization provenance.
- [x] Durable graph-set publication attempts and compensation.
- [x] Publication operational audit and retention planning.
- [x] Signed actor-use aggregate provenance in reviewed relation metadata.
- [x] Isolated authorization-only and signed publication journals.
- [x] Transition audit and expired-duplicate retirement preflight.
- [x] Crash-recoverable expired weaker-publication retirement saga.
- [x] Isolated third retirement journal and exact weaker-lease takeover.
- [x] Retirement operational audit and conservative retention planning.
- [x] Deterministic text-free retirement snapshot export and verification.
- [x] Read-only snapshot restore preflight against an initialized target.
- [x] Crash-recoverable terminal snapshot restore into an initialized empty target.
- [x] Fourth isolated restore-intent journal with exact snapshot/target scope.
- [x] Atomic all-or-none target insertion and exact replay recovery.
- [x] Final target write lock across verification and restore-intent completion.

## Completed GraphRAG path

- [x] Bounded authoritative lexical seed selection.
- [x] Explicit within-document and reviewed cross-document edge expansion.
- [x] Generation, graph-set and provenance validation.
- [x] Path-aware selected-node and traversal lineage.
- [x] Strict node/document/edge/path/lineage/abstention evaluation.
- [x] Query-digest-only benchmark fixtures.
- [x] Paired regression policy with floors and non-inferiority margins.
- [x] Live authoritative benchmark bridge.
- [x] Resumable benchmark run store.
- [x] Governed historical baseline registry.
- [x] Canonical `tools.models.Citation` conversion.
- [x] Privacy-hardened graph citation metadata.
- [x] Authoritative `search_evidence_graph` tool boundary.
- [x] Owner-scoped current graph-set discovery tool.
- [x] Lazy agent registration and import-order recovery.
- [x] Existing citation registry, deduplication, relabeling and API serialization.
- [x] Safe browser DOM propagation and local-link refusal contracts.

## Current implementation priorities

### Exact-current verification

- [ ] Obtain a complete unchanged checkout of the current `main` head.
- [ ] Run complete repository pytest and coverage.
- [ ] Run Ruff and full-tree compilation.
- [ ] Execute production live-agent, FastAPI and frontend regressions.
- [ ] Run Docker/Compose persistence and restart tests.
- [ ] Run Windows path, permission and reparse-point tests.

### Distributed execution and fault injection

- [ ] Add database-backed or distributed leadership for periodic graph jobs.
- [ ] Test independent-process publication, retirement and restore contention.
- [ ] Inject process kills at every publication, retirement and restore phase.
- [ ] Inject SQLite busy/locked, WAL, I/O error and disk-full failures.
- [ ] Test graph-set pointer races with independent processes.
- [ ] Test long-running lease renewal under real wall-clock delays.
- [ ] Test restore-target writers blocked by final exact-target completion locks.

### Backup, restore and retention governance

- [x] Read-only retirement snapshot export.
- [x] Descriptor-safe offline snapshot verification.
- [x] Read-only restore preflight.
- [x] Crash-recoverable empty-target restore executor.
- [x] Terminal-only source snapshot enforcement.
- [x] No-overwrite and no-merge restore semantics.
- [x] Recovery after target commit before intent-phase persistence.
- [x] Recovery after target phase persistence before completion.
- [ ] Add asymmetric or hardware-backed audit snapshot signatures.
- [ ] Add trusted timestamps and signer key rotation.
- [ ] Add durable legal-hold registry and authorization.
- [ ] Add signed chain-of-custody manifests.
- [ ] Add mandatory backup-before-restore evidence and post-restore comparison receipts.
- [ ] Add destructive-retention authorization and deletion journal.
- [ ] Add secure deletion and database compaction policy.

### Scientific graph quality

- [ ] Add reviewed scientific claim extraction adapters.
- [ ] Add reviewed entity normalization and resolution adapters.
- [ ] Add reviewed citation-link extraction adapters.
- [ ] Add correction lineage for extractor and reviewer changes.
- [ ] Add semantic claim-support and entailment evaluation.
- [ ] Add contradiction adjudication workflows.
- [ ] Add scientific dataset cards and extraction-quality gates.

### GraphRAG evaluation and connected execution

- [ ] Execute current and historical GraphRAG benchmarks on representative corpora.
- [ ] Add connected-provider discovery-before-search behavior tests.
- [ ] Add measured latency, memory, storage and cost observations.
- [ ] Add repeated-seed and bootstrap confidence analyses where appropriate.
- [ ] Add larger path-completeness and cross-document-support datasets.
- [ ] Add human review of GraphRAG failure clusters.

### Identity and review governance

- [ ] Add asymmetric reviewer assertions and key IDs.
- [ ] Integrate external IAM/OIDC or directory-backed identity.
- [ ] Add issuer key rotation and overlap windows.
- [ ] Add hardware-backed signing.
- [ ] Add multi-party or quorum review thresholds.
- [ ] Add reviewer agreement and disagreement reporting.
- [ ] Add signed review/export manifests and external transparency records.

## Permanent boundaries

- The evidence graph remains derived and rebuildable, not a fifth authoritative document store.
- Cross-document semantic relations remain explicit and reviewed; no automatic approval is performed.
- Cryptographic actor assertions prove configured key possession, not scientific correctness.
- GraphRAG retrieval does not itself generate an answer.
- An authorization-only graph set cannot be retroactively relabeled as signed provenance.
- Snapshot integrity digests are not digital signatures.
- Restore execution accepts only terminal snapshots and an initialized globally empty target.
- Restore never overwrites, merges or deletes target history.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
