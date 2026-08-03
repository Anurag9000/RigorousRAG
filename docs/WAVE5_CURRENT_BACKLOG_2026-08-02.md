# Wave 5 current implementation backlog

Last updated: 2026-08-03

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
- [x] Restore-intent operational audit and lease/exhaustion classification.
- [x] Conservative restore-intent retention planning.
- [x] Integrity-backed durable restore legal-hold registry.
- [x] Process-owned hold placement/release and monotonic hold lifecycle.
- [x] Read-only durable-hold integration with retention planning.
- [x] Process-owned pre-restore SQLite backup receipts.
- [x] Process-owned post-restore exact-comparison receipts.
- [x] Two-connection nonblocking SQLite backup under a write-reservation guard.
- [x] Atomic no-overwrite backup and receipt artifact publication.
- [x] Backup/schema/count/hash verification and receipt reconstruction.
- [x] Durable pre/post restore custody manifest with replay-stable actor provenance.
- [x] Custody evidence enforcement in canonical restore seed/execute/reconcile paths.
- [x] Custody-manifest query-only operational audit.
- [x] Conservative custody-manifest retention planning with durable legal holds.
- [x] Durable intent before pre-restore backup/receipt artifact publication.
- [x] Lease-based recovery after paired artifact publication.
- [x] Immutable backup-only, receipt-only, and collision orphan classification.
- [x] Completed artifact-pair live revalidation and tamper refusal.
- [x] Query-only artifact status and listing.
- [x] Artifact operational audit and conservative retention planning.
- [x] Durable restore-hold protection for artifact records.
- [x] Deterministic complete external restore chain-of-custody manifest.
- [x] Live target, pre/post receipt, custody, artifact-path, and chronology validation.
- [x] Actor-ID reduction and raw-path-free external custody payloads.
- [x] Descriptor-safe offline chain verification.
- [x] Optional protected-key HMAC-SHA256 custody envelope with key-ID pinning.

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
- [ ] Test independent-process legal-hold placement/release contention.
- [ ] Test concurrent custody artifact and receipt publication races.
- [ ] Test independent-process custody binding and post-finalization contention.
- [ ] Test independent-process artifact-journal lease and output-path races.
- [ ] Test concurrent external custody export and live artifact replacement races.

### Backup, restore and retention governance

- [x] Read-only retirement snapshot export.
- [x] Descriptor-safe offline snapshot verification.
- [x] Read-only restore preflight.
- [x] Crash-recoverable empty-target restore executor.
- [x] Terminal-only source snapshot enforcement.
- [x] No-overwrite and no-merge restore semantics.
- [x] Recovery after target commit before intent-phase persistence.
- [x] Recovery after target phase persistence before completion.
- [x] Restore-intent operational audit.
- [x] Non-destructive restore-intent retention planning.
- [x] Durable legal-hold registry and process-owned authorization.
- [x] Integrity-verified active-hold retention integration.
- [x] Pre-restore empty-target SQLite backup receipt.
- [x] Post-restore exact-comparison receipt.
- [x] Offline custody receipt and backup verification.
- [x] Durably bind one pre-receipt/backup pair to each restore intent.
- [x] Require bound custody evidence in canonical execute/reconcile paths.
- [x] Durable custody manifest and post-receipt finalization state.
- [x] Custody-manifest operational audit and conservative retention planning.
- [x] Track paired and orphaned backup/receipt publication outcomes durably.
- [x] Preserve orphan evidence without automatic deletion or overwrite.
- [x] Artifact audit and retention planning with orphan permanence.
- [x] Complete external chain-of-custody export and offline verification.
- [x] Shared-secret HMAC authentication envelope with explicit key ID.
- [ ] Add asymmetric or hardware-backed audit/custody signatures.
- [ ] Add trusted timestamps and signer key rotation.
- [ ] Add public-key signed external chain-of-custody export.
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
- [ ] Add public-key signed review/export manifests and external transparency records.

## Permanent boundaries

- The evidence graph remains derived and rebuildable, not a fifth authoritative document store.
- Cross-document semantic relations remain explicit and reviewed; no automatic approval is performed.
- Cryptographic actor assertions prove configured key possession, not scientific correctness.
- GraphRAG retrieval does not itself generate an answer.
- An authorization-only graph set cannot be retroactively relabeled as signed provenance.
- Snapshot and custody integrity digests are not digital signatures.
- HMAC custody envelopes prove shared-secret possession, not public non-repudiation.
- Restore execution accepts only terminal snapshots and an initialized globally empty target.
- Restore never overwrites, merges or deletes target history.
- Legal holds protect retention planning but do not authorize deletion or restore mutation.
- Custody manifests bind evidence to execution but do not authorize deletion.
- Artifact orphan classifications preserve evidence and never authorize cleanup.
- External custody exports are evidence-only and cannot import or mutate state.
- Retention candidates are not deletion authorization.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
