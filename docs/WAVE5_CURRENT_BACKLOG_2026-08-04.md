# Wave 5 current implementation backlog

Last updated: 2026-08-04

This file supersedes `WAVE5_CURRENT_BACKLOG_2026-08-02.md`. The earlier file remains historical evidence of the implementation sequence.

## Completed evidence-graph and GraphRAG foundation

The committed Wave 5 foundation now includes:

- generation-scoped typed evidence graphs, tombstones, immutable graph generations and exact authoritative readers;
- durable graph-generation jobs, leases, reconciliation, audit and retention planning;
- cross-document graph sets with exact member-generation provenance;
- explicit relation proposals, governed human decisions and proposer/reviewer separation;
- process-owned and signed reviewer actors, authorization receipts and one-assertion-per-decision enforcement;
- governed immediate and durable signed graph-set publication;
- authorization-only/signed journal isolation, transition audit and crash-recoverable weaker-attempt retirement;
- bounded authoritative GraphRAG selection, evaluation, live benchmarks, resumable runs and governed baselines;
- canonical citation conversion, graph-set discovery, agent registration, API serialization and safe browser propagation.

## Completed restore, custody and deletion governance

The restore-governance stack now includes:

- deterministic retirement snapshots and descriptor-safe offline verification;
- read-only restore preflight and crash-recoverable terminal-history restore into an initialized empty target;
- restore-intent audit, conservative retention planning and integrity-backed legal holds;
- pre/post restore receipts, backup artifacts, durable custody manifests and chain-of-custody exports;
- HMAC and Ed25519 custody envelopes, governed key history and RFC 3161 timestamp verification;
- expiring deletion authorization, single-use reservation and crash-recoverable logical deletion;
- deletion markers/tombstones, hold/deletion serialization and deletion operational audit;
- durable hold-placement permits and read-only permit diagnostics;
- governed age-gated recovery of stale hold-placement permits;
- active quarantine holds before releasing permits that lack a committed original hold;
- active-original-hold exact-replay refusal and released-hold cleanup;
- immutable permit-recovery receipts with exact original-digest replay confirmation;
- fresh signed-actor replay after quarantine creation without weakening quarantine scope.

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
- [ ] Test independent-process publication, retirement, restore, hold, permit-recovery and deletion contention.
- [ ] Inject process kills at every publication, retirement, restore, hold-permit, permit-recovery and deletion phase.
- [ ] Inject SQLite busy/locked, WAL, I/O-error and disk-full failures.
- [ ] Test graph-set pointer races with independent processes.
- [ ] Test long-running lease renewal under real wall-clock delays.
- [ ] Test independent-process legal-hold placement/release contention.
- [ ] Test quarantine creation followed by process death before permit/receipt commit.
- [ ] Test authorization revocation versus reservation/consumption contention.
- [ ] Test concurrent custody artifact and receipt publication races.
- [ ] Test independent-process signer-key registration/retirement and signature-output races.

### Backup, restore and retention governance

- [x] Governed recovery for active hold-placement permits without a committed active original hold.
- [x] Missing-hold quarantine before permit release.
- [x] Process-owned exact-confirmation recovery receipt.
- [x] Fresh actor replay of an already committed quarantine hold.
- [x] Exact original permit-digest confirmation on completed recovery replay.
- [ ] Add HSM/KMS-backed private-key operations and governed key-generation ceremonies.
- [ ] Add externally distributed signer certificates, directory records or transparency logs.
- [ ] Add secure physical-erasure and database-compaction policy.
- [ ] Add platform-specific evidence for SQLite page, WAL, backup, filesystem-snapshot and media erasure.
- [ ] Add governed retention and archival policy for permit-recovery receipts and quarantine holds.

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

- [ ] Add asymmetric reviewer assertions and governed key IDs.
- [ ] Integrate external IAM/OIDC or directory-backed identity.
- [ ] Add issuer key rotation and overlap windows.
- [ ] Add hardware-backed signing.
- [ ] Add multi-party or quorum review thresholds.
- [ ] Add reviewer agreement and disagreement reporting.
- [ ] Add public-key-signed review/export manifests and external transparency records.

## Permanent boundaries

- The evidence graph remains derived and rebuildable, not a fifth authoritative document store.
- Cross-document semantic relations remain explicit and reviewed; no automatic approval is performed.
- GraphRAG retrieval does not itself generate an answer.
- An authorization-only graph set cannot be retroactively relabeled as signed provenance.
- Restore never overwrites or merges target history.
- Legal holds protect retention and are serialized against deletion; they do not authorize deletion.
- Permit recovery never releases an active original hold.
- Missing original holds are replaced by an active quarantine before permit release.
- A quarantine hold requires separate review and explicit release.
- Logical database-row deletion is not secure physical erasure or SQLite page reclamation.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
