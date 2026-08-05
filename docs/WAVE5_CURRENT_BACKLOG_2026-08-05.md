# Wave 5 current implementation backlog

Last updated: 2026-08-05

This file supersedes `WAVE5_CURRENT_BACKLOG_2026-08-04.md`. Earlier ledgers remain historical evidence of the implementation sequence.

## Completed evidence-graph and GraphRAG foundation

The committed Wave 5 foundation includes:

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

The restore-governance stack includes:

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
- fresh signed-actor replay after quarantine creation without weakening quarantine scope;
- read-only permit-recovery receipt/quarantine audit and conservative retention planning.

## Completed reviewed scientific-claim foundation

- [x] Closed-schema scientific-claim extractor output.
- [x] Strict duplicate-key, NaN/Infinity and unknown-field refusal.
- [x] Generation, content-hash and profile-fingerprint binding.
- [x] Exact section/page/character provenance validation.
- [x] Server-computed evidence-span SHA-256.
- [x] Bounded scientific claim taxonomy and modality vocabulary.
- [x] Deterministic proposal and extraction-batch digests excluding timestamps.
- [x] Immutable SQLite proposal, decision and authorization storage.
- [x] Database/path/payload/column integrity checks.
- [x] Correction lineage with one successor and same-generation scope.
- [x] Canonical predecessor-first same-batch submission independent of caller order.
- [x] Process-owned reviewer actor binding.
- [x] Owner/document/decision-scoped reviewer policy and expiry.
- [x] Proposer–reviewer and correction-author–reviewer separation.
- [x] Atomic terminal decision plus authorization insertion.
- [x] Stable exact replay preserving original review timestamps.
- [x] Predecessor supersession required before corrected-claim approval.
- [x] Conversion of exact authorized approvals to existing `GraphAnnotation` claim nodes.
- [x] Obsolete approved-claim refusal after an approved correction.
- [x] No automatic graph write or semantic relation inference.
- [x] Privacy-conscious status/list/decision/annotation CLI.
- [x] Deterministic extraction precision/recall/F1 evaluation.
- [x] Exact evidence and locator accuracy.
- [x] Span-IoU, claim token-F1, type, modality and confidence-calibration metrics.
- [x] Text-free digest-bound evaluation report verification.
- [x] Descriptor-safe strict local evaluation fixture CLI.
- [x] Owner-scoped immutable exact-version extractor registry.
- [x] Model/rule extractor kinds with exact implementation/configuration/schema digests.
- [x] Capability scopes for claim types, modalities and languages.
- [x] Separate extractor-administrator policy and expiry.
- [x] Process-actor-bound registration and monotonic retirement.
- [x] No reactivation of retired extractor versions.
- [x] Exact record-digest retirement confirmation.
- [x] Canonical registered extraction for model and rule provenance.
- [x] Registry provenance embedded into deterministic claim proposals.
- [x] Credential-, prompt-, response- and source-text-free registry CLI.

## Current implementation priorities

### Exact-current verification

- [ ] Obtain a complete unchanged checkout of the current `main` head.
- [ ] Run complete repository pytest and coverage.
- [ ] Run Ruff and full-tree compilation.
- [ ] Execute production live-agent, FastAPI and frontend regressions.
- [ ] Run Docker/Compose persistence and restart tests.
- [ ] Run Windows path, permission and reparse-point tests.
- [ ] Execute all scientific-claim and extractor-registry repository-native contracts together.

### Distributed execution and fault injection

- [ ] Add database-backed or distributed leadership for periodic graph jobs.
- [ ] Test independent-process publication, retirement, restore, hold, permit-recovery and deletion contention.
- [ ] Test independent-process claim proposal, correction and terminal-review contention.
- [ ] Test independent-process extractor registration/retirement contention.
- [ ] Inject process kills around claim proposal submission and decision/authorization commit.
- [ ] Inject process kills around extractor registration and retirement compare-and-swap.
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
- [x] Governed audit and retention planning for permit-recovery receipts and quarantine holds.
- [ ] Add HSM/KMS-backed private-key operations and governed key-generation ceremonies.
- [ ] Add externally distributed signer certificates, directory records or transparency logs.
- [ ] Add secure physical-erasure and database-compaction policy.
- [ ] Add platform-specific evidence for SQLite page, WAL, backup, filesystem-snapshot and media erasure.

### Scientific graph quality

- [x] Add reviewed scientific claim extraction foundation.
- [x] Add governed exact-version model/rule extractor registry and execution boundary.
- [ ] Add actual production model/rule extractor implementations.
- [ ] Add governed extractor benchmark promotion and rollback reports.
- [ ] Add active-version selection/promotion pointers while retaining exact-version execution.
- [ ] Add deprecation reasons, compatibility windows and migration planning.
- [ ] Add reviewed entity normalization and resolution proposals.
- [ ] Add reviewed citation-link extraction proposals.
- [ ] Add reviewed method and dataset extraction proposals.
- [ ] Add correction lineage across extractor upgrades and reviewer changes.
- [ ] Add semantic claim-support and entailment evaluation.
- [ ] Add explicit support/contradiction proposal and adjudication workflows.
- [ ] Add inter-reviewer agreement and disagreement reports for claims.
- [ ] Add scientific dataset cards and extraction-quality gates.
- [ ] Add multilingual scientific-claim extraction and normalization evaluation.
- [ ] Add automatic derived-graph job input only after reviewed proposal publication is coordinated with authoritative generation reconciliation.

### GraphRAG evaluation and connected execution

- [ ] Execute current and historical GraphRAG benchmarks on representative corpora.
- [ ] Add connected-provider discovery-before-search behavior tests.
- [ ] Add measured latency, memory, storage and cost observations.
- [ ] Add repeated-seed and bootstrap confidence analyses where appropriate.
- [ ] Add larger path-completeness and cross-document-support datasets.
- [ ] Add human review of GraphRAG failure clusters.
- [ ] Benchmark GraphRAG with reviewed claims versus structural-only graphs.
- [ ] Add claim-correction-aware retrieval regression fixtures.

### Identity and review governance

- [ ] Add asymmetric reviewer assertions and governed key IDs.
- [ ] Integrate external IAM/OIDC or directory-backed identity.
- [ ] Add issuer key rotation and overlap windows.
- [ ] Add hardware-backed signing.
- [ ] Add multi-party or quorum review thresholds.
- [ ] Add reviewer agreement and disagreement reporting.
- [ ] Add public-key-signed review/export manifests and external transparency records.
- [x] Add separately configurable extractor-administrator and claim-reviewer roles.
- [ ] Add separately governed semantic-relation adjudicator roles.

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
- Scientific extractor output is a proposal, not a graph fact.
- Extractor registration identifies governed bytes/configuration and capabilities; it is not benchmark promotion or proof of scientific quality.
- Claim approval is a governed review record, not independent scientific verification.
- Claim detection/span metrics are not semantic entailment.
- Reviewed claim annotations do not imply support, contradiction, causality or truth.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
