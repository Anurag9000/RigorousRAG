# Wave 5 current implementation backlog — revision 2

Last updated: 2026-08-05

This file supersedes `WAVE5_CURRENT_BACKLOG_2026-08-05.md`. Earlier Wave 5 ledgers remain historical evidence of the implementation sequence.

## Completed evidence-graph and GraphRAG foundation

- [x] Generation-scoped typed evidence graphs and tombstones.
- [x] Immutable graph generations and exact authoritative readers.
- [x] Durable graph-generation jobs, leases, reconciliation, audit and retention planning.
- [x] Cross-document graph sets with exact member-generation provenance.
- [x] Explicit relation proposals and governed reviewer decisions.
- [x] Process-owned and signed reviewer actors, authorization receipts and separation of duties.
- [x] Governed immediate and durable signed graph-set publication.
- [x] Publication journal isolation, transition audit and crash-recoverable weaker-attempt retirement.
- [x] Bounded authoritative GraphRAG selection and path lineage.
- [x] GraphRAG evaluation, live benchmark bridge, resumable runs and governed baselines.
- [x] Canonical citation conversion, graph-set discovery and agent registration.
- [x] Existing API serialization, citation registry and safe browser propagation.

## Completed restore, custody and deletion governance

- [x] Deterministic retirement snapshots and descriptor-safe verification.
- [x] Read-only restore preflight and crash-recoverable empty-target restore.
- [x] Restore-intent operational audit and conservative retention planning.
- [x] Integrity-backed legal holds and hold-placement permits.
- [x] Pre/post restore receipts, custody manifests and backup artifacts.
- [x] HMAC and Ed25519 custody envelopes and RFC 3161 timestamp verification.
- [x] Expiring deletion authorization and single-use reservation.
- [x] Crash-recoverable logical deletion and deletion audit.
- [x] Governed stale hold-permit recovery and quarantine holds.
- [x] Immutable recovery receipts and quarantine retention planning.

## Completed reviewed scientific-claim foundation

### Extraction and proposals

- [x] Closed-schema scientific-claim extractor output.
- [x] Duplicate-key, unknown-field and NaN/Infinity refusal.
- [x] Generation, content-hash and profile-fingerprint binding.
- [x] Exact section/page/character provenance validation.
- [x] Server-computed evidence-span SHA-256.
- [x] Bounded claim taxonomy and modality vocabulary.
- [x] Deterministic proposal/batch identities excluding timestamps.
- [x] Immutable proposal, decision and authorization storage.
- [x] Database/path/payload/column integrity checks.
- [x] One-successor correction lineage and same-generation correction scope.
- [x] Canonical predecessor-first atomic batch submission.

### Governed review and graph conversion

- [x] Owner/document/decision-scoped claim-review policy and expiry.
- [x] Process-owned reviewer actor binding.
- [x] Proposer–reviewer and correction-author–reviewer separation.
- [x] Atomic terminal decision plus authorization insertion.
- [x] Stable exact replay preserving original review timestamps.
- [x] Predecessor supersession before corrected-claim approval.
- [x] Exact approved/auth-backed conversion to existing claim `GraphAnnotation` objects.
- [x] Obsolete approved-claim refusal after approved correction.
- [x] No automatic graph write or semantic relation inference.
- [x] Privacy-conscious proposal/review/annotation operator CLI.

### Evaluation

- [x] Extraction precision, recall and F1.
- [x] Exact evidence and locator accuracy.
- [x] Span-IoU and claim token-F1.
- [x] Claim-type and modality accuracy.
- [x] Confidence Brier score.
- [x] Deterministic one-to-one matching.
- [x] Text-free report digests and independent verification.
- [x] Descriptor-safe strict local evaluation fixture CLI.

## Completed governed extractor lifecycle

### Exact-version registry

- [x] Owner-scoped immutable model/rule extractor versions.
- [x] Exact implementation, configuration and output-schema digests.
- [x] Claim-type, modality and language capability scopes.
- [x] Separate extractor-administrator policy and expiry.
- [x] Process-actor-bound registration and monotonic retirement.
- [x] No reactivation of retired exact versions.
- [x] Exact record-digest retirement confirmation.
- [x] Canonical registered execution for model and rule provenance.
- [x] Registry provenance embedded into deterministic claim proposals.
- [x] Credential-, prompt-, response- and source-text-free registry operations.

### Benchmark aggregation and promotion

- [x] Verified evaluation-report to benchmark-case conversion.
- [x] Every benchmark proposal bound to the exact registry record digest.
- [x] Unique dataset/case identities and cross-version refusal.
- [x] Deterministic micro and weighted aggregate benchmark suite metrics.
- [x] Separate promotion policy with metric floors/ceilings and administrator scope.
- [x] Immutable eligible/ineligible promotion reports and reason codes.
- [x] Separate promotion database from the immutable extractor registry.
- [x] Append-only activation history and one current pointer per owner/extractor name.
- [x] Exact expected-current pointer confirmation.
- [x] Timestamp-stable report and activation replay.
- [x] Recovery after activation-row or pointer publication.
- [x] Active exact-version resolution through registry and eligible report revalidation.
- [x] Append-only rollback to an earlier eligible active exact version.
- [x] Retired rollback-target refusal.
- [x] Descriptor-safe text-free benchmark suite CLI.
- [x] Text-free assess/promote/current/history/resolve/rollback operator surface.

## Current implementation priorities

### Exact-current verification

- [ ] Obtain a complete unchanged checkout of current `main`.
- [ ] Run complete repository pytest and coverage.
- [ ] Run Ruff and full-tree compilation.
- [ ] Execute all scientific-claim, registry, benchmark and promotion contracts together.
- [ ] Execute production live-agent, FastAPI and frontend regressions.
- [ ] Run Docker/Compose persistence and restart tests.
- [ ] Run Windows path, permission and reparse-point tests.

### Distributed execution and fault injection

- [ ] Add database-backed/distributed leadership for periodic graph jobs.
- [ ] Test independent-process claim proposal, review and correction contention.
- [ ] Test independent-process extractor registration, promotion, rollback and retirement contention.
- [ ] Inject process kills around proposal, decision/authorization, registration and activation transactions.
- [ ] Inject SQLite busy/locked, WAL, I/O-error and disk-full failures.
- [ ] Test long-running lease renewal and graph-set pointer races under real processes.
- [ ] Execute the complete restore/hold/deletion/custody fault matrix on supported platforms.

### Scientific graph quality

- [ ] Add actual production model and rule extractor implementations.
- [ ] Add governed provider adapter registration without credential persistence.
- [ ] Add benchmark dataset manifests, cards, provenance and gold-review governance.
- [ ] Add historical promotion comparison and non-inferiority confidence intervals.
- [ ] Add deprecation reasons, compatibility windows and migration planning.
- [ ] Add reviewed entity normalization/resolution proposals.
- [ ] Add reviewed citation-link extraction proposals.
- [ ] Add reviewed method and dataset extraction proposals.
- [ ] Add semantic claim-support and entailment evaluation.
- [ ] Add explicit support/contradiction proposal and adjudication workflows.
- [ ] Add inter-reviewer agreement and disagreement reports.
- [ ] Add multilingual claim extraction and normalization evaluation.
- [ ] Coordinate reviewed claim publication with authoritative-generation graph reconciliation.

### GraphRAG evaluation and connected execution

- [ ] Execute current and historical GraphRAG benchmarks on representative corpora.
- [ ] Benchmark reviewed-claim graphs against structural-only graphs.
- [ ] Add claim-correction-aware retrieval regression fixtures.
- [ ] Add connected-provider discovery-before-search tests.
- [ ] Measure latency, memory, storage and provider cost.
- [ ] Add repeated-seed/bootstrap analyses and human review of failure clusters.

### Identity and transparency

- [ ] Add asymmetric reviewer/administrator assertions and governed key IDs.
- [ ] Integrate external IAM/OIDC or directory-backed identity.
- [ ] Add hardware-backed signing and key rotation.
- [ ] Add multi-party/quorum scientific review.
- [ ] Add signed registry, benchmark, promotion and review exports.
- [ ] Add externally verifiable transparency records.

## Permanent boundaries

- The evidence graph remains derived and rebuildable, not an authoritative document store.
- Extractor output is a proposal, not a graph fact.
- Extractor registration identifies governed bytes/configuration and capabilities; it is not quality proof.
- Promotion means one exact benchmark suite passed one exact policy; it is not proof of general scientific correctness.
- Claim approval is a governed review record, not independent replication.
- Claim detection/span metrics are not semantic entailment.
- Reviewed claim annotations do not imply support, contradiction, causality or truth.
- Promotion never approves proposals or publishes graph nodes.
- Rollback never reactivates a retired version or rewrites activation history.
- Cross-document semantic relations remain explicit and reviewed.
- GraphRAG retrieval does not itself generate an answer.
- Restore never overwrites or merges target history.
- Logical database-row deletion is not secure physical erasure.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
