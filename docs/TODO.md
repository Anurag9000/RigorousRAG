# Exhaustive implementation TODO

This backlog is ordered by dependency and risk. The checked Wave 1 items are implemented; all other entries remain active until code, tests, documentation and verification are committed together.

## Wave 1 — completed foundation

- [x] Hybrid candidate ranking and score traces.
- [x] BM25 candidate-pool scoring.
- [x] RRF and weighted fusion.
- [x] MMR and source caps.
- [x] Heuristic and optional cross-encoder rerankers.
- [x] Uploaded-document controls with dense compatibility defaults.
- [x] BEIR loader and retrieval/citation metrics.
- [x] Deterministic manifests and resumable result store.
- [x] Offline baseline CLI and focused tests.

## Wave 2A — model registry and sparse-store contracts

- [ ] Implement `EmbeddingProfile` validation, canonical aliases and SHA-256 fingerprints.
- [ ] Add profiles for MiniLM, E5, BGE, GTE, Instructor, SPECTER2 and BGE-M3.
- [ ] Reject unknown/duplicate profile fields, invalid dimensions, controls and non-standard numbers.
- [ ] Implement fielded sparse schema with owner/document/generation isolation.
- [ ] Add positions, page, section, field and metadata provenance.
- [ ] Implement transactional replace, snapshot, restore, delete and corruption refusal.
- [ ] Add symlink/reparse/database-identity protections.
- [ ] Add field-weighted BM25 and exact document filters.
- [ ] Test owner isolation, field weighting, rollback, snapshots, path replacement and corrupt rows.

## Wave 2B — cross-store coordination

- [ ] Add public vector generation snapshot/restore APIs.
- [ ] Add sparse generation snapshot/restore APIs.
- [ ] Coordinate vector then sparse writes under one document lock.
- [ ] Restore both prior generations after any failure.
- [ ] Record generation/profile fingerprints and content hashes.
- [ ] Wire API ingestion, batch ingestion and retry recovery.
- [ ] Coordinate deletion before registry/source cleanup.
- [ ] Add reconciliation scans for vector-only, sparse-only and registry-only documents.
- [ ] Add repair commands with dry-run and immutable audit output.

## Wave 2C — corpus-level hybrid search

- [ ] Generate dense and sparse candidates independently.
- [ ] Fuse by RRF and calibrated weighted scores.
- [ ] Add field filters, date filters, document filters and source caps.
- [ ] Add reranker cascades with latency/cost budgets.
- [ ] Return complete component scores, generations and profile fingerprints.
- [ ] Add migration/reindex CLI with resume, shadow validation and cutover.

## Waves 3–10

- [ ] Adaptive/corrective RAG and evidence sufficiency.
- [ ] Query decomposition, routing and bounded multi-hop planning.
- [ ] Evidence graph and GraphRAG-derived artifacts.
- [ ] Layout/table/formula/figure/OCR multimodal ingestion.
- [ ] Scientific methods/results/claim/evidence schemas.
- [ ] Dense, sparse-neural, late-interaction and listwise model interfaces.
- [ ] Comprehensive public/domain/adversarial datasets and dataset cards.
- [ ] Repeated experiment matrices, statistics and regression history.
- [ ] Observability, benchmark dashboards and failure diagnostics.
- [ ] Distributed queues, SQL registry, outbox/saga and object storage.
- [ ] External deployment controls: TLS, egress, secrets, malware scanning and parser sandbox.

## Required verification before release claims

- [ ] Dependency installation and `pip check`.
- [ ] Whitespace and generated-artifact checks.
- [ ] Python compilation.
- [ ] Fatal Ruff checks and configured lint policy.
- [ ] Full pytest and measured branch coverage on Python 3.10, 3.11 and 3.12.
- [ ] Windows compatibility tests.
- [ ] Docker Compose validation.
- [ ] Docker image build and health/readiness smoke tests.
- [ ] Clean-clone CLI/API ingestion and retrieval smoke tests.
- [ ] Concurrency, fault-injection and rollback tests.
- [ ] Final line-by-line regression audit of one unchanged exact `main` SHA.
